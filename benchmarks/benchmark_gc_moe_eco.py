"""
plain / EPLB / oracle / ECO on recorded routing, timed as benchmarks/benchmark_gc_moe.py does: one call captured in a
CUDA graph, inner_iters replays per event window, L2 flushed and ranks barriered before every window, per-rank
median over windows, slowest rank reported.

Modes (--mode NAME:KIND:P[:FILE], repeatable):
  plain : P = 384 logical experts, --off routing
  eplb  : P physical slots, FILE = EPLB physical routing (slot -> expert map unknown: weights by slot % 384)
  ora   : P physical slots, FILE = oracle physical routing (zero-overhead ECO); --phy2log gives the slot map
  eco   : P physical slots, FILE = EcoMeta (.npy, [layers, words]; FILE_phy2log.npy), --off logical routing
Calls: --calls evenly spaced indices of the routing files; call i uses layer i % layers.
--cold-l2: every replay first overwrites L2 (memset of 2x L2 bytes inside the graph); a flush-only graph is timed
the same way and its per-rank median is subtracted (no other layer runs between MoE calls in this harness).
--check: on the first sampled call, compare every ora / eco mode's output with plain (same per-expert weights).
Shared experts are fused as in benchmarks/benchmark_gc_moe_se.py (default DSV4.1-Flash preset: 1 shared expert,
shared weight K granularity 32 -> weight recipe (1, 32, 32), activation recipe (1, 1, 32)); --num-shared-experts 0
times routed experts only. --block-m pins block M for every mode (the shared-input SF layout follows it).

Usage: torchrun --nproc_per_node=8 benchmarks/benchmark_gc_moe_eco.py --off routing_off.npy --T 128 \
    --mode plain384:plain:384 --mode eco448:eco:448:eco448_metas.npy ...
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import deep_gemm   # noqa: F401
except ImportError:   # hosts without a DeepGEMM build: torch-only casts / SF layout (gcbench dg_quant.py on PYTHONPATH)
    import dg_quant
    dg_quant.install()
import benchmark_gc_moe as base   # noqa: E402  (imports deep_gemm / gc_moe from this repo, timing helpers)

import numpy as np   # noqa: E402
import torch   # noqa: E402
import torch.distributed as dist   # noqa: E402

deep_gemm, gc_moe = base.deep_gemm, base.gc_moe
HIDDEN, INTER, TOPK, E = 5120, 2304, 6, 384


def expert_weights(logical_ids, device):
    """FP4 W1/W2 per physical slot; a slot's weights depend only on its logical expert (copies are identical)."""
    def cast(n, k, salt):
        w = torch.empty((len(logical_ids), n, k // 2), device=device, dtype=torch.int8)
        w_sf = torch.empty((len(logical_ids), n, k // 32), device=device, dtype=torch.float)
        for i, e in enumerate(logical_ids):
            gen = torch.Generator(device=device).manual_seed(7 + 2 * int(e) + salt)
            bf16 = torch.randn((n, k), device=device, dtype=torch.bfloat16, generator=gen) * 0.05
            w[i], w_sf[i] = base.per_token_cast_to_fp4(bf16, use_ue8m0=True, gran_k=32)
        return w, deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), len(logical_ids))
    return gc_moe.transform_weights_for_gc_moe(cast(2 * INTER, HIDDEN, 0), cast(HIDDEN, INTER, 1), activation="swiglu")


def shared_weights(n_shared, gran_k, device):
    """Fused shared experts as tests/test_gc_moe_se.py builds them (seed 11, FP8 weights, block or per-token SF)."""
    from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
    si = INTER * n_shared
    gen = torch.Generator(device=device).manual_seed(11)
    raw = (torch.randn((2 * si, HIDDEN), dtype=torch.bfloat16, device=device, generator=gen) * 0.05,
           torch.randn((HIDDEN, si), dtype=torch.bfloat16, device=device, generator=gen) * 0.05)
    out = []
    for w in raw:
        n, k = w.shape
        if gran_k == 128:
            q, sf = per_block_cast_to_fp8(w, use_ue8m0=True, gran_k=128)
            recipe = (128, 128)
        else:   # (1, 32, 32) blocks, expanded to per-row (1, 32) scales
            q, sf = per_block_cast_to_fp8(w, use_ue8m0=True, gran_k=32)
            sf = sf.repeat_interleave(32, dim=0)[:n]
            recipe = (1, 32)
        out.append((q, deep_gemm.transform_sf_into_required_layout(sf, n, k, recipe, num_groups=None)))
    return gc_moe.transform_weights_for_gc_moe(*out, activation="swiglu")


def shared_input_sf(x_sf, like, block_m):
    """tests/test_gc_moe_se.py make_shared_input_sf: token SF rows in the shared-L1 block layout of `block_m`"""
    dst = torch.zeros_like(like)
    t = torch.arange(x_sf.size(0), dtype=torch.long, device=x_sf.device)
    w = t % block_m
    dst[t // block_m * ((block_m + 127) // 128 * 128) + (w // 128) * 128 + (w % 32) * 4 + (w % 128) // 32] = x_sf
    return dst


def eco_metas(path, device):
    """EcoMeta rows with the plan role's static unit tables appended (header[15] = offset), warm-start flag set"""
    from gc_moe.eco.unit_ref import unit_section
    metas_np = np.load(path)
    P, R = int(metas_np[0, 2]), int(metas_np[0, 3])
    out = []
    for m in metas_np:
        m = np.asarray(m, np.int64) & 0xFFFFFFFF
        off = (len(m) + 3) // 4 * 4
        sec = unit_section(m[16 + E * 16:16 + E * 16 + P], R, E).astype(np.int64) & 0xFFFFFFFF
        mm = np.concatenate([m, np.zeros(off - len(m), np.int64), sec])
        mm[15] = off
        mm[14] |= 2
        out.append(torch.from_numpy(np.ascontiguousarray(mm.astype(np.uint32).view(np.int32))).to(device))
    return out, np.load(path.replace(".npy", "_phy2log.npy"))


def align_first_launch():
    """First call of every binary set compiles on each rank at its own pace; finish compiling everywhere before
    launching (combine readiness waits only 60 s per peer)."""
    import gc_moe.jit_kernels.impls.sm100_fp8_fp4_gc_moe as jit_impl
    aligned = set()

    def launch(config, y, l1, l2, sym_buffer, s1, s2, stats, **eco):
        try:   # upstream >= 7902831 needs the world size
            context = jit_impl.get_context(num_ranks=sym_buffer.group.size())
        except TypeError:
            context = jit_impl.get_context()
        binaries, arguments = jit_impl.prepare_role_binaries(config, y, l1, l2, sym_buffer, s1, s2, stats, context, **eco)
        if id(binaries[0]) not in aligned and not torch.cuda.is_current_stream_capturing():
            aligned.add(id(binaries[0]))
            torch.cuda.synchronize()
            dist.barrier()
        context.launch(binaries, arguments)
    jit_impl.launch_gc_moe = launch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="logical routing [calls, ranks, T, topk] (plain / eco)")
    ap.add_argument("--mode", action="append", required=True, help="NAME:KIND:P[:FILE]")
    ap.add_argument("--phy2log", default="", help="ora modes: slot map file pattern with {P}")
    ap.add_argument("--calls", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--inner-iters", type=int, default=20)
    ap.add_argument("--cold-l2", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--num-shared-experts", type=int, default=1)
    ap.add_argument("--shared-weight-gran-k", type=int, choices=[32, 128], default=32)
    ap.add_argument("--block-m", type=int, default=0, help="pin block M for every mode (0: gc-moe heuristic)")
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank, R = dist.get_rank(), dist.get_world_size()
    align_first_launch()
    import gc_moe.apis.gc_moe as gc_api
    if args.block_m:
        gc_api.get_block_m_for_gc_moe = lambda *_a, **_k: args.block_m
    NS = args.num_shared_experts
    shared = shared_weights(NS, args.shared_weight_gran_k, device) if NS else None
    shared_kw = dict(shared_l1_weights=shared[0], shared_l2_weights=shared[1],
                     shared_recipe=(1, 128, 128) if args.shared_weight_gran_k == 128 else (1, 1, 32)) if NS else {}

    off = np.load(args.off, mmap_mode="r")
    calls_total, r_ranks, T, topk = off.shape
    assert r_ranks == R and topk == TOPK, off.shape
    modes = []
    for spec in args.mode:
        name, kind, P, *rest = spec.split(":")
        modes.append((name, kind, int(P), rest[0] if rest else None))
    n_calls = min([calls_total] + [np.load(f, mmap_mode="r").shape[0] for _, k, _, f in modes if k in ("eplb", "ora")])
    call_ids = sorted({int(round(j * (n_calls - 1) / max(args.calls - 1, 1))) for j in range(args.calls)})

    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    x = torch.randn((T, HIDDEN), device=device, dtype=torch.bfloat16, generator=gen) * 0.5
    x_fp8, x_sf = base.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    topk_w = torch.rand((T, TOPK), device=device, generator=gen).softmax(-1)
    base._flush_l2(device)   # allocate the flush buffer outside any capture
    flush = base._L2_FLUSH_BUF

    results, y_first = {}, {}
    for name, kind, P, path in modes:
        L = P // R
        metas = states = phy2log = None
        routing = off
        if kind == "eco":
            metas, phy2log = eco_metas(path, device)
            states = [torch.zeros(P + E, dtype=torch.int16, device=device) for _ in metas]
        elif kind in ("eplb", "ora"):
            routing = np.load(path, mmap_mode="r")
            if kind == "ora":
                phy2log = np.load(args.phy2log.format(P=P))
        ids = (np.arange(rank * L, (rank + 1) * L) % E if phy2log is None
               else np.asarray(phy2log[0][rank * L:(rank + 1) * L]))
        weights = expert_weights(ids, device)
        buf = gc_moe.get_symm_buffer_for_gc_moe(dist.group.WORLD, P, T, TOPK, HIDDEN, INTER, num_shared_experts=NS,
                                                **({"eco": True} if kind == "eco" else {}))
        bm = gc_api.get_block_m_for_gc_moe(R, P, buf.num_max_tokens_per_rank, T, TOPK)
        if kind == "eco" and os.environ.get("GC_MOE_ECO_BLOCK_M"):
            bm = int(os.environ["GC_MOE_ECO_BLOCK_M"])
        sh_sf = shared_input_sf(x_sf, buf.shared_l1_acts_sf, bm) if NS else None
        y = torch.empty((T, HIDDEN), device=device, dtype=torch.bfloat16)
        per_call = []
        for i in call_ids:
            buf.x[:T].copy_(x_fp8)
            buf.x_sf[:T].copy_(x_sf)
            buf.topk_idx[:T].copy_(torch.from_numpy(np.asarray(routing[i, rank]).astype(np.int64)).to(device))
            buf.topk_weights[:T].copy_(topk_w)
            if NS:
                buf.shared_l1_acts_sf.copy_(sh_sf)
            extra = {}
            if kind == "eco":
                layer = i % len(metas)
                extra = dict(eco_meta=metas[layer], eco_state=states[layer], eco_logical_experts=E)

            def call():
                gc_moe.fp8_fp4_gc_moe(y, *weights, buf, recipe=(1, 1, 32), activation="swiglu",
                                      activation_clamp=10.0, fast_math=True, **shared_kw, **extra)

            if args.check and i == call_ids[0]:
                call()
                torch.cuda.synchronize()
                y_first[name] = y.clone()
            timed = (lambda: (flush.zero_(), call())) if args.cold_l2 else call
            m = base._bench_cuda_graph_ms(timed, warmup=args.warmup, iters=args.iters, inner_iters=args.inner_iters,
                                          device=device, barrier_fn=dist.barrier)
            med = m["median_ms"] * 1e3
            if args.cold_l2:
                f = base._bench_cuda_graph_ms(lambda: flush.zero_(), warmup=args.warmup, iters=args.iters,
                                              inner_iters=args.inner_iters, device=device, barrier_fn=dist.barrier)
                med -= f["median_ms"] * 1e3
            per_rank = [None] * R
            dist.all_gather_object(per_rank, med)
            per_call.append(dict(call=i, slowest_rank_median_us=max(per_rank), per_rank_median_us=per_rank))
            if rank == 0:
                print(f"{name:<10} call {i:5d}  slowest {max(per_rank):8.2f} us  fastest {min(per_rank):8.2f}", flush=True)
        s = [c["slowest_rank_median_us"] for c in per_call]
        results[name] = dict(kind=kind, P=P, block_m=bm, mean_us=float(np.mean(s)), median_us=float(np.median(s)),
                             per_call=per_call)
        if rank == 0:
            print(f"== {name} (bm{bm}, shared {NS}): mean over {len(s)} calls {np.mean(s):8.2f} us (median {np.median(s):8.2f})", flush=True)
        buf.destroy()
        del weights
        torch.cuda.empty_cache()
        dist.barrier()

    check = {}
    if args.check and "plain" in [k for _, k, _, _ in modes]:
        ref = y_first[next(n for n, k, _, _ in modes if k == "plain")]
        for name, kind, _, _ in modes:
            if kind in ("ora", "eco"):
                d = torch.tensor([(y_first[name].float() - ref.float()).abs().max().item()], device=device)
                dist.all_reduce(d, op=dist.ReduceOp.MAX)
                check[name] = float(d.item())
        if rank == 0:
            print("check max|diff| vs plain:", check, flush=True)
    if rank == 0 and args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        json.dump(dict(T=T, ranks=R, calls=call_ids, num_shared_experts=NS, shared_weight_gran_k=args.shared_weight_gran_k,
                       block_m_pinned=args.block_m, cold_l2=args.cold_l2, inner_iters=args.inner_iters,
                       iters=args.iters, warmup=args.warmup, check=check,
                       summary={n: dict(mean_us=r["mean_us"], median_us=r["median_us"]) for n, r in results.items()},
                       results=results), open(args.output_json, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

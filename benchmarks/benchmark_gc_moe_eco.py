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
import torch.profiler   # noqa: E402

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


def build_xorder_table(off, R, L, K):
    """per_dst[R][L] int32 heat table for x-order dispatch. off: [calls, R,
    T, topk] global expert ids; global id g = dst*L + local. Greedy
    max-coverage picks K early-set experts per dst, the rest follow by mean
    count (XORDER_DISPATCH.md appendix)."""
    calls = off.shape[0]
    flat = off.reshape(calls, R, -1)
    cnt = np.stack([np.bincount(flat[c].ravel(), minlength=R * L).reshape(R, L)
                    for c in range(calls)]).astype(np.float64)   # [calls, R, L]
    per_dst = []
    for d in range(R):
        X = cnt[:, d, :]
        S, cur = [], np.zeros(calls)
        for _ in range(min(K, L)):
            gains = np.maximum(cur[:, None], X).mean(0) - cur.mean()
            gains[S] = -np.inf      # never re-select: ties at gain 0 would duplicate
            eb = int(np.argmax(gains))
            S.append(eb)
            cur = np.maximum(cur, X[:, eb])
        rest = [i for i in np.argsort(-X.mean(0)) if i not in S]
        per_dst.append(S + rest)
    return np.asarray(per_dst, dtype=np.int32).reshape(-1)      # [R*L] dst-major


def _clock_sampler(local_rank: int, period: float = 1.0):
    """Background SM-clock sampler. NVML only (no CUDA context calls), so it is
    safe while a CUDA graph is being captured. Returns (samples, stop_event),
    samples = [(t_sec, sm_mhz)]."""
    import threading
    import time
    try:
        import pynvml
        pynvml.nvmlInit()
        _h = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

        def _get():
            return (int(pynvml.nvmlDeviceGetClockInfo(_h, pynvml.NVML_CLOCK_SM)),
                    int(pynvml.nvmlDeviceGetClockInfo(_h, pynvml.NVML_CLOCK_MEM)))
    except Exception:
        import ctypes
        _nv = ctypes.CDLL('libnvidia-ml.so.1')
        _nv.nvmlInit_v2()
        _h = ctypes.c_void_p()
        _nv.nvmlDeviceGetHandleByIndex_v2(local_rank, ctypes.byref(_h))
        _c = ctypes.c_uint()

        def _get():
            _nv.nvmlDeviceGetClockInfo(_h, 0, ctypes.byref(_c))
            sm = int(_c.value)
            _nv.nvmlDeviceGetClockInfo(_h, 1, ctypes.byref(_c))
            return (sm, int(_c.value))
    samples = []
    stop = threading.Event()
    t0 = time.time()

    def _loop():
        while not stop.is_set():
            try:
                samples.append((time.time() - t0, _get()))
            except Exception:
                pass
            stop.wait(period)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return samples, stop


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
    ap.add_argument("--trace-calls", type=int, default=0,
                    help="GC_MOE_ECO_TRACE=1 builds: after timing a sampled call, run it eagerly this many times (L2 "
                         "flushed, trace ring zeroed before each) and record every rank's per-call milestones (us since "
                         "its role1 start; median over the runs): barrier, pull_end, l1_first, l1_last, l2_last, gemm_done, "
                         "role2_exit, role3_done. Trace runs are for attribution only, not timing.")
    ap.add_argument("--block-m", type=int, default=0, help="pin block M for every mode (0: gc-moe heuristic)")
    ap.add_argument("--xorder", type=int, default=0, metavar="K",
                    help="x-order dispatch: per-dst heat table, early-set size K (0: off; plain modes only)")
    ap.add_argument("--eager-iters", type=int, default=0,
                    help="time N eager calls instead of CUDA-graph replays (needed for stream "
                         "attributes such as the L2 persistence window) and dump the shared "
                         "block duration")
    ap.add_argument("--se-idle-probe", type=int, default=0,
                    help="replay one captured call N times with 2 s gaps (no allocations in "
                         "between) and record the shared block duration each time")
    ap.add_argument("--se-offset-sweep", type=int, default=0,
                    help="re-allocate the shared weights behind N x 64 MB of padding and re-measure "
                         "the shared block duration per offset (placement probe)")
    ap.add_argument("--output-json", default="")
    ap.add_argument("--dump-y", default="", help="save each mode's first-call y to DIR/<mode>.rank<R>.npy")
    ap.add_argument("--dump-shared-dbg", default="",
                    help="save every rank's per-cluster shared (start, done) array and trace ring to DIR")
    ap.add_argument("--profile-calls", type=int, default=0,
                    help="after timing, replay one captured call this many times under torch.profiler (CUDA) "
                         "on plain modes; exports per-rank chrome traces to --profile-dir and the matching "
                         "trace-ring rows")
    ap.add_argument("--profile-dir", default="")
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _be = os.environ.get("GC_MOE_DIST_BACKEND", "nccl")
    if _be == "nccl":
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        dist.init_process_group(backend=_be)
    rank, R = dist.get_rank(), dist.get_world_size()
    if dist.get_backend() != "nccl":
        # gloo needs CPU tensors: transparently bounce every all_gather through CPU
        _real_all_gather = dist.all_gather

        def _cpu_all_gather(out_list, in_tensor):
            src = in_tensor.detach().cpu()
            outs = [torch.zeros_like(src) for _ in out_list]
            _real_all_gather(outs, src)
            for o, v in zip(out_list, outs):
                o.copy_(v.to(o.device))
        dist.all_gather = _cpu_all_gather
    align_first_launch()
    import gc_moe.apis.gc_moe as gc_api
    if args.block_m:
        gc_api.get_block_m_for_gc_moe = lambda *_a, **_k: args.block_m
    NS = args.num_shared_experts
    # Mimic a larger resident footprint (e.g. R=1 holds all routed experts) so the
    # shared weights and workspace land elsewhere in the heap.
    _fp_gb = float(os.environ.get("GC_MOE_EXTRA_FOOTPRINT_GB", "0"))
    _fp_t = (torch.empty(int(_fp_gb * 2 ** 30), dtype=torch.uint8, device=device)
             if _fp_gb > 0 else None)
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
    _xo_cache = {}

    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    x = torch.randn((T, HIDDEN), device=device, dtype=torch.bfloat16, generator=gen) * 0.5
    x_fp8, x_sf = base.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    topk_w = torch.rand((T, TOPK), device=device, generator=gen).softmax(-1)
    base._flush_l2(device)   # allocate the flush buffer outside any capture
    flush = base._L2_FLUSH_BUF

    results, y_first = {}, {}
    for name, kind, P, path in modes:
        L = P // R
        os.environ['GC_MOE_XORDER'] = '1' if (args.xorder and kind == 'plain') else '0'
        xo_table = None
        if args.xorder and kind == 'plain':
            xo_key = (R, L, args.xorder)
            if xo_key not in _xo_cache:
                _xo_cache[xo_key] = build_xorder_table(off, R, L, args.xorder)
            xo_table = torch.tensor(_xo_cache[xo_key], dtype=torch.int32, device=device)
        elif args.xorder and rank == 0:
            print(f"{name}: --xorder skipped (kind={kind}, plain modes only)", flush=True)
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
        _pad_mb = int(os.environ.get("GC_MOE_ALLOC_PAD_MB", "0"))
        _pad_t = (torch.empty(_pad_mb * 2 ** 20, dtype=torch.uint8, device=device)
                  if _pad_mb else None)     # shifts the symmetric allocation's placement
        buf = gc_moe.get_symm_buffer_for_gc_moe(dist.group.WORLD, P, T, TOPK, HIDDEN, INTER, num_shared_experts=NS,
                                                **({"eco": True} if kind == "eco" or args.trace_calls or args.profile_calls or
                                      args.eager_iters or args.dump_shared_dbg else {}))
        bm = gc_api.get_block_m_for_gc_moe(R, P, buf.num_max_tokens_per_rank, T, TOPK)
        if kind == "eco" and os.environ.get("GC_MOE_ECO_BLOCK_M"):
            bm = int(os.environ["GC_MOE_ECO_BLOCK_M"])
        sh_sf = shared_input_sf(x_sf, buf.shared_l1_acts_sf, bm) if NS else None
        y = torch.empty((T, HIDDEN), device=device, dtype=torch.bfloat16)
    
        def _agather(t):
            # gloo backend needs CPU tensors; nccl needs CUDA
            return t.cpu() if dist.get_backend() != "nccl" else t
        per_call = []
        clk_samples, clk_stop = _clock_sampler(local_rank, 1.0)
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
                                      activation_clamp=10.0, fast_math=True, **shared_kw,
                                      xorder_table=xo_table, **extra)

            if args.check and i == call_ids[0]:
                verify = kind == "eco" and os.environ.get("GC_MOE_ECO_ROUTE_VERIFY") == "1"
                if verify:
                    from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS
                    words = buf.buffer.view(torch.int32)
                    off_p = (buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_route_peer"]) // 4
                    off_d = (buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_route_dbg"]) // 4
                    words[off_d:off_d + T * TOPK].fill_(-1)
                    torch.cuda.synchronize()
                    dist.barrier()
                call()
                torch.cuda.synchronize()
                y_first[name] = y.clone()
                if args.dump_y:
                    os.makedirs(args.dump_y, exist_ok=True)
                    np.save(os.path.join(args.dump_y, f"{name}.rank{rank}.npy"),
                            y.float().cpu().numpy())
                if verify:
                    dist.barrier()
                    peer, pulled = words[off_p:off_p + T * TOPK], words[off_d:off_d + T * TOPK]
                    bad = torch.tensor([int((peer != pulled).sum()), int((pulled < 0).sum())], device=device)
                    dist.all_reduce(_agather(bad))
                    if rank == 0:
                        print(f"{name}: route peer verify: {int(bad[0])} mismatches, {int(bad[1])} unpulled "
                              f"of {T * TOPK * R} routes", flush=True)
            timed = (lambda: (flush.zero_(), call())) if args.cold_l2 else call
            eager_shared = None
            if args.eager_iters:
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                ev = []
                from gc_moe.layout.gc_moe import SIGNAL_OFFSETS as _SOE, SHARED_DBG_OFF as _SDOE, \
                    SHARED_DBG_WORDS as _SDWE
                _off = (_SOE["expert_recv_count_sum"] + _SDOE) // 4
                _shared_calls = []   # per timed call: (median, max_span, end)
                for _ in range(args.eager_iters):
                    s_ev, e_ev = torch.cuda.Event(True), torch.cuda.Event(True)
                    s_ev.record()
                    call()
                    e_ev.record()
                    torch.cuda.synchronize()
                    ev.append(s_ev.elapsed_time(e_ev) * 1e3)
                    # read THIS call's per-cluster stamps (role1 clears them
                    # at the next call's start), so the shared metrics average
                    # over the timed calls instead of sampling the last one.
                    _v = buf.buffer.view(torch.int32)[_off:_off + _SDWE] \
                        .view(torch.int64).view(68, 2).to(torch.float64).cpu().numpy() / 1000.0
                    _st = np.where(_v[:, 0] < 0, -1.0 - _v[:, 0], np.nan)
                    _dn = np.where(_v[:, 1] > 0, _v[:, 1], np.nan)
                    _shared_calls.append((float(np.nanmedian(_dn - _st)),
                                          float(np.nanmax(_dn - _st)),
                                          float(np.nanmax(_dn))))
                # cross-call aggregation: median / max / min of each metric
                _sa = np.array(_shared_calls) if _shared_calls else np.zeros((1, 3))
                def _agg3(col):
                    return (float(np.median(_sa[:, col])), float(np.max(_sa[:, col])),
                            float(np.min(_sa[:, col])))
                eager_shared = _agg3(0)[0]        # median of per-call medians
                # gating metrics: the shared phase gates downstream on its LAST
                # cluster, so also report the slowest cluster's span and the
                # absolute end time (ns since this rank's role1 start).
                eager_shared_max = _agg3(1)[0]     # median of per-call max_span
                eager_shared_end = _agg3(2)[0]     # median of per-call end (gate)
                _end_worst = _agg3(2)[1]           # worst-call gate
                _end_best = _agg3(2)[2]            # best-call gate
                if args.dump_shared_dbg:
                    # per-cluster (start, done) in us since this rank's role1 start,
                    # plus the per-call trace ring (same time base)
                    os.makedirs(args.dump_shared_dbg, exist_ok=True)
                    np.save(os.path.join(args.dump_shared_dbg,
                                         f"shared_dbg.rank{rank}.npy"), _v)
                    from gc_moe.layout.gc_moe import SHARED_DBG2_OFF as _SD2O, \
                        SHARED_DBG2_WORDS as _SD2W
                    _o2 = (_SOE["expert_send_count"] + _SD2O) // 4
                    _v2 = buf.buffer.view(torch.int32)[_o2:_o2 + _SD2W] \
                        .view(torch.int64).view(68, 4).to(torch.float64).cpu().numpy() / 1000.0
                    np.save(os.path.join(args.dump_shared_dbg,
                                         f"shared_dbg2.rank{rank}.npy"), _v2)
                    try:
                        from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS as _ESO, \
                            ECO_TRACE_RING_FIELDS as _RF
                        _roff = (buf.layout.eco_base + _ESO["eco_trace_ring"]) // 4
                        _ring = buf.buffer.view(torch.int32)[_roff:_roff + 64 * _RF].view(64, _RF)
                        np.save(os.path.join(args.dump_shared_dbg,
                                             f"trace_ring.rank{rank}.npy"), _ring.cpu().numpy())
                    except Exception as _exc:                       # noqa: BLE001
                        print(f"trace ring dump skipped: {_exc}", flush=True)
                _gdev = device if dist.get_backend() == "nccl" else torch.device("cpu")
                _es = torch.tensor([eager_shared], device=_gdev, dtype=torch.float64)
                _eg = [torch.zeros(1, dtype=torch.float64, device=_gdev) for _ in range(R)]
                dist.all_gather(_eg, _es)
                eager_shared_ranks = [float(t[0]) for t in _eg]
                _extra_ranks = {}
                for _nm, _val in (("eager_shared_max_ranks", eager_shared_max),
                                  ("eager_shared_end_ranks", eager_shared_end),
                                  ("eager_shared_end_worst_ranks", _end_worst),
                                  ("eager_shared_end_best_ranks", _end_best)):
                    _t = torch.tensor([_val], device=_gdev, dtype=torch.float64)
                    _o = [torch.zeros(1, dtype=torch.float64, device=_gdev) for _ in range(R)]
                    dist.all_gather(_o, _t)
                    _extra_ranks[_nm] = [float(x[0]) for x in _o]
                m = {"median_ms": float(np.median(ev)) / 1e3, "eager_us": ev,
                     "shared_calls": _shared_calls}
            else:
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
            if args.trace_calls:
                from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS, ECO_TRACE_RING_FIELDS as RF
                ring = buf.buffer.view(torch.int32)[(buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_trace_ring"]) // 4:][
                    :64 * RF].view(64, RF)
                # one graph = L2 flush + ring clear + the call; 3 back-to-back replays per run reach the replay steady
                # state (no host launch skew between ranks), the last replay's (only) ring entry is read
                cs = torch.cuda.Stream(device=device)
                cs.wait_stream(torch.cuda.current_stream(device))
                tg = torch.cuda.CUDAGraph()
                with torch.cuda.stream(cs):
                    with torch.cuda.graph(tg, stream=cs):
                        flush.zero_()
                        ring.zero_()
                        call()
                torch.cuda.current_stream(device).wait_stream(cs)
                torch.cuda.synchronize()
                slots = buf.buffer.view(torch.int64)[(buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_trace"]) // 8:][:64]
                slots[44:62].zero_()
                torch.cuda.synchronize()
                runs = []
                for _ in range(args.trace_calls):
                    dist.barrier()
                    for _ in range(3):
                        tg.replay()
                    torch.cuda.synchronize()
                    e = ring.to(torch.int64).bitwise_and(0xFFFFFFFF).to(torch.float64) / 1000.0
                    row = e[int(torch.argmax(e[:, 1]))]
                    # fields: 0 barrier, 6 pull end, 13 first L1 start (inverted), 3 last L1, 7 last L2, 1 GEMM done
                    # (role3 start), 8 last role2 CTA exit, 9 last role3 warp (ECO reuses some fields: read 0/1/9)
                    raw = [float(v) for v in row]
                    for k in (13, 14, 16, 20, 21, 22, 23, 24, 25):   # inverted max = earliest
                        raw[k] = 4294967.295 - raw[k] if raw[k] > 0 else float("nan")
                    runs.append([row[0], row[6], 4294967.295 - row[13] if row[13] > 0 else float("nan"),
                                 row[3], row[7], row[1], row[8], row[9]] + raw)
                del tg
                loc = torch.tensor(np.nanmedian(np.array([[float(v) for v in r] for r in runs]), 0), device=device)
                alls = [torch.zeros_like(loc) for _ in range(R)]
                dist.all_gather(alls, _agather(loc))
                g_ = torch.stack(alls).cpu().numpy().round(2)
                per_call[-1]["trace_us"] = dict(zip(("barrier", "pull_end", "l1_first", "l1_last", "l2_last",
                                                     "gemm_done", "role2_exit", "role3_done"), g_[:, :8].T.tolist()))
                per_call[-1]["trace_raw_us"] = g_[:, 8:].tolist()   # [rank][ring field 0..15], 13/14 de-inverted
                # slots 3/4 (plain pull, scheduler instrumentation): per-expert count-ready
                # min/max, ns since this call's role1 start (u64 stores, last replay wins)
                cr = [torch.zeros(2, dtype=torch.float64, device=device) for _ in range(R)]
                dist.all_gather(cr, _agather(torch.stack([slots[3].to(torch.float64), slots[4].to(torch.float64)])))
                cr_ = torch.stack(cr).cpu().numpy() / 1000.0
                per_call[-1]["count_ready_us"] = np.stack(
                    [cr_[:, 0], cr_[:, 1], cr_[:, 1] - cr_[:, 0]], 1).round(3).tolist()
                ab = slots[[24, 59, 60, 61]].clone()   # abs ns: role1 start, plan kernel start, role3 end, prev role3 end
                # role1 phase split (plain pull): slot30=r1_counted(max), slot31=r1_srcwrote(max) ns since role1 start
                r1ph = [torch.zeros(2, dtype=torch.float64, device=device) for _ in range(R)]
                dist.all_gather(r1ph, torch.stack([slots[30].to(torch.float64), slots[31].to(torch.float64)]))
                r1ph_ = torch.stack(r1ph).cpu().numpy() / 1000.0
                per_call[-1]["r1_phase_us"] = np.stack(
                    [r1ph_[:, 0], r1ph_[:, 1], r1ph_[:, 1] - r1ph_[:, 0]], 1).round(3).tolist()
                # dispatch sub-phase probes (ns since role1 start): 37=route-loop start,
                # 38=route pass end, 39=post grid_sync, 40=post publish+sync
                pb = slots[37:41].clone()
                pbs = [torch.zeros_like(pb) for _ in range(R)]
                dist.all_gather(pbs, _agather(pb))
                per_call[-1]["probe_ns"] = torch.stack(pbs).cpu().numpy().tolist()
                abs_ = [torch.zeros_like(ab) for _ in range(R)]
                dist.all_gather(abs_, _agather(ab))
                # per rank (us, relative to the previous call's last role3 warp): role1 start, plan kernel start
                # (0 = none), this call's role3 end
                A_ = torch.stack(abs_).cpu().numpy().astype(np.float64)
                per_call[-1]["abs_us"] = np.round(np.stack([(A_[:, 0] - A_[:, 3]) / 1e3,
                                                            np.where(A_[:, 1] > 0, (A_[:, 1] - A_[:, 3]) / 1e3, 0.0),
                                                            (A_[:, 2] - A_[:, 3]) / 1e3], 1), 2).tolist()
                ph = slots[44:60].clone()
                phs = [torch.zeros_like(ph) for _ in range(R)]
                dist.all_gather(phs, _agather(ph))
                # serial-pull per-token phase sums (ns): [rank][replica: n, locate, index, ring wait, issue, sf/weight,
                # publish | single-copy: same 7] (GC_MOE_ECO_TRACE=1 ECO builds)
                per_call[-1]["pull_phase_ns"] = torch.stack(phs).cpu().numpy()[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]].tolist()
            if rank == 0:
                print(f"{name:<10} call {i:5d}  slowest {max(per_rank):8.2f} us  fastest {min(per_rank):8.2f}", flush=True)
        profile_ring = None
        if args.profile_calls and kind in ("plain", "eplb", "ora") and args.profile_dir:
            from gc_moe.layout.gc_moe import ECO_SIGNAL_OFFSETS, ECO_TRACE_RING_FIELDS as _RF
            _ring = buf.buffer.view(torch.int32)[(buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_trace_ring"]) // 4:][
                :64 * _RF].view(64, _RF)
            _slots = buf.buffer.view(torch.int64)[(buf.layout.eco_base + ECO_SIGNAL_OFFSETS["eco_trace"]) // 8:][:64]
            torch.cuda.synchronize()
            call_idx0 = int(_slots[33].item())
            os.makedirs(args.profile_dir, exist_ok=True)
            # profile N replays of one captured call: ranks stay launch-synchronized
            # (eager calls drift ~ms apart and inflate every cross-rank count wait)
            cs = torch.cuda.Stream(device=device)
            cs.wait_stream(torch.cuda.current_stream(device))
            tg = torch.cuda.CUDAGraph()
            with torch.cuda.stream(cs):
                with torch.cuda.graph(tg, stream=cs):
                    call()
            torch.cuda.current_stream(device).wait_stream(cs)
            torch.cuda.synchronize()
            from gc_moe.layout.gc_moe import SIGNAL_OFFSETS as _SO0, SHARED_DBG_OFF as _SDO0, \
                SHARED_DBG_WORDS as _SDW0
            _dbg_off0 = (_SO0["expert_recv_count_sum"] + _SDO0) // 4
            _dbg_view = buf.buffer.view(torch.int32)[_dbg_off0:_dbg_off0 + _SDW0] \
                .view(torch.int64).view(68, 2).to(torch.float64)
            _dbg_seq = []
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                dist.barrier()
                for _ in range(args.profile_calls):
                    tg.replay()
                    torch.cuda.synchronize()
                    _dbg_seq.append(_dbg_view.clone())   # per-replay snapshot
                torch.cuda.synchronize()
            prof.export_chrome_trace(os.path.join(args.profile_dir, f"rank{rank}.json"))
            del tg
            entries = torch.tensor([(call_idx0 + j) % 64 for j in range(1, args.profile_calls + 1)],
                                   device=device)
            pring = _ring.index_select(0, entries).to(torch.int64).bitwise_and(0xFFFFFFFF)
            pr = [torch.zeros((args.profile_calls, _RF), dtype=torch.float64, device=device) for _ in range(R)]
            dist.all_gather(pr, _agather(pring.to(torch.float64)))
            profile_ring = torch.stack(pr).cpu().numpy().tolist()   # [rank][call][ring field] u32 ns
            # debug: dump every ring entry + the call counter so entry indices can be verified offline
            allring = _ring.to(torch.int64).bitwise_and(0xFFFFFFFF).to(torch.float64)
            pr2 = [torch.zeros((64, _RF), dtype=torch.float64, device=device) for _ in range(R)]
            dist.all_gather(pr2, _agather(allring))
            profile_ring_all = torch.stack(pr2).cpu().numpy().tolist()
            # per-cluster shared-expert completion times (kernel debug array)
            from gc_moe.layout.gc_moe import SIGNAL_OFFSETS as _SO
            from gc_moe.layout.gc_moe import SHARED_DBG_OFF as _SDO, SHARED_DBG_WORDS as _SDW
            _dbg_off = (_SO["expert_recv_count_sum"] + _SDO) // 4
            _dbg = buf.buffer.view(torch.int32)[_dbg_off:_dbg_off + _SDW].view(torch.int64).view(68, 2).to(torch.float64)
            pr4 = [torch.zeros((68, 2), dtype=torch.float64, device=device) for _ in range(R)]
            dist.all_gather(pr4, _agather(_dbg))
            profile_shared_dbg = torch.stack(pr4).cpu().numpy().tolist()   # [rank][cluster][2] u64 ns
            _seq = torch.stack(_dbg_seq).reshape(-1)                       # [K*136]
            _sq = [torch.zeros_like(_seq) for _ in range(R)]
            dist.all_gather(_sq, _agather(_seq))
            profile_shared_dbg_seq = [t.view(args.profile_calls, 68, 2).cpu().numpy().tolist()
                                      for t in _sq]                        # [rank][replay][cluster][2]
            ctr = torch.tensor([float(call_idx0)], device=device, dtype=torch.float64)
            pr3 = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(R)]
            dist.all_gather(pr3, _agather(ctr))
            profile_call_idx0 = [int(v[0]) for v in pr3]
        clk_stop.set()
        _sm = np.array([c[0] for _, c in clk_samples], dtype=np.float64) if clk_samples else np.zeros(1)
        _mem = np.array([c[1] for _, c in clk_samples], dtype=np.float64) if clk_samples else np.zeros(1)
        _tail = np.array([c[1] for t, c in clk_samples if t >= clk_samples[-1][0] - 60], dtype=np.float64) \
            if clk_samples else _mem
        _stat = torch.tensor([float(np.median(_sm)), float(_mem.min()),
                              float(np.median(_tail)), float(len(_sm))],
                             device=device, dtype=torch.float64)
        _gath = [torch.zeros(4, dtype=torch.float64, device=device) for _ in range(R)]
        dist.all_gather(_gath, _agather(_stat))
        clk_stats = torch.stack(_gath).cpu().numpy().tolist()   # [rank] med, min, tail60_med, n
        # device-address placement of the buffers the shared expert streams
        _ptrs = [buf.buffer.data_ptr(), buf.x.data_ptr()]
        if NS:
            _ptrs += [shared[0][0].data_ptr(), shared[0][1].data_ptr(),
                      shared[1][0].data_ptr(), shared[1][1].data_ptr()]
        else:
            _ptrs += [0, 0, 0, 0]
        _pt = torch.tensor([float(v) for v in _ptrs], device=device, dtype=torch.float64)
        _pg = [torch.zeros(6, dtype=torch.float64, device=device) for _ in range(R)]
        dist.all_gather(_pg, _agather(_pt))
        ptr_stats = torch.stack(_pg).cpu().numpy().tolist()   # [rank][buffer, x, w1, w1sf, w2, w2sf]
        se_idle = None
        if args.se_idle_probe and NS:
            import time as _t
            _fn = (lambda: gc_moe.fp8_fp4_gc_moe(y, *weights, buf, recipe=(1, 1, 32),
                                                 activation="swiglu", activation_clamp=10.0,
                                                 fast_math=True, **shared_kw, xorder_table=xo_table))
            for _ in range(3):
                _fn()
            torch.cuda.synchronize()
            _cs = torch.cuda.Stream(device=device)
            _cs.wait_stream(torch.cuda.current_stream(device))
            _tg = torch.cuda.CUDAGraph()
            with torch.cuda.stream(_cs):
                with torch.cuda.graph(_tg, stream=_cs):
                    _fn()
            torch.cuda.current_stream(device).wait_stream(_cs)
            torch.cuda.synchronize()
            _seq = []
            for _ in range(args.se_idle_probe):
                _tg.replay()
                torch.cuda.synchronize()
                _v = buf.buffer.view(torch.int32)[_dbg_off0:_dbg_off0 + _SDW0] \
                    .view(torch.int64).view(68, 2).to(torch.float64).cpu().numpy()
                _st = np.where(_v[:, 0] < 0, -1.0 - _v[:, 0], np.nan) / 1000.0
                _dn = np.where(_v[:, 1] > 0, _v[:, 1], np.nan) / 1000.0
                _seq.append(float(np.nanmedian(_dn - _st)))
                _t.sleep(2.0)
            del _tg
            _sq2 = torch.tensor(_seq, device=device, dtype=torch.float64)
            _sg2 = [torch.zeros_like(_sq2) for _ in range(R)]
            dist.all_gather(_sg2, _agather(_sq2))
            se_idle = [t.cpu().numpy().tolist() for t in _sg2]
        se_sweep = None
        if args.se_offset_sweep and NS:
            from gc_moe.layout.gc_moe import SIGNAL_OFFSETS as _SO0, SHARED_DBG_OFF as _SDO0, \
                SHARED_DBG_WORDS as _SDW0
            _dbg_off0 = (_SO0["expert_recv_count_sum"] + _SDO0) // 4
            _pads = [0, 64, 128, 192, 256, 384, 512, 768]

            def _mk_call(_shk):
                def _f():
                    gc_moe.fp8_fp4_gc_moe(y, *weights, buf, recipe=(1, 1, 32), activation="swiglu",
                                          activation_clamp=10.0, fast_math=True, **_shk,
                                          xorder_table=xo_table)
                return _f

            def _capture(_fn):
                for _ in range(3):
                    _fn()
                torch.cuda.synchronize()
                _cs = torch.cuda.Stream(device=device)
                _cs.wait_stream(torch.cuda.current_stream(device))
                _tg = torch.cuda.CUDAGraph()
                with torch.cuda.stream(_cs):
                    with torch.cuda.graph(_tg, stream=_cs):
                        _fn()
                torch.cuda.current_stream(device).wait_stream(_cs)
                torch.cuda.synchronize()
                return _tg

            def _measure(_tg):
                for _ in range(2):
                    _tg.replay()
                    torch.cuda.synchronize()
                _tg.replay()
                torch.cuda.synchronize()
                _v = buf.buffer.view(torch.int32)[_dbg_off0:_dbg_off0 + _SDW0] \
                    .view(torch.int64).view(68, 2).to(torch.float64).cpu().numpy()
                _st = np.where(_v[:, 0] < 0, -1.0 - _v[:, 0], np.nan) / 1000.0
                _dn = np.where(_v[:, 1] > 0, _v[:, 1], np.nan) / 1000.0
                return float(np.nanmedian(_dn - _st))

            # reference allocation: never touched by the sweep, re-measured after
            # every pad step -> its drift is the concurrent-interference control
            _shref = shared_weights(NS, args.shared_weight_gran_k, device)
            _ref_tg = _capture(_mk_call(dict(
                shared_l1_weights=_shref[0], shared_l2_weights=_shref[1],
                shared_recipe=((1, 128, 128) if args.shared_weight_gran_k == 128 else (1, 1, 32)))))
            _ref_durs = [_measure(_ref_tg)]
            _durs, _ptrs = [], []
            for _pad_mb in _pads:
                _pad = (torch.empty(int(_pad_mb) * 2 ** 20, dtype=torch.uint8, device=device)
                        if _pad_mb else None)
                _shw = shared_weights(NS, args.shared_weight_gran_k, device)
                _tg = _capture(_mk_call(dict(
                    shared_l1_weights=_shw[0], shared_l2_weights=_shw[1],
                    shared_recipe=((1, 128, 128) if args.shared_weight_gran_k == 128
                                   else (1, 1, 32)))))
                _durs.append(_measure(_tg))
                _ptrs.append(float(_shw[0][0].data_ptr()))
                del _tg, _shw, _pad
                torch.cuda.empty_cache()
                _ref_durs.append(_measure(_ref_tg))   # interference control
            _sw = torch.tensor(_durs + _ptrs, device=device, dtype=torch.float64)
            _sg = [torch.zeros(2 * len(_pads), dtype=torch.float64, device=device) for _ in range(R)]
            dist.all_gather(_sg, _agather(_sw))
            _rf = torch.tensor(_ref_durs, device=device, dtype=torch.float64)
            _rg = [torch.zeros_like(_rf) for _ in range(R)]
            dist.all_gather(_rg, _agather(_rf))
            se_sweep = dict(pads=_pads,
                            dur=[t[:len(_pads)].cpu().numpy().tolist() for t in _sg],
                            ptr=[t[len(_pads):].cpu().numpy().tolist() for t in _sg],
                            ref=[t.cpu().numpy().tolist() for t in _rg])
            if rank == 0:
                print("se offset sweep (rank0):", [round(v, 1) for v in se_sweep["dur"][0]], flush=True)
        s = [c["slowest_rank_median_us"] for c in per_call]
        results[name] = dict(kind=kind, P=P, block_m=bm, mean_us=float(np.mean(s)), median_us=float(np.median(s)),
                             per_call=per_call, clk_stats=clk_stats, ptr_stats=ptr_stats,
                             **({"eager_shared_us": eager_shared,
                                 "eager_shared_ranks": eager_shared_ranks,
                                 **_extra_ranks}
                                if eager_shared is not None else {}),
                             **({"se_sweep": se_sweep} if se_sweep is not None else {}),
                             **({"se_idle": se_idle} if se_idle is not None else {}),
                             **({"profile_ring_ns": profile_ring,
                                 "profile_ring_all": profile_ring_all,
                                 "profile_call_idx0": profile_call_idx0,
                                 "profile_shared_dbg": profile_shared_dbg,
                                 "profile_shared_dbg_seq": profile_shared_dbg_seq} if profile_ring is not None else {}))
        if rank == 0:
            print(f"== {name} (bm{bm}, shared {NS}): mean over {len(s)} calls {np.mean(s):8.2f} us (median {np.median(s):8.2f})", flush=True)
            if eager_shared is not None:
                _g = _extra_ranks["eager_shared_end_ranks"]
                _gw = _extra_ranks["eager_shared_end_worst_ranks"]
                _gb = _extra_ranks["eager_shared_end_best_ranks"]
                print(f"   shared gate (us, {args.eager_iters}-call median): "
                      f"min {min(_g):5.1f} max {max(_g):5.1f} ratio {max(_g)/max(min(_g),1e-9):.2f}x | "
                      f"worst-call {min(_gw):5.1f}-{max(_gw):5.1f} ({max(_gw)/max(min(_gw),1e-9):.2f}x) | "
                      f"best-call {min(_gb):5.1f}-{max(_gb):5.1f} ({max(_gb)/max(min(_gb),1e-9):.2f}x) | "
                      f"per-rank [{' '.join(f'{v:.1f}' for v in _g)}]", flush=True)
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

"""Native-methodology MoE timing with fused shared experts, gc-moe only.

Same CUDA-graph timing as benchmark_gc_moe_se.py (back-to-back replays,
inner_iters amortization, no host sync between calls), but:
- no deep_gemm baseline (gc_moe candidate only)
- identity routing (all tokens route to local experts)
- optional torch.profiler chrome trace export per rank
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'tests'))
os.environ['DG_PRINT_CONFIGS'] = '0'
import numpy as np
import torch
import torch.distributed as dist
import gc_moe
assert Path(gc_moe.__file__).resolve().is_relative_to(_ROOT), gc_moe.__file__
from test_gc_moe_se import (
    ROUTED_RECIPE, allocate_buffer, build_weights,
    per_token_cast_to_fp8, prepare_shared_sf, stage_inputs,
)
from benchmark_gc_moe import (
    _bench_cuda_graph_ms, _collect_throughput_workload,
    _collect_weight_storage, _common_weight_storage,
    _print_throughput_metrics, _throughput_definitions,
    _throughput_rates, _weight_storage,
)

_L2_FLUSH_BUF: Optional[torch.Tensor] = None


def _flush_l2(device: torch.device) -> None:
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None or _L2_FLUSH_BUF.device != device:
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        _L2_FLUSH_BUF = torch.empty(max(l2_bytes * 2, 256 * 1024 * 1024),
                                    dtype=torch.int8, device=device)


def build_identity_tokens(T, hidden, num_experts, num_topk, rank, world, device):
    """All tokens route to this rank's local experts (identity routing)."""
    g_x = torch.Generator(device=device).manual_seed(42 + rank * 7919)
    x = torch.randn(T, hidden, dtype=torch.bfloat16, device=device, generator=g_x) * 0.5
    local_experts = num_experts // world
    first = rank * local_experts
    # Each token picks num_topk experts from its rank's local range
    g_r = torch.Generator(device=device).manual_seed(1000 + rank * 31)
    scores = torch.randn(T, local_experts, dtype=torch.float32, device=device, generator=g_r)
    topk_w, topk_local = torch.topk(scores, min(num_topk, local_experts), dim=-1,
                                    largest=True, sorted=False)
    topk_w = torch.softmax(topk_w, dim=-1)
    topk_idx = (topk_local + first).to(torch.int64)
    # Pad if num_topk > local_experts (repeat experts)
    if topk_idx.shape[1] < num_topk:
        pad = topk_idx[:, :1].expand(-1, num_topk - topk_idx.shape[1])
        topk_idx = torch.cat([topk_idx, pad], dim=1)
        pad_w = torch.zeros(T, num_topk - topk_w.shape[1], device=device)
        topk_w = torch.cat([topk_w, pad_w], dim=1)
    return x, topk_idx, topk_w


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--intermediate-hidden', type=int, default=2048)
    p.add_argument('--num-experts', type=int, default=256)
    p.add_argument('--num-topk', type=int, default=8)
    p.add_argument('--num-shared-experts', type=int, default=1)
    p.add_argument('--shared-weight-gran-k', type=int, choices=[32, 128], default=128)
    p.add_argument('--num-tokens-per-rank-list', default='128')
    p.add_argument('--num-max-tokens-per-rank', type=int, default=0)
    p.add_argument('--activation', choices=['swiglu'], default='swiglu')
    p.add_argument('--activation-alpha', type=float, default=1.0)
    p.add_argument('--activation-clamp', type=float, default=10.0)
    p.add_argument('--fast-math', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--inner-iters', type=int, default=100)
    p.add_argument('--cuda-graph-max-tokens-per-rank', type=int, default=256)
    p.add_argument('--output-json', default='')
    p.add_argument('--profile', action='store_true',
                   help='export per-rank chrome trace (torch.profiler, CUDA)')
    p.add_argument('--profile-dir', default='prof_se')
    args = p.parse_args()

    sweep = [int(t) for t in args.num_tokens_per_rank_list.split(',') if t.strip()]
    capacity = args.num_max_tokens_per_rank or max(sweep)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()

    # gc_moe weights only (no deep_gemm baseline)
    _, gr, _, gs, _, recipe, _ = build_weights(args, rank, world, device, reference=False)
    local_experts = args.num_experts // world
    shared_storage = _common_weight_storage(_weight_storage(gs))
    routed_storage = _common_weight_storage(_weight_storage(gr, local_experts))
    del _
    weight_storage_by_rank = _collect_weight_storage(routed_storage, shared_storage)

    gb = allocate_buffer(args, capacity, 'gc')
    torch.cuda.empty_cache()
    results = []

    for tokens in sweep:
        x, indices, weights = build_identity_tokens(
            tokens, args.hidden, args.num_experts, args.num_topk, rank, world, device)
        x8, xs = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        gsf, gbm = prepare_shared_sf(args, xs, gb, tokens, world, 'gc')
        stage_inputs(gb, x8, xs, indices, weights, gsf)
        gc_y = torch.empty((tokens, args.hidden), dtype=torch.bfloat16, device=device)

        def candidate_fn():
            gc_moe.fp8_fp4_gc_moe(gc_y, *gr, gb,
                shared_l1_weights=gs[0], shared_l2_weights=gs[1],
                recipe=ROUTED_RECIPE, shared_recipe=recipe,
                activation=args.activation,
                activation_clamp=args.activation_clamp,
                activation_alpha=args.activation_alpha,
                fast_math=bool(args.fast_math))

        graph_mode = args.cuda_graph_max_tokens_per_rank > 0 and tokens <= args.cuda_graph_max_tokens_per_rank

        if args.profile:
            # Profiled pass: wrap the timing in torch.profiler
            os.makedirs(args.profile_dir, exist_ok=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                measured = _bench_cuda_graph_ms(
                    candidate_fn, warmup=args.warmup, iters=max(args.iters, 5),
                    inner_iters=max(args.inner_iters // 10, 10),
                    device=device, barrier_fn=dist.barrier)
            prof.export_chrome_trace(os.path.join(args.profile_dir, f"rank{rank}.json"))
        else:
            # Timing pass: native methodology (back-to-back graph replays)
            measured = _bench_cuda_graph_ms(
                candidate_fn, warmup=args.warmup, iters=args.iters,
                inner_iters=args.inner_iters,
                device=device, barrier_fn=dist.barrier)

        measured.update(rank=rank, mode='cuda_graph' if graph_mode else 'eager')
        all_ranks = [None] * world
        dist.all_gather_object(all_ranks, measured)
        medians = [row['median_ms'] * 1000 for row in all_ranks]
        operator_stats = dict(operator='gc_moe.fp8_fp4_gc_moe', ranks=all_ranks,
                              per_rank_median_us=medians, slowest_rank_median_us=max(medians))
        workload = _collect_throughput_workload(
            indices, args.num_experts, args.hidden, args.intermediate_hidden,
            weight_storage_by_rank, args.intermediate_hidden * args.num_shared_experts)
        throughput = {'gc': _throughput_rates(workload, measured['slowest_rank_median_us'])}
        base = max(medians)  # slowest rank
        results.append(dict(num_tokens_per_rank=tokens, num_tokens_global=tokens * world,
                            mode='cuda_graph' if graph_mode else 'eager',
                            gc_us=base, gc_block_m=gbm,
                            operator_stats=operator_stats, workload=workload,
                            throughput=throughput))
        if rank == 0:
            print(f'TOKENS tokens_per_rank={tokens} total={tokens*world} '
                  f'GC_SE={base:.2f}us mode={results[-1]["mode"]}', flush=True)
            _print_throughput_metrics(tokens, throughput)
        del gc_y, x, x8, xs, indices, weights, gsf
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0 and args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(world_size=world, config=vars(args),
                                       results=results), indent=2) + '\n')
        print('Wrote', str(out), flush=True)

    torch.cuda.synchronize(device)
    gb.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

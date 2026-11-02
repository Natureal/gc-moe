"""ECO plan role (role 4): nvcc-compiled CUDA C++ cubin, cached by configuration and source hash."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache

from ..layout.gc_moe import ECO_SIGNAL_OFFSETS, SIGNAL_OFFSETS

ECO_PLAN_ROLE = 4
# Plan role build (`eco_unit_kernel.cu`): 16 plan warps + 2 index warps
ECO_UNIT_THREADS = 576


# Rank-time weights (block wb, active slot wa; a token weighs 1) per gc-moe block M, fitted on gc-moe kernel times
# (`gcbench/fit_cost_bm.py`); other block sizes take the nearest entry.
ECO_TIME_WEIGHTS = {16: (249, 140), 32: (272, 117), 64: (260, 111), 96: (335, 38), 128: (293, 52)}


def eco_unit_config(num_ranks: int, block_m: int) -> tuple:
    """Plan role build for an EP size and the call's block M (gc-moe's choice; the plan unit): (block M, block weight
    wb, active-slot weight wa, warm auction rounds, cold auction rounds, two-hop rounds). The planner minimizes the
    slowest rank's time wb * blocks + wa * active slots + tokens; GC_MOE_ECO_OBJECTIVE=token balances the tokens alone
    (wb = wa = 0, two rounds up to EP16). Dense rank graphs (<= 32 ranks) need one-hop rounds only; EP64 starts with
    two two-hop rounds. GC_MOE_ECO_UNIT_CFG overrides (wb, wa, warm rounds, cold rounds, two-hop rounds)."""
    if os.environ.get('GC_MOE_ECO_UNIT_CFG'):
        return (block_m,) + tuple(int(x) for x in os.environ['GC_MOE_ECO_UNIT_CFG'].split(','))
    token = os.environ.get('GC_MOE_ECO_OBJECTIVE', 'time') == 'token'
    wb, wa = (0, 0) if token else ECO_TIME_WEIGHTS[min(ECO_TIME_WEIGHTS, key=lambda b: (abs(b - block_m), b))]
    nr, hop2 = (1 + token, 0) if num_ranks <= 16 else (2, 0) if num_ranks <= 32 else (3, 2)
    return (block_m, wb, wa, nr, 6, hop2)
# Timeline trace slots (globaltimer ns, GC_MOE_ECO_TRACE=1): mode store / min / max
ECO_TRACE_SLOTS = {
    0: ('role1_start', 'min'), 1: ('role1_barrier_done', 'store'), 2: ('role1_counts_ready', 'store'),
    3: ('plan_start', 'store'), 4: ('plan_counts_seen', 'store'), 5: ('plan_published', 'store'),
    6: ('role1_plan_wait_first', 'min'), 7: ('role1_plan_seen_last', 'max'), 8: ('role1_end', 'max'),
    9: ('role2_sched_start', 'min'), 10: ('role2_counts_seen', 'min'), 11: ('role2_plan_wait_first', 'min'),
    12: ('role2_plan_seen_last', 'max'), 13: ('role2_sched_end', 'max'), 14: ('role3_start', 'min'),
    15: ('role3_end', 'max'), 16: ('role2_end', 'max'),
    # plan role: start done, auction rounds done, replicated indices written
    17: ('plan_start_done', 'store'), 18: ('plan_rounds_done', 'store'), 19: ('plan_index_done', 'store'),
    # Graph-replay accumulators (sums over calls of `t - role1 start` of the call; 21: call count, 24: start)
    20: ('acc_barrier_done', 'sum'), 21: ('acc_calls', 'sum'), 22: ('acc_role3_start', 'sum'),
    23: ('acc_role3_end_b0', 'sum'), 24: ('role1_start_b0', 'store'), 25: ('acc_plan_published', 'sum'),
    26: ('acc_role2_plan_stall', 'sum'), 27: ('acc_role1_plan_stall', 'sum'), 28: ('acc_role2_stalls', 'sum'),
    29: ('acc_role1_stalls', 'sum'),
}
_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCES = ('eco_unit_kernel.cu', 'eco_unit_auction.cuh', 'include/eco_meta.cuh', 'include/eco_compat.cuh')


def _nvcc() -> str:
    return os.environ.get('GC_MOE_NVCC', '/usr/local/cuda/bin/nvcc')


def _cache_dir() -> str:
    path = os.environ.get('GC_MOE_ECO_CACHE_DIR',
                          os.path.join(os.path.expanduser('~'), '.cache', 'gc_moe', 'eco'))
    os.makedirs(path, exist_ok=True)
    return path


def _source_digest() -> str:
    digest = hashlib.sha256()
    for name in _SOURCES:
        with open(os.path.join(_SOURCE_DIR, name), 'rb') as f:
            digest.update(f.read())
    return digest.hexdigest()


@lru_cache(maxsize=64)
def compile_eco_plan(num_physical: int, num_ranks: int, num_logical: int, eco_base: int,
                     topk_offset: int, topk: int, capacity: int, unit: tuple,
                     arch: str = 'sm_103a', trace: bool = False) -> bytes:
    """Cubin of the plan role (`eco_unit_kernel.cu`, ECO_UNIT_THREADS threads) for one (P, R, E) configuration,
    buffer layout (`eco_base`: the buffer's ECO region, `GCBufferLayout.eco_base`) and `unit` build (see
    `eco_unit_config`)."""
    if num_logical % 2 or num_physical % num_ranks or num_ranks > 64 or num_physical > 2048:
        raise ValueError('ECO plan needs an even logical expert count, equal ranks <= 64 and <= 2048 slots')
    defines = {
        'ECO_P': num_physical, 'ECO_R': num_ranks, 'ECO_E': num_logical,
        'ECO_THREADS': ECO_UNIT_THREADS,
        'OFF_EPOCH': SIGNAL_OFFSETS['recv_parity'],
        'OFF_ECO_COUNTS_READY': eco_base + ECO_SIGNAL_OFFSETS['eco_counts_ready'],
        'OFF_ECO_SLOT_COUNT': eco_base + ECO_SIGNAL_OFFSETS['eco_slot_count'],
        'OFF_ECO_NUM_FIXED': eco_base + ECO_SIGNAL_OFFSETS['eco_num_fixed'],
        'OFF_ECO_REPS_READY': eco_base + ECO_SIGNAL_OFFSETS['eco_reps_ready'],
        'OFF_SOURCE_IDX': SIGNAL_OFFSETS['size'],
        'OFF_TOPK': topk_offset, 'ECO_TOPK': topk, 'ECO_CAPACITY': capacity,
        'OFF_ECO_PLAN_READY': eco_base + ECO_SIGNAL_OFFSETS['eco_plan_ready'],
        'OFF_ECO_QUOTA': eco_base + ECO_SIGNAL_OFFSETS['eco_quota'],
        'OFF_ECO_QUOTA_START': eco_base + ECO_SIGNAL_OFFSETS['eco_quota_start'],
        'OFF_ECO_COUNTS': eco_base + ECO_SIGNAL_OFFSETS['eco_counts'],
        'OFF_ECO_TRACE': eco_base + ECO_SIGNAL_OFFSETS['eco_trace'],
        'OFF_ECO_TRACE_RING': eco_base + ECO_SIGNAL_OFFSETS['eco_trace_ring'],
        'ECO_TRACE': int(trace),
    }
    defines.update(zip(('UNIT_BM', 'UNIT_WB', 'UNIT_WA', 'UNIT_NR', 'UNIT_NRC', 'UNIT_HOP2'), unit))
    flags = [f'-D{k}={v}u' if k.startswith('OFF_') else f'-D{k}={v}' for k, v in sorted(defines.items())]
    key = hashlib.sha256((_source_digest() + ' '.join(flags) + arch).encode()).hexdigest()[:32]
    path = os.path.join(_cache_dir(), f'eco_plan_{key}.cubin')
    if not os.path.exists(path):
        # Temporary file in the cache directory: the rename stays on one filesystem (atomic)
        fd, out = tempfile.mkstemp(suffix='.cubin', dir=os.path.dirname(path))
        os.close(fd)
        try:
            cmd = [_nvcc(), '-cubin', '-std=c++17', '-O3', f'-arch={arch}', '-lineinfo', *flags,
                   os.path.join(_SOURCE_DIR, 'eco_unit_kernel.cu'),
                   '-o', out]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f'ECO plan compilation failed:\n{" ".join(cmd)}\n{result.stderr}')
            os.replace(out, path)
        finally:
            if os.path.exists(out):
                os.remove(out)
    with open(path, 'rb') as f:
        return f.read()

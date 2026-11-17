"""CuTe DSL source for three persistent GC MoE roles and four worker phases."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import math
from cutlass import Float32, Int32, Uint8, Uint16, Uint32, Uint64

from ..comm.gc_barrier import (
    DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
    grid_sync,
    nvlink_barrier,
    wait_sys_u64_eq,
)
from ..common.math import (
    bf16x2_high_f32_bits,
    bf16x2_max,
    bf16x2_min,
    bits_float,
    f32_to_bf16_bits,
    f32x2_to_bf16x2_bits,
    fast_exp_neg_f32,
    fast_rcp,
    abs_f32,
    f32x2_to_e4m3x2_bits,
    fmax_f32,
    pack_bf16_pair,
    ue8m0_fp8_inv,
    ue8m0_fp8_scale_exp,
    warp_group4_max,
)
from ..common.tma_copy import load_mixed_k128_tma, use_small_tile_path
from ..layout.gc_moe import (
    ECO_MAX_ROUTES,
    ECO_SIGNAL_OFFSETS,
    ECO_TRACE_RING_FIELDS,
    ACCUM_BARRIER_STAGES,
    COMBINE_WARPS,
    FINALIZE_THREADS,
    GCBufferLayout,
    MAX_RANKS,
    SIGNAL_OFFSETS,
    eco_recv_counts_offset,
)
from ..layout.sym_buffer import mapped_peer_base
from ..mma.sm100 import (
    MixedMmaTile,
    blockscaled_instr_desc_hi,
    issue_mixed_k128,
    kmajor_smem_desc,
    kmajor_smem_desc_from_start,
    select_weight_format,
)
from ..ptx.ld_st import (
    st_release_u32,
    EVICT_FIRST,
    EVICT_NORMAL,
    atom_add_u64,
    atom_release_gpu_add_u32,
    atom_shared_add_u32,
    atom_sys_add_u64,
    bulk_zero_shared,
    ld_acquire_gpu_u32,
    ld_acquire_gpu_u64,
    ld_acquire_sys_u32,
    ld_acquire_sys_u64,
    ld_global_u32x3,
    ld_global_u32x4,
    ld_global_u64,
    ld_shared_u32x4,
    ld_volatile_u32,
    ld_volatile_u64,
    atom_max_u32_global,
    red_global_max_u64,
    red_global_min_u64,
    red_gpu_add_s32,
    red_gpu_add_u32,
    red_release_gpu_add_u32,
    red_release_gpu_xor_u64,
    red_release_sys_add_u64,
    st_global_u32,
    st_global_u32x4,
    st_global_u64,
    st_release_sys_u64,
    st_shared_u32x4,
    stmatrix_bf16x8_trans,
    stmatrix_fp8x4_trans_late,
)
from ..ptx.tcgen05 import (
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tmem_free_2cta,
    tmem_load_16dp256b4,
)
from ..ptx.tma import (
    bulk_g2s_1d,
    bulk_s2g_1d,
    generic_tensormap_ptr,
    prefetch_tensormap,
    tile_s2g_2d,
    tma_store_arrive,
    tma_store_fence,
    tma_store_wait_0,
    tma_store_wait_1,
)
from ..ptx.utils import (
    accumulate_bf16_into_f32,
    assume_u32_lt,
    device_trap,
    elect_one_sync,
    fence_acq_rel_sys,
    read_clock64,
    prefetch_global_l2,
    read_globaltimer,
    sync_aligned,
    sync_unaligned,
    undefined_u32,
    warp_first_set_bit,
    warp_nth_set_bit,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)
from ..scheduler.gc_moe import (
    GCBlockPhase,
    TASK_EMPTY_CONSUMERS,
    advance_expert,
    cached_num_tokens,
    eco_wait_plan,
    release_task_info,
    schedule_tasks,
)


# A fixed small-decode floor, independent of the local expert count: 64 CTAs
# provide up to 512 combine warps before clamping to the compute SM budget.
SMALL_DECODE_FINALIZE_SMS = 64


@cute.jit
def _swiglu_pair(gate0_accum: Float32, gate1_accum: Float32,
                up0_accum: Float32, up1_accum: Float32,
                weight0: Float32, weight1: Float32,
                clamp_value: Float32, alpha: Float32,
                clamp_enabled: cutlass.Constexpr,
                oai_enabled: cutlass.Constexpr,
                fast_math: cutlass.Constexpr):
    """Reference W1 float2 SwiGLU, preserving the BF16x2 and FP32 order."""
    gate_bits = f32x2_to_bf16x2_bits(gate0_accum, gate1_accum)
    up_bits = f32x2_to_bf16x2_bits(up0_accum, up1_accum)
    if cutlass.const_expr(clamp_enabled):
        bound = Uint32(f32_to_bf16_bits(clamp_value))
        bound_pair = bound | (bound << Uint32(16))
        gate_bits = bf16x2_min(gate_bits, bound_pair)
        up_bits = bf16x2_max(up_bits, bound_pair ^ Uint32(0x80008000))
        up_bits = bf16x2_min(up_bits, bound_pair)
    gate0 = bits_float((gate_bits & Uint32(0xFFFF)) << Uint32(16))
    gate1 = bits_float(bf16x2_high_f32_bits(gate_bits))
    sigmoid0, sigmoid1 = gate0, gate1
    if cutlass.const_expr(oai_enabled):
        sigmoid0, sigmoid1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (alpha, alpha), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        exp0 = fast_exp_neg_f32(sigmoid0)
        exp1 = fast_exp_neg_f32(sigmoid1)
    else:
        exp0 = cute.math.exp(-sigmoid0, fastmath=False)
        exp1 = cute.math.exp(-sigmoid1, fastmath=False)
    denom0, denom1 = cute.arch.add_packed_f32x2(
        (Float32(1), Float32(1)), (exp0, exp1), rnd='rn', ftz=False)
    if cutlass.const_expr(fast_math):
        gate0, gate1 = cute.arch.mul_packed_f32x2(
            (gate0, gate1), (fast_rcp(denom0), fast_rcp(denom1)), rnd='rn', ftz=False)
    else:
        gate0 = cute.math.div(
            gate0, denom0, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
        gate1 = cute.math.div(
            gate1, denom1, fastmath=False, ftz=False,
            rounding=cute.math.RoundingMode.NEAREST_EVEN)
    up0 = bits_float((up_bits & Uint32(0xFFFF)) << Uint32(16))
    up1 = bits_float(bf16x2_high_f32_bits(up_bits))
    if cutlass.const_expr(oai_enabled):
        up0, up1 = cute.arch.add_packed_f32x2(
            (up0, up1), (Float32(1), Float32(1)), rnd='rn', ftz=False)
    activation0, activation1 = cute.arch.mul_packed_f32x2(
        (gate0, gate1), (up0, up1), rnd='rn', ftz=False)
    return cute.arch.mul_packed_f32x2(
        (activation0, activation1), (weight0, weight1), rnd='rn', ftz=False)


@cute.jit
def sync_epilogue_warpgroup(warpgroup: Uint32) -> None:
    """Reference WG-local barrier ID 3 + warpgroup, with 128 callers."""
    sync_aligned(128, Uint32(3) + warpgroup)


@cute.jit
def _load_w2_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Keep both 16dp halves of one 8-row W2 atom in registers."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w2_tmem(accum_empty: cute.Pointer, stage: Uint32) -> None:
    tcgen05_before_thread_sync()
    cute.arch.mbarrier_arrive(accum_empty + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def _load_w1_atom(raw_values: cute.Tensor, buffer: cutlass.Constexpr,
                  atom: cutlass.Constexpr, address: Uint32) -> None:
    """Source-contract 16dp W1 atom: gate/up for two post channels."""
    first = tmem_load_16dp256b4(address)
    second = tmem_load_16dp256b4(address | Uint32(0x00100000))
    for register in cutlass.range_constexpr(4):
        raw_values[buffer, atom, register] = first[register]
        raw_values[buffer, atom, register + 4] = second[register]


@cute.jit
def _release_w1_tmem(accum_empty: cute.Pointer, stage: Uint32) -> None:
    tcgen05_before_thread_sync()
    cute.arch.mbarrier_arrive(accum_empty + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def _w1_pack_fp8x4(a: Float32, b: Float32,
                    c: Float32, d: Float32) -> Uint32:
    """Pack reference channel0/token0, channel0/token1, channel1/token0/1."""
    ab = f32x2_to_e4m3x2_bits(a, b)
    cd = f32x2_to_e4m3x2_bits(c, d)
    return Uint32(ab) | (Uint32(cd) << Uint32(16))


@cute.jit
def _load_w2_source_metadata(base: cute.Pointer, row_in_block: Uint32,
                              valid_m: Uint32, pool_token: Uint32,
                              rank_idx: Uint32, is_shared,
                              topk: cutlass.Constexpr,
                              has_shared: cutlass.Constexpr):
    """Read valid rows with fresh undefined values for inactive rows."""
    src_rank = undefined_u32()
    src_token = undefined_u32()
    topk_slot = undefined_u32()
    if row_in_block < valid_m:
        if cutlass.const_expr(has_shared):
            src_rank = rank_idx
            src_token = pool_token
            topk_slot = Uint32(topk)
            if not is_shared:
                src_rank, src_token, topk_slot = ld_global_u32x3(
                    base + pool_token * Uint32(3))
        else:
            src_rank, src_token, topk_slot = ld_global_u32x3(
                base + pool_token * Uint32(3))
    return src_rank, src_token, topk_slot


@cute.jit
def _initialize_private_combine_bar(bar: cute.Pointer) -> None:
    """Publish one warp-owned barrier before that warp's first use."""
    if elect_one_sync():
        cute.arch.mbarrier_init(bar, 1)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_warp()


class GCMoERoleKernel:
    """Keep the role on the Python object so CuTe erases other role paths."""

    def __init__(self, role: int, dispatch_sms: int = 1, compute_sms: int = 2,
                 block_m: int = 16, hidden: int = 256,
                 intermediate: int = 128, num_tokens: int = 16,
                 num_ranks: int = 1, local_experts: int = 1,
                 shared_experts: int = 0, capacity: int = 1920,
                 shared_weight_gran_k: int = 32,
                 finalize_output: bool = False,
                 ring_tokens: int = 1920,
                 sf_ring_tokens: int = 30720,
                 topk: int = 1,
                 activation_clamp: float = math.inf,
                 fast_math: bool = True,
                 has_stats: bool = False,
                 nvlink_timeout_secs: int = DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS,
                 *, role_config, combine_geometry: tuple[int, int, int] | None = None,
                 routed_weight_linger: bool = False,
                 eco_logical_experts: int = 0, eco_trace: bool = False,
                 eco_rows_after_pub: bool = False, eco_route_prep: bool = False,
                 eco_spec: bool = False,
                 cta_base: int = 0):
        if role_config.role != role or role_config.block_m != block_m:
            raise ValueError('kernel role/block M differs from its host configuration')
        math_stages = role_config.num_stages
        if role not in (1, 2, 3):
            raise ValueError('role must be dispatch, compute or finalize')
        if local_experts < 1 or local_experts > 512 or \
                num_ranks * local_experts > 2048 or topk < 1 or \
                topk + bool(shared_experts) > 32:
            raise NotImplementedError('device scheduler requires 1..512 local experts, <=2048 global experts, topk>=1 and topk+shared slot<=32')
        if num_ranks < 1 or num_ranks > MAX_RANKS or not 0 <= num_tokens <= capacity:
            raise NotImplementedError('dispatch routing requires 1..72 ranks within capacity')
        if math_stages < 2 or math_stages > 32:
            raise ValueError('GC math pipeline needs 2..32 TMA stages')
        if not 0 < nvlink_timeout_secs <= 2_147_483_647:
            raise ValueError('NVLink barrier timeout must fit positive INT_MAX seconds')
        self.role = role
        # ECO online split: `topk_idx` holds logical experts, every one of the `num_ranks * local_experts`
        # physical slots serves one logical expert (EcoMeta), single-copy slots go first on every rank and
        # replica slots take the quotas of the plan role (role 4)
        self.eco = eco_logical_experts > 0
        self.eco_rows_after_pub = self.eco and role == 1 and bool(eco_rows_after_pub)
        self.eco_route_prep = self.eco and role == 1 and bool(eco_route_prep)
        # Timing probe: speculate replica pulls/tasks with the previous call's published
        # quotas instead of waiting for this call's plan (wrong output, do not check).
        self.eco_spec = self.eco and bool(eco_spec)
        # Quota/start banks are indexed by call parity inside the same fields; spec reads
        # bank[epoch&1^1], the previous call's plan. Falls back to a single bank when the
        # field cannot hold both.
        self.eco_qbank = 2 * num_ranks * local_experts <= 2048
        self.eco_qbank_slots = (num_ranks * local_experts) if self.eco_qbank else 0
        if self.eco_spec and not self.eco_qbank:
            raise ValueError('speculative ECO needs both quota banks to fit the field')
        self.eco_logical = eco_logical_experts
        if self.eco and (eco_logical_experts % 8 or (num_ranks * local_experts) % 8 or
                         eco_logical_experts >= num_ranks * local_experts):
            raise ValueError('ECO needs logical and physical expert counts in multiples of 8, with replicas')
        self.route_experts = eco_logical_experts if self.eco else num_ranks * local_experts
        # EcoMeta int32 offsets: header 16, copies[E][32] i16 (two per int32), phy2log[P], num_fixed[R]
        self.eco_copies_words = 16
        self.eco_phy2log_words = 16 + eco_logical_experts * 16
        self.eco_num_fixed_words = self.eco_phy2log_words + num_ranks * local_experts
        # The dispatch-only metadata ends at this fixed offset. The planner's variable-length unit tables
        # live in its original metadata, so different placements share the same dispatch binary.
        self.eco_slot_info_words = self.eco_num_fixed_words + num_ranks
        # Timeline trace (globaltimer ns into the ECO region's 'eco_trace', see ECO_TRACE_SLOTS in eco/plan.py)
        # Decoupled from `self.eco`: role-boundary stamps are useful for plain mode too (the buffer
        # must then still be allocated with an ECO region to hold the trace bank).
        self.eco_trace = bool(eco_trace)
        self.dispatch_sms = dispatch_sms
        self.compute_sms = compute_sms
        self.cta_base = cta_base
        self.block_m = block_m
        self.num_tokens = num_tokens
        self.hidden = hidden
        self.intermediate = intermediate
        # Base grid; only role3 with finalize_output overrides it below.
        self.finalize_sms = (min(compute_sms, SMALL_DECODE_FINALIZE_SMS)
                             if num_tokens <= 128 else compute_sms)
        self.num_ranks = num_ranks
        self.local_experts = local_experts
        self.topk = topk
        self.has_activation_clamp = not math.isinf(activation_clamp)
        self.activation_clamp = activation_clamp if self.has_activation_clamp else 0.0
        self.fast_math = fast_math
        self.has_stats = has_stats
        self.shared_experts = shared_experts
        self.use_small_tile_path = use_small_tile_path(shared_experts, block_m)
        self.capacity = capacity
        self.finalize_output = finalize_output
        self.shared_weight_gran_k = shared_weight_gran_k
        # Use FIRST for shared weights/SFs with larger decode tiles and
        # at most one routed block per expert on average. Keep BM16 NORMAL.
        self.shared_weight_cache_hint = (
            EVICT_FIRST if block_m >= 32 and
            num_tokens * topk <= block_m * local_experts
            else EVICT_NORMAL)
        self.math_stages = math_stages
        self.nvlink_timeout_secs = nvlink_timeout_secs
        self.routed_weight_linger = routed_weight_linger
        # Block configuration and the shared C/D storage union.
        self.store_m_l1 = role_config.store_m_l1
        self.store_m_l2 = role_config.store_m_l2
        self.cd_l1_bytes = 2 * 2 * self.store_m_l1 * 64
        self.cd_l2_bytes = 2 * self.store_m_l2 * 128 * 2
        self.cd_bytes = max(self.cd_l1_bytes, self.cd_l2_bytes)
        # float2[kNumEpilogueWarps][STORE_BLOCK_M_L1 / 2].
        self.amax_float_count = 8 * self.store_m_l1
        self.non_epi_registers = 40 if local_experts <= 64 else 72
        self.epi_registers = 208 if local_experts <= 64 else 168
        tile = MixedMmaTile(block_m, hidden, intermediate)
        self.l1_clusters = tile.l1_clusters
        self.l2_clusters = tile.l2_clusters
        self.l1_full_mask = (1 << tile.l1_n_blocks) - 1
        self.tmem_cols = tile.tmem_cols
        self.accum_stages = tile.epilogue_stages
        if self.accum_stages > ACCUM_BARRIER_STAGES:
            raise ValueError('epilogue stages exceed the reserved accumulator barriers')
        self.sf_block_m = role_config.sf_block_m
        self.sfa_tmem_cols = tile.sfa_tmem_cols
        self.dispatch_warps = role_config.routing_warps
        # The double-buffered pull pipeline keeps one token's resolved state
        # live across the next token's remote transfer; grant its warps a
        # deeper register budget.
        dispatch_registers = (64 if role_config.pull_stages == 2 else 48) \
            if local_experts <= 64 else 96
        self.dispatch_registers = dispatch_registers
        role_registers = (
            self.dispatch_warps * 32 * dispatch_registers if role == 1 else
            128 * self.non_epi_registers + 256 * self.epi_registers if role == 2 else
            128 * dispatch_registers + (COMBINE_WARPS * 32) * self.epi_registers)
        if role_registers > 64512:
            raise ValueError('role register allocation exceeds the SM budget')
        self.pull_bytes = role_config.pull_bytes
        self.pull_stages = role_config.pull_stages
        self.pull_chunks = hidden // self.pull_bytes
        # GC_MOE_PULL_PIPE=0 keeps the widened staging but runs the serial loop.
        self.pull_pipe = __import__('os').environ.get('GC_MOE_PULL_PIPE', '1') == '1'
        if role == 3 and finalize_output:
            if combine_geometry is None:
                raise ValueError('finalize requires host combine geometry')
            (self.combine_chunks, self.combine_chunk_bytes,
             self.combine_uint4_per_lane) = combine_geometry
            # Size fused output work independently of the expert-cleanup grid.
            self.finalize_sms = min(
                compute_sms,
                max(SMALL_DECODE_FINALIZE_SMS if num_tokens <= 128 else
                    min(compute_sms, local_experts + 1),
                    (num_tokens * self.combine_chunks + (COMBINE_WARPS - 1)) //
                    COMBINE_WARPS))
        self.shared_layout = role_config.shared_layout
        layout = GCBufferLayout(num_ranks, num_ranks * local_experts, capacity, topk,
                                hidden, intermediate, ring_tokens, sf_ring_tokens,
                                shared_experts, eco=self.eco)
        # ECO region of the buffer (`GCBufferLayout.eco_base`, eco layouts only): signals and the count table
        self.eco_base = layout.eco_base if (self.eco or self.eco_trace) else 0
        self.eco_counts_bytes = self.eco_base + ECO_SIGNAL_OFFSETS['eco_counts'] if self.eco else 0
        self.eco_recv_count_base = self.eco_base + \
            eco_recv_counts_offset(num_ranks, num_ranks * local_experts) if self.eco else 0
        # Replica receive-count source (bisect switch, default = production): 3 = dedicated u64
        # rows the planner materializes in the quiet ECO region — uniform native-format rows,
        # single ld_global_u64 per source, no tag poll; ~2.5 us faster than the packed table and
        # ~3 us faster than the routing-flooded native region. Debug only: 0 = packed plan table,
        # 1 = native expert_recv_count rows, 2 = same rows via volatile loads,
        # 4 = no row read at all (constant counts — wrong output, timing probe).
        self.eco_recv_counts = int(__import__('os').environ.get(
            'GC_MOE_ECO_RECV_COUNTS', '3')) if self.eco else 3
        # Timing probe (wrong output): replica slots skip the canonical index read and synthesize a token id
        self.eco_idx_probe = self.eco and __import__('os').environ.get('GC_MOE_ECO_IDX_PROBE', '0') == '1'
        # Pull this warp's replica tokens as soon as the plan is visible (between its fixed tokens), then
        # resume the remaining fixed tokens: replica pulls no longer queue behind the whole fixed range
        self.eco_interleave = self.eco and __import__('os').environ.get('GC_MOE_ECO_INTERLEAVE', '0') == '1'
        # Token -> (source rank, source slot) by binary search over the peel level (~log2(max count) warp sums)
        # instead of walking the peel rounds from position 0 (deep positions = many rounds)
        self.peel_bs = __import__('os').environ.get('GC_MOE_PEEL_BS', '0') == '1'
        # Split pull (ECO): the last GC_MOE_ECO_SPLIT_PULL dispatch warps of each SM pull only replica tokens (from the
        # plan's arrival, alongside the fixed pulls); the others pull only fixed tokens in the unchanged pool order
        self.eco_split_pull = int(__import__('os').environ.get('GC_MOE_ECO_SPLIT_PULL', '0')) if self.eco else 0
        if role == 1 and self.eco_split_pull and not (role_config.pull_stages == 2 and role_config.pull_bytes == hidden):
            raise ValueError(f'GC_MOE_ECO_SPLIT_PULL={self.eco_split_pull} needs the double-buffered pull pipeline, '
                             f'which does not fit shared memory here (pull_stages={role_config.pull_stages}, '
                             f'{role_config.routing_warps} dispatch warps): lower GC_MOE_ROUTING_WARPS or unset the split')
        # Timing probe only (output may be wrong): ECO combine uses the plain per-token readiness wait
        # (peer = routed id // local slots) instead of acquiring every peer
        self.eco_combine_probe = self.eco and __import__('os').environ.get('GC_MOE_ECO_COMBINE_PROBE') == '1'
        # Exact combine readiness (default): each route waits only for the rank that computes it. Every rank's plan
        # is identical, so the source derives it locally: canonical position (round-robin over sources) of its
        # slot, then the copy whose [quota_start, quota_start + quota) holds it.
        # 2 (default): the plan CTA derives the peers after its publish; 1: role1 after its pulls; 0: all peers
        _exact = int(__import__('os').environ.get('GC_MOE_ECO_COMBINE_EXACT', '2'))
        self.eco_combine_exact = (_exact if self.eco and not self.eco_combine_probe and
                                  capacity * topk <= ECO_MAX_ROUTES else 0)
        self.eco_route_verify = self.eco_combine_exact and \
            __import__('os').environ.get('GC_MOE_ECO_ROUTE_VERIFY') == '1'
        self.topk_offset_words = layout.views['topk_idx'].offset // 4
        self.input_x_words = layout.views['x'].offset // 4
        self.input_sf_words = layout.views['x_sf'].offset // 4
        self.input_weight_words = layout.views['topk_weights'].offset // 4
        self.l1_x_words = layout.views['l1_acts'].offset // 4
        self.l1_sf_words = layout.views['l1_acts_sf'].offset // 4
        self.l1_weight_words = layout.views['l1_topk_weights'].offset // 4
        self.l2_x_words = layout.views['l2_acts'].offset // 4
        self.l2_sf_words = layout.views['l2_acts_sf'].offset // 4
        self.shared_l2_x_words = layout.views['shared_l2_acts'].offset // 4
        self.shared_l2_sf_words = layout.views['shared_l2_acts_sf'].offset // 4
        self.shared_sf_tokens = ((capacity + 7) // 8) * 128
        self.combine_words = layout.views['combine_tokens'].offset // 4
        self.metadata_words = layout.workspace_offsets['token_src_metadata'] // 4
        self.ring_tokens = layout.ring_tokens
        self.sf_ring_tokens = layout.sf_ring_tokens
        self.ring_blocks = layout.ring_tokens // block_m
        sf_block_m = self.sf_block_m
        sf_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                         for t in range(block_m))
        sf_even_row_max = max((t & ~127) + (t & 31) * 4 + ((t >> 5) & 3)
                              for t in range(0, block_m, 2))
        last_ring_sf_row = (self.ring_blocks - 1) * sf_block_m
        l1_sf_local_max = ((hidden // 128 - 1) * self.sf_ring_tokens +
                           last_ring_sf_row + sf_row_max)
        routed_w1_sf_local_max = 4 * (
            (intermediate // 128 - 1) * self.sf_ring_tokens +
            last_ring_sf_row + sf_even_row_max) + 19
        self.wide_l1_sf = (l1_sf_local_max > 0xffffffff or
                           self.l1_sf_words + l1_sf_local_max > 0xffffffff)
        self.wide_routed_w1_sf = routed_w1_sf_local_max > 0xffffffff
        self.wide_shared_w1_sf = False
        if shared_experts > 0:
            last_shared_sf_row = ((capacity + block_m - 1) // block_m - 1) * sf_block_m
            shared_w1_sf_local_max = 4 * (
                (shared_experts * intermediate // 128 - 1) *
                self.shared_sf_tokens + last_shared_sf_row + sf_even_row_max) + 19
            self.wide_shared_w1_sf = shared_w1_sf_local_max > 0xffffffff

    @cute.jit
    def _initialize_shared_storage(self, slab: cute.Pointer,
                                    physical_warp: Uint32, lane: Uint32) -> None:
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if physical_warp == 0:
            if elect_one_sync():
                if cutlass.const_expr(self.role == 1):
                    bulk_zero_shared(
                        slab + self.shared_layout.expert_counts,
                        Uint32(((self.num_ranks * self.local_experts * 4 + 1023) //
                                1024) * 1024))
        elif physical_warp == 1:
            if cutlass.const_expr(self.role == 1):
                bars = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
                for index in cutlass.range(
                        Int32(lane), Int32(self.dispatch_warps * self.pull_stages),
                        Int32(32), unroll_full=True):
                    cute.arch.mbarrier_init(bars + index, 1)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 2:
            # CuTe 4.8.1's single-thread-per-CTA docstring is conservative
            # for disjoint barriers; the wrapper does not elect a lane.
            # Each object below has exactly one initializer. Every such
            # lane executes the init fence before the cluster rendezvous
            # publishes initialization to consumers. The constructor's
            # math_stages <= 32 bound keeps all stages within one warp.
            if cutlass.const_expr(self.role == 2):
                cta_rank, _, _ = cute.arch.block_in_cluster_idx()
                data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
                data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
                if lane < self.math_stages:
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(data_full + lane, 4)
                    cute.arch.mbarrier_init(data_empty + lane, 1)
                accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
                accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
                if lane < self.accum_stages:
                    cute.arch.mbarrier_init(accum_full + lane, 1)
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(accum_empty + lane, TASK_EMPTY_CONSUMERS)
                full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
                empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
                if lane < 2:
                    cute.arch.mbarrier_init(full + lane, 1)
                    if cta_rank == 0:
                        cute.arch.mbarrier_init(empty + lane, TASK_EMPTY_CONSUMERS)
            elif cutlass.const_expr(self.role == 3):
                combine = cute.recast_ptr(slab + self.shared_layout.combine_bars, dtype=Uint64)
                if lane < COMBINE_WARPS:
                    cute.arch.mbarrier_init(combine + lane, 1)
            cute.arch.mbarrier_init_fence()
        elif physical_warp == 3:
            if cutlass.const_expr(self.role == 2):
                tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
                cute.arch.alloc_tmem(self.tmem_cols, tptr, is_two_cta=True)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def _eco_trace(self, signals: cute.Tensor, slot: cutlass.Constexpr, mode: cutlass.Constexpr) -> None:
        """Timeline trace point (one issuing thread): mode 0 store, 1 min, 2 max"""
        if cutlass.const_expr(self.eco_trace):
            ptr = signals.iterator + ((self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) + slot * 8) // 4
            now = read_globaltimer()
            if cutlass.const_expr(mode == 0):
                st_global_u64(ptr, now)
            elif cutlass.const_expr(mode == 1):
                red_global_min_u64(ptr, now)
            else:
                red_global_max_u64(ptr, now)

    @cute.jit
    def _eco_trace_accum(self, signals: cute.Tensor, slot: cutlass.Constexpr) -> None:
        """Graph-replay trace (one issuing thread): accumulate `now - this call's role1 start` into `slot`"""
        if cutlass.const_expr(self.eco_trace):
            base = signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4
            start = ld_volatile_u64(base + 24 * 2)
            atom_add_u64(base + slot * 2, read_globaltimer() - start)

    @cute.jit
    def _eco_trace_ring(self, signals: cute.Tensor, field: cutlass.Constexpr, is_max: cutlass.Constexpr) -> None:
        """Per-call trace ring entry field (one issuing thread): ns since this call's role1 start"""
        if cutlass.const_expr(self.eco_trace):
            base = signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4
            elapsed = Uint32(read_globaltimer() - ld_volatile_u64(base + 24 * 2))
            entry = Uint32(ld_volatile_u64(base + 33 * 2)) % Uint32(64)
            ptr = signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace_ring']) // 4 + entry * Uint32(ECO_TRACE_RING_FIELDS) + field
            if cutlass.const_expr(field == 13 or field == 14):
                elapsed = Uint32(0xffffffff) - elapsed
            if cutlass.const_expr(is_max):
                atom_max_u32_global(ptr, elapsed)
            else:
                st_global_u32(ptr, elapsed)

    @cute.jit
    def _eco_table_count(self, signals: cute.Tensor, gen: Uint32, source: Uint32,
                         expert: Uint32, tag: Uint32) -> Uint32:
        """ECO packed count table word [gen][source][expert/3]: u32 (tag<<30)|c0|c1<<10|c2<<20
        written by the source's plan CTA (row stride: physical slots * 2 bytes)."""
        row_words = Uint32(self.num_ranks * self.local_experts // 2)
        ptr = signals.iterator + self.eco_counts_bytes // 4 + \
            (gen * Uint32(self.num_ranks) + source) * row_words + expert // Uint32(3)
        entry = ld_volatile_u32(ptr)
        while (entry >> Uint32(30)) != tag:
            entry = ld_volatile_u32(ptr)
        return (entry >> ((expert % Uint32(3)) * Uint32(10))) & Uint32(0x3ff)

    @cute.jit
    def _eco_route_peers(self, signals: cute.Tensor, tensor_maps: cute.Tensor, routes: cute.Tensor,
                         rank_idx: Uint32, num_tokens: Uint32, lane: Uint32, global_warp: Uint32,
                         gen: Uint32, epoch: Uint32) -> None:
        """Exact combine readiness: overwrite each local route's recorded canonical slot j with the rank that
        computes it. Single-copy expert: its copy's rank. Replicated: canonical position
        p = sum_r min(c_r, j) + #{r < self : c_r > j} (round-robin over sources, ascending rank), then the active
        copy with quota_start <= p < quota_start + quota (identical plan on every rank, so purely local)."""
        eco_wait_plan(signals.iterator, epoch, False, self.eco_base, True)
        qbank = epoch & Uint32(1)
        tag = (epoch // Uint32(2)) % Uint32(3) + Uint32(1)
        peer_off = (self.eco_base + ECO_SIGNAL_OFFSETS['eco_route_peer']) // 4
        quota_off = (self.eco_base + ECO_SIGNAL_OFFSETS['eco_quota']) // 4 + qbank * Uint32(self.eco_qbank_slots)
        start_off = (self.eco_base + ECO_SIGNAL_OFFSETS['eco_quota_start']) // 4 + \
            qbank * Uint32(self.eco_qbank_slots)
        rp_route = global_warp
        while rp_route < num_tokens * Uint32(self.topk):
            rp_expert = routes[rp_route]
            if rp_expert < Uint64(self.route_experts):
                rp_e = Uint32(rp_expert)
                rp_word = Uint32(tensor_maps[Uint32(self.eco_copies_words) + rp_e * Uint32(16) +
                                             lane // Uint32(2)])
                rp_copy = (rp_word >> ((lane % Uint32(2)) * Uint32(16))) & Uint32(0xffff)
                rp_active = rp_copy < Uint32(0x8000)        # -1 padding and passive copies have bit 15 set
                rp_amask = Uint32(cute.arch.vote_ballot_sync(rp_active))
                rp_peer = Uint32(0)
                if Uint32(cute.arch.popc(rp_amask)) == Uint32(1):
                    rp_peer = cute.arch.shuffle_sync(rp_copy, warp_first_set_bit(rp_amask)) // \
                        Uint32(self.local_experts)
                else:
                    rp_j = Uint32(signals[peer_off + rp_route])
                    rp_pos = Uint32(0)
                    for rp_wave in cutlass.range_constexpr((self.num_ranks + 31) // 32):
                        rp_src = Uint32(rp_wave * 32) + lane
                        rp_c = Uint32(0)
                        if rp_src < Uint32(self.num_ranks):
                            rp_c = self._eco_table_count(signals, gen, rp_src, rp_e, tag)
                        rp_add = min(rp_c, rp_j)
                        if rp_src < rank_idx and rp_c > rp_j:
                            rp_add += Uint32(1)
                        rp_pos += cute.arch.warp_redux_sync(rp_add, 'add')
                    rp_hit = False
                    if rp_active:
                        rp_st = Uint32(signals[start_off + rp_copy])
                        rp_hit = rp_pos >= rp_st and rp_pos < rp_st + Uint32(signals[quota_off + rp_copy])
                    rp_hmask = Uint32(cute.arch.vote_ballot_sync(rp_hit))
                    rp_peer = cute.arch.shuffle_sync(rp_copy, warp_first_set_bit(rp_hmask)) // \
                        Uint32(self.local_experts)
                if lane == Uint32(0):
                    signals[peer_off + rp_route] = rp_peer
            cute.arch.sync_warp()
            rp_route += Uint32(self.dispatch_sms * self.dispatch_warps)

    @cute.jit
    def _eco_slot_total(self, signals: cute.Tensor, rank_idx: Uint32, slot_idx: Uint32,
                        num_fixed: Uint32, recv_ptr: cute.Pointer,
                        spec_epoch: Uint32 = None) -> Uint32:
        """Finalize's slot total: native receive sum for a single-copy slot, quota for a replica."""
        total = Uint32(0)
        if slot_idx < num_fixed:
            # volatile: remote atomics update the counter out-of-band; a cached L1 line could go stale
            total = Uint32(ld_volatile_u64(recv_ptr + slot_idx * Uint32(2)))
        else:
            qbank = spec_epoch & Uint32(1)
            if cutlass.const_expr(self.eco_spec):
                qbank = (spec_epoch + Uint32(1)) & Uint32(1)
            total = Uint32(signals[(self.eco_base + ECO_SIGNAL_OFFSETS['eco_quota']) // 4 +
                                   qbank * Uint32(self.eco_qbank_slots) +
                                   rank_idx * Uint32(self.local_experts) + slot_idx])
        return total

    @cute.jit
    def _eco_load_plan(self, signals: cute.Tensor, rank_idx: Uint32, epoch: Uint32,
                       num_fixed: Uint32, expert_counts: cute.Tensor, lane: Uint32,
                       expert_starts: cute.Tensor = None) -> None:
        """ECO: wait for the plan role of this call, then cache the replica slots' quotas
        (and, when `expert_starts` is given, their canonical-order starts). Replica index
        reads go to the canonical copy's rows on its owner rank — complete since the rank
        barrier — so there is no replica-row readiness to wait for here."""
        if cutlass.const_expr(not self.eco_spec):
            eco_wait_plan(signals.iterator, epoch, self.eco_trace, self.eco_base, True)
        if elect_one_sync():
            self._eco_trace_ring(signals, 2, True)  # field 2: last dispatch warp saw the plan
        if elect_one_sync():
            self._eco_trace(signals, 7, 2)
            self._eco_trace_ring(signals, 5, True)  # field 5: last dispatch warp saw every source's replica indices
        qbank = epoch & Uint32(1)
        if cutlass.const_expr(self.eco_spec):
            qbank = (epoch + Uint32(1)) & Uint32(1)
        for wave in cutlass.range_constexpr((self.local_experts + 31) // 32):
            slot_idx = Uint32(wave * 32) + lane
            if slot_idx >= num_fixed and slot_idx < Uint32(self.local_experts):
                expert_counts[wave] = Uint32(signals[(self.eco_base + ECO_SIGNAL_OFFSETS['eco_quota']) // 4 +
                                                     qbank * Uint32(self.eco_qbank_slots) +
                                                     rank_idx * Uint32(self.local_experts) + slot_idx])
                if expert_starts is not None:
                    expert_starts[wave] = Uint32(
                        signals[(self.eco_base + ECO_SIGNAL_OFFSETS['eco_quota_start']) // 4 +
                                qbank * Uint32(self.eco_qbank_slots) +
                                rank_idx * Uint32(self.local_experts) + slot_idx])
        cute.arch.sync_warp()

    @cute.jit
    def _pull_drain(self, signals: cute.Tensor, warp_pull: cute.Pointer,
                    warp_bar: cute.Pointer, pull_phase: Uint32, pull_phase2: Uint32,
                    pend: Uint32, pend_issued: Uint32, pend_par: Uint32,
                    pend_tidx: Uint32, pend_pull: Uint32, pend_pool: Uint32,
                    pend_ring: Uint32, pend_src_rank: Uint32, pend_src_base: Uint64,
                    pend_src_token: Uint32, pend_topk_slot: Uint32,
                    pend_full_inc: Uint32, pend_empty: Uint32,
                    eco_fixed_tokens: Uint32, lane: Uint32):
        """Drain one in-flight whole-chunk pull: remote SF/weight fetch + L1
        stores, barrier wait, S2G store, metadata publish and the M-block
        progress release. A deferred pull (ring slot still occupied) does its
        empty-wait and issues serially here instead."""
        if pend != Uint32(0):
            remote_sf_base = pend_src_base + Uint64(self.input_sf_words * 4) + \
                Uint64(pend_src_token) * Uint64(self.hidden // 128 * 4)
            token_topk = pend_src_token * Uint32(self.topk) + pend_topk_slot
            remote_weight = cute.make_ptr(
                Uint32, pend_src_base + Uint64(self.input_weight_words * 4) +
                Uint64(token_topk) * Uint64(4), cute.AddressSpace.gmem)
            weight = remote_weight[0]
            sf_row = pend_ring * ((self.block_m + 127) // 128 * 128) + \
                ((pend_pull % self.block_m) & ~127) + \
                ((pend_pull % self.block_m) & 31) * 4 + \
                (((pend_pull % self.block_m) >> 5) & 3)
            if pend_issued == Uint32(0):
                empty_ptr = signals.iterator + \
                    SIGNAL_OFFSETS['l1_empty_count'] // 4 + pend_ring
                while ld_acquire_gpu_u32(empty_ptr) < pend_empty:
                    pass
            if cutlass.const_expr((self.hidden // 128) % 4 == 0):
                sf_vectors = self.hidden // 512
                for q_round in cutlass.range(
                        (sf_vectors + 31) // 32, unroll_full=True):
                    q = q_round * 32 + lane
                    if q < sf_vectors:
                        remote_vec = cute.make_ptr(
                            Uint32, remote_sf_base + Uint64(q * Uint32(16)),
                            cute.AddressSpace.gmem, assumed_align=16)
                        values = ld_global_u32x4(remote_vec)
                        for component in cutlass.range_constexpr(4):
                            sf_col = q * 4 + component
                            if cutlass.const_expr(self.wide_l1_sf):
                                l1_sf_bytes = cute.recast_ptr(
                                    signals.iterator, dtype=Uint8) + \
                                    Uint64(self.l1_sf_words) * Uint64(4)
                                sf_word = Uint64(sf_col) * \
                                    Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                cute.recast_ptr(
                                    l1_sf_bytes + sf_word * Uint64(4),
                                    dtype=Uint32)[0] = values[component]
                            else:
                                signals[self.l1_sf_words +
                                        sf_col * self.sf_ring_tokens + sf_row] = \
                                    values[component]
            else:
                for sf_round in cutlass.range(
                        (self.hidden // 128 + 31) // 32, unroll_full=True):
                    sf_col = sf_round * 32 + lane
                    if sf_col < self.hidden // 128:
                        remote_word = cute.make_ptr(
                            Uint32, remote_sf_base + Uint64(sf_col * Uint32(4)),
                            cute.AddressSpace.gmem, assumed_align=4)
                        if cutlass.const_expr(self.wide_l1_sf):
                            l1_sf_bytes = cute.recast_ptr(
                                signals.iterator, dtype=Uint8) + \
                                Uint64(self.l1_sf_words) * Uint64(4)
                            sf_word = Uint64(sf_col) * \
                                Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                            cute.recast_ptr(
                                l1_sf_bytes + sf_word * Uint64(4),
                                dtype=Uint32)[0] = remote_word[0]
                        else:
                            signals[self.l1_sf_words +
                                    sf_col * self.sf_ring_tokens + sf_row] = \
                                remote_word[0]
            cute.arch.sync_warp()
            if elect_one_sync():
                warp_pull_p = warp_pull + pend_par * Uint32(self.pull_bytes // 4)
                warp_bar_p = warp_bar + pend_par
                if pend_issued == Uint32(0):
                    if cutlass.const_expr(self.eco and self.eco_trace):
                        if pend_tidx >= eco_fixed_tokens and pend_tidx < \
                                eco_fixed_tokens + \
                                Uint32(self.dispatch_sms * self.dispatch_warps):
                            self._eco_trace_ring(signals, 13, True)
                    bulk_g2s_1d(
                        warp_pull_p,
                        cute.make_ptr(
                            Uint32, pend_src_base +
                            Uint64(self.input_x_words * 4) +
                            Uint64(pend_src_token) * Uint64(self.hidden),
                            cute.AddressSpace.gmem, assumed_align=16),
                        Uint32(self.pull_bytes), warp_bar_p)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        warp_bar_p, self.pull_bytes)
                if pend_par == Uint32(0):
                    cute.arch.mbarrier_wait(warp_bar_p, pull_phase)
                    pull_phase ^= Uint32(1)
                else:
                    cute.arch.mbarrier_wait(warp_bar_p, pull_phase2)
                    pull_phase2 ^= Uint32(1)
                dst_x = cute.recast_ptr(
                    cute.recast_ptr(signals.iterator, dtype=Uint8) +
                    Uint64(self.l1_x_words) * Uint64(4) +
                    Uint64(pend_pool % self.ring_tokens) * Uint64(self.hidden),
                    dtype=Uint32)
                bulk_s2g_1d(dst_x, warp_pull_p, Uint32(self.pull_bytes))
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                signals[self.l1_weight_words +
                        pend_pool % self.ring_tokens] = weight
                signals[self.metadata_words + pend_pool * 3] = pend_src_rank
                signals[self.metadata_words + pend_pool * 3 + 1] = pend_src_token
                signals[self.metadata_words + pend_pool * 3 + 2] = pend_topk_slot
                red_release_gpu_add_u32(
                    signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 +
                    pend_ring, Uint32(pend_full_inc))
                if cutlass.const_expr(self.eco and self.eco_trace):
                    if pend_tidx >= eco_fixed_tokens and pend_tidx < \
                            eco_fixed_tokens + \
                            Uint32(self.dispatch_sms * self.dispatch_warps):
                        self._eco_trace_ring(signals, 14, True)
            cute.arch.sync_warp()
        return pull_phase, pull_phase2

    @cute.jit
    def _pull_pipeline(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                       source_idx: cute.Tensor, tensor_maps: cute.Tensor,
                       pull: cute.Pointer, pull_bar: cute.Pointer,
                       rank_idx: Uint32, lane: Uint32, gen: Uint32,
                       physical_warp: Int32, token_idx: Uint32,
                       current_expert: Int32, expert_start: Uint32,
                       expert_end: Uint32, pool_block_offset: Uint32,
                       stored_rank_count: cute.Tensor, expert_counts: cute.Tensor,
                       expert_starts: cute.Tensor, eco_canons: cute.Tensor,
                       eco_fixed_tokens: Uint32,
                       eco_num_fixed: Uint32, eco_epoch: Uint32,
                       eco_plan_loaded: Uint32) -> None:
        """Pipelined pull: each iteration resolves the next token and issues
        its remote load into the idle staging buffer before draining the
        in-flight one, so the remote flight hides resolution + SF work."""
        rank_waves = (self.num_ranks + 31) // 32
        cached_rank_expert = Int32(-1)
        eco_peel_start = Uint32(0)
        eco_canon_rank = rank_idx
        eco_canon_slot = Uint32(0)
        slot_total = Uint32(0)
        pull_phase = Uint32(0)
        pull_phase2 = Uint32(0)
        parity = Uint32(0)
        pend = Uint32(0)
        pend_issued = Uint32(0)
        pend_par = Uint32(0)
        pend_tidx = Uint32(0)
        pend_pull = Uint32(0)
        pend_pool = Uint32(0)
        pend_ring = Uint32(0)
        pend_src_rank = Uint32(0)
        pend_src_base = Uint64(0)
        pend_src_token = Uint32(0)
        pend_topk_slot = Uint32(0)
        pend_full_inc = Uint32(0)
        pend_empty = Uint32(0)
        # Peel resume point within this expert (zeros resume from scratch).
        pc_off = Uint32(0)
        pc_res = Uint32(0)
        pc_abs = Uint32(0)
        pull_words = self.pull_bytes // 4
        warp_pull = pull + Uint32(physical_warp * 2) * Uint32(pull_words)
        warp_bar = pull_bar + Uint32(physical_warp * 2)
        rep_entry_ns = Uint64(0)
        rep_tok_ns = Uint64(0)
        entry_t0 = Uint64(0)
        tok_t0 = Uint64(0)
        # Replica slots' index-row base on the canonical owner rank, assigned
        # once per slot boundary in the replica phase.
        eco_idx_ptr = source_idx.iterator
        # interleave: fixed-token resume state when phase 0 left early for the replica tokens
        il_early = Uint32(0)
        il_tok = Uint32(0)
        il_exp = Int32(0)
        il_es = Uint32(0)
        il_ee = Uint32(0)
        il_pbo = Uint32(0)
        token_stride = Uint32(self.dispatch_sms * self.dispatch_warps)
        sp_fixed_only = Uint32(0)
        if cutlass.const_expr(self.eco_split_pull > 0):
            sp_wf = self.dispatch_warps - self.eco_split_pull
            sp_pw = Uint32(physical_warp)
            sp_blk = (token_idx - sp_pw) // Uint32(self.dispatch_warps)
            if sp_pw < Uint32(sp_wf):
                token_idx = sp_blk * Uint32(sp_wf) + sp_pw
                token_stride = Uint32(self.dispatch_sms * sp_wf)
                sp_fixed_only = Uint32(1)
            else:
                token_idx = eco_fixed_tokens + sp_blk * Uint32(self.eco_split_pull) + (sp_pw - Uint32(sp_wf))
                token_stride = Uint32(self.dispatch_sms * self.eco_split_pull)
            current_expert = Int32(-1)
            expert_start = Uint32(0)
            expert_end = Uint32(0)
            pool_block_offset = Uint32(0)
            current_expert, expert_start, expert_end, pool_block_offset = \
                advance_expert(
                    expert_counts, token_idx, current_expert,
                    expert_start, expert_end, pool_block_offset, lane,
                    self.local_experts, self.block_m)
        for eco_phase in cutlass.range_constexpr((3 if self.eco_interleave else 2) if self.eco else 1):
            phase_bound = Int32(self.local_experts)
            if cutlass.const_expr(self.eco and eco_phase != 1):
                phase_bound = Int32(eco_num_fixed)
            if cutlass.const_expr(self.eco and eco_phase == 2):
                # resume the fixed tokens left when the plan arrived (else nothing is left)
                current_expert = Int32(self.local_experts)
                if il_early != Uint32(0):
                    token_idx = il_tok
                    current_expert = il_exp
                    expert_start = il_es
                    expert_end = il_ee
                    pool_block_offset = il_pbo
                cached_rank_expert = Int32(-1)
            if cutlass.const_expr(self.eco and eco_phase == 1):
                if cutlass.const_expr(self.eco_interleave):
                    if il_early != Uint32(0):   # first replica token of this warp's token stride
                        token_idx = token_idx + (eco_fixed_tokens - token_idx + token_stride - Uint32(1)) // \
                            token_stride * token_stride
                # Fixed-token range is done; replica slots start here and
                # need the plan role's quotas loaded once before use.
                if eco_plan_loaded == Uint32(0) and sp_fixed_only == Uint32(0) and \
                        eco_num_fixed < Uint32(self.local_experts):
                    self._eco_load_plan(signals, rank_idx, eco_epoch,
                                        eco_num_fixed, expert_counts, lane,
                                        expert_starts)
                    eco_plan_loaded = Uint32(1)
                current_expert = Int32(-1)
                expert_start = Uint32(0)
                expert_end = Uint32(0)
                pool_block_offset = Uint32(0)
                cached_rank_expert = Int32(-1)
                current_expert, expert_start, expert_end, pool_block_offset = \
                    advance_expert(
                        expert_counts, token_idx, current_expert,
                        expert_start, expert_end, pool_block_offset, lane,
                        self.local_experts, self.block_m)
                if sp_fixed_only != Uint32(0):   # split pull: the replica warps own every replica token
                    current_expert = Int32(self.local_experts)
            while current_expert < phase_bound:
                if cached_rank_expert != current_expert:
                    cached_rank_expert = current_expert
                    if cutlass.const_expr(self.eco and self.eco_trace):
                        entry_t0 = Uint64(0)
                        if Uint32(current_expert) >= eco_num_fixed:
                            entry_t0 = read_globaltimer()
                    pc_off = pc_res = pc_abs = Uint32(0)
                    if cutlass.const_expr(self.eco):
                        # The plan kernel materializes replica slots' per-source
                        # counts into native-format u64 rows (quiet ECO region
                        # under mode 3), so every slot reads counts identically.
                        # quota_start comes from registers cached by _eco_load_plan.
                        # Replica slots carry no source-index rows of their own:
                        # the dispatch role wrote only the logical expert's lowest
                        # copy's (canonical) rows, so this slot's index reads go to
                        # that copy's rows on its rank.
                        eco_peel_start = Uint32(0)
                        eco_slot_expert = Uint32(current_expert)
                        eco_canon_rank = rank_idx
                        eco_canon_slot = Uint32(current_expert)
                        if cutlass.const_expr(eco_phase == 1):
                            eco_peel_start = cached_num_tokens(
                                expert_starts, current_expert, lane,
                                self.local_experts)
                            # Canonical copy resolved once per call into
                            # eco_canons (logical_expert<<16 | global_slot).
                            _pk = cached_num_tokens(
                                eco_canons, current_expert, lane,
                                self.local_experts)
                            _le = _pk >> Uint32(16)
                            _gs = _pk & Uint32(0xffff)
                            eco_canon_rank = _gs // Uint32(self.local_experts)
                            eco_canon_slot = _gs % Uint32(self.local_experts)
                            # Per-slot index-row base on the canonical owner
                            # (peer-mapped VA resolves the local case too).
                            eco_idx_ptr = cute.make_ptr(
                                Uint32,
                                mapped_peer_base(peer_ptrs, eco_canon_rank,
                                                 self.num_ranks) +
                                Uint64(SIGNAL_OFFSETS['size']) +
                                Uint64(eco_canon_slot) *
                                Uint64(self.num_ranks * self.capacity * 4),
                                cute.AddressSpace.gmem)
                            if cutlass.const_expr(self.eco_recv_counts == 0):
                                eco_slot_expert = _le
                    for wave in cutlass.range_constexpr(rank_waves):
                        source_rank = Uint32(wave * 32) + lane
                        stored_rank_count[wave] = Uint32(0)
                        if source_rank < Uint32(self.num_ranks):
                            count_ptr = signals.iterator + \
                                SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                                gen * Uint32(self.num_ranks *
                                             self.local_experts * 2) + \
                                (source_rank * Uint32(self.local_experts) +
                                 Uint32(current_expert)) * Uint32(2)
                            if cutlass.const_expr(self.eco and self.eco_recv_counts == 0):
                                if cutlass.const_expr(eco_phase == 1):
                                    stored_rank_count[wave] = self._eco_table_count(
                                        signals, gen, source_rank, eco_slot_expert,
                                        (eco_epoch // Uint32(2)) % Uint32(3) + Uint32(1))
                                else:
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(count_ptr))
                            elif cutlass.const_expr(self.eco and self.eco_recv_counts == 2):
                                if cutlass.const_expr(eco_phase == 1):
                                    stored_rank_count[wave] = Uint32(
                                        ld_volatile_u64(count_ptr))
                                else:
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(count_ptr))
                            elif cutlass.const_expr(self.eco and self.eco_recv_counts == 3):
                                if cutlass.const_expr(eco_phase == 1):
                                    eco_ptr = signals.iterator + \
                                        self.eco_recv_count_base // 4 + \
                                        gen * Uint32(self.num_ranks *
                                                     self.local_experts * 2) + \
                                        (source_rank * Uint32(self.local_experts) +
                                         Uint32(current_expert)) * Uint32(2)
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(eco_ptr))
                                else:
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(count_ptr))
                            elif cutlass.const_expr(self.eco and self.eco_recv_counts == 4):
                                # Timing probe: replica entries skip the count-row
                                # load entirely (constant counts, wrong output).
                                if cutlass.const_expr(eco_phase == 1):
                                    stored_rank_count[wave] = Uint32(32)
                                else:
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(count_ptr))
                            else:
                                stored_rank_count[wave] = Uint32(
                                    ld_global_u64(count_ptr))
                    if cutlass.const_expr(self.eco_spec):
                        lane_total = Uint32(0)
                        for wave in cutlass.range_constexpr(rank_waves):
                            lane_total += stored_rank_count[wave]
                        slot_total = cute.arch.warp_redux_sync(lane_total, 'add')
                    # Warm L2 for this expert's source-index rows while the
                    # in-flight pull drains; the first 64 entries cover the used
                    # prefix of every source's span. Replica slots read the
                    # canonical copy's rows — warm those only when it is local.
                    pf_slot = Uint32(current_expert)
                    if cutlass.const_expr(self.eco and eco_phase == 1):
                        pf_slot = Uint32(self.local_experts)
                        if eco_canon_rank == rank_idx:
                            pf_slot = eco_canon_slot
                    for wave in cutlass.range_constexpr(rank_waves):
                        pf_src = Uint32(wave * 32) + lane
                        if pf_src < Uint32(self.num_ranks) and \
                                pf_slot < Uint32(self.local_experts):
                            idx_row = (pf_slot *
                                       Uint32(self.num_ranks) + pf_src) * \
                                Uint32(self.capacity)
                            prefetch_global_l2(source_idx.iterator + idx_row)
                            prefetch_global_l2(source_idx.iterator +
                                               idx_row + Uint32(32))
                received = expert_end - expert_start
                pull_token = token_idx - expert_start
                pool_block = pool_block_offset + pull_token // self.block_m
                pool_token = pool_block * Uint32(self.block_m) + \
                    pull_token % Uint32(self.block_m)
                ring_block = pool_block % self.ring_blocks
                generation = pool_block // self.ring_blocks

                if cutlass.const_expr(self.eco and self.eco_trace):
                    tok_t0 = Uint64(0)
                    if Uint32(current_expert) >= eco_num_fixed:
                        tok_t0 = read_globaltimer()
                # Resume the peel where the previous token of this expert selected;
                # (0,0,0) is the identity (fresh walk from position 0).
                peel_abs = Uint32(pull_token) + eco_peel_start
                peel_offset = pc_off
                peel_slot = pc_res + (peel_abs - pc_abs)
                selected = False
                src_rank = Uint32(0)
                src_slot = Uint32(0)
                if cutlass.const_expr(self.eco_spec):
                    # Stale speculative spans can overrun this call's order: fold the
                    # overflow back into it (a rank-0 dummy hotspots every rank on
                    # the same remote address). slot_total == 0 has no order:
                    # rank-0 dummy.
                    if slot_total == Uint32(0):
                        selected = True
                    elif peel_slot >= slot_total:
                        peel_slot = peel_slot % slot_total
                        peel_offset = Uint32(0)
                if cutlass.const_expr(self.peel_bs and not self.eco_spec):
                    # level-major order: position p sits at level L (= source slot) with
                    # N(L) = sum_s min(count_s, L) <= p < N(L + 1), on the (p - N(L))-th source with count > L
                    lane_max = Uint32(0)
                    for wave in cutlass.range_constexpr(rank_waves):
                        lane_max = max(lane_max, stored_rank_count[wave])
                    bs_hi = cute.arch.warp_redux_sync(lane_max, 'max')
                    bs_lo = Uint32(0)
                    bs_nlo = Uint32(0)
                    while bs_lo < bs_hi:
                        bs_mid = (bs_lo + bs_hi + Uint32(1)) // Uint32(2)
                        lane_n = Uint32(0)
                        for wave in cutlass.range_constexpr(rank_waves):
                            lane_n += min(stored_rank_count[wave], bs_mid)
                        bs_n = cute.arch.warp_redux_sync(lane_n, 'add')
                        if bs_n <= peel_abs:
                            bs_lo = bs_mid
                            bs_nlo = bs_n
                        else:
                            bs_hi = bs_mid - Uint32(1)
                    bs_pos = peel_abs - bs_nlo
                    bs_seen = Uint32(0)
                    for wave in cutlass.range_constexpr(rank_waves):
                        bs_mask = Uint32(cute.arch.vote_ballot_sync(
                            stored_rank_count[wave] > bs_lo))
                        bs_act = Uint32(cute.arch.popc(bs_mask))
                        if bs_pos >= bs_seen and bs_pos < bs_seen + bs_act:
                            src_rank = Uint32(wave * 32) + warp_nth_set_bit(
                                bs_mask, bs_pos - bs_seen + Uint32(1))
                        bs_seen += bs_act
                    src_slot = bs_lo
                    selected = True
                while not selected:
                    lane_active = Uint32(0)
                    lane_minimum = Uint32(0xffffffff)
                    for wave in cutlass.range_constexpr(rank_waves):
                        count = stored_rank_count[wave] - min(stored_rank_count[wave],
                                                              peel_offset)
                        lane_active += Uint32(1) if count > Uint32(0) else Uint32(0)
                        if count > Uint32(0):
                            lane_minimum = min(lane_minimum, count)
                    active = cute.arch.warp_redux_sync(lane_active, 'add')
                    minimum = cute.arch.warp_redux_sync(lane_minimum, 'min')
                    round_tokens = active * minimum
                    if peel_slot < round_tokens:
                        position = peel_slot % active
                        seen = Uint32(0)
                        for wave in cutlass.range_constexpr(rank_waves):
                            mask = Uint32(cute.arch.vote_ballot_sync(
                                stored_rank_count[wave] > peel_offset))
                            wave_active = Uint32(cute.arch.popc(mask))
                            if position >= seen and position < seen + wave_active:
                                src_rank = Uint32(wave * 32) + warp_nth_set_bit(
                                    mask, position - seen + Uint32(1))
                            seen += wave_active
                        src_slot = peel_offset + peel_slot // active
                        selected = True
                    else:
                        peel_slot -= round_tokens
                        peel_offset += minimum
                pc_off = peel_offset
                pc_res = peel_slot
                pc_abs = peel_abs
                if cutlass.const_expr(self.eco and eco_phase == 1 and self.eco_idx_probe):
                    token_topk = (src_slot * Uint32(131) + Uint32(current_expert) * Uint32(7)) % \
                        (Uint32(self.num_tokens) * Uint32(self.topk))
                elif cutlass.const_expr(self.eco and eco_phase == 1):
                    # Replica slot: one indexed load off the per-slot canonical
                    # base (peer-mapped VA covers both remote and local canon).
                    token_topk = Uint32(eco_idx_ptr[
                        Uint32(src_rank) * Uint32(self.capacity) + src_slot])
                else:
                    token_topk = source_idx[(Uint32(current_expert) *
                                             Uint32(self.num_ranks) +
                                             src_rank) * Uint32(self.capacity) +
                                            src_slot]
                if cutlass.const_expr(self.eco_spec):
                    token_topk = min(token_topk, Uint32(self.num_tokens) *
                                     Uint32(self.topk) - Uint32(1))
                if cutlass.const_expr(self.eco_route_verify):
                    if lane == Uint32(0):
                        st_global_u32(cute.make_ptr(
                            Uint32, mapped_peer_base(peer_ptrs, src_rank, self.num_ranks) +
                            Uint64(self.eco_base + ECO_SIGNAL_OFFSETS['eco_route_dbg']) + Uint64(token_topk) * Uint64(4),
                            cute.AddressSpace.gmem, assumed_align=4), rank_idx)
                src_token = token_topk // Uint32(self.topk)
                topk_slot = token_topk % Uint32(self.topk)
                is_last_token = token_idx == expert_end - Uint32(1)
                full_increment = Uint32(self.block_m) - (pull_token % self.block_m) \
                    if is_last_token else Uint32(1)
                src_base = mapped_peer_base(peer_ptrs, src_rank, self.num_ranks)
                src_x = cute.make_ptr(
                    Uint32, src_base + Uint64(self.input_x_words * 4) +
                    Uint64(src_token) * Uint64(self.hidden),
                    cute.AddressSpace.gmem, assumed_align=16)
                # Issue this token's remote load into the idle buffer unless its
                # ring slot is still held by an older generation (then the drain
                # does the empty-wait and pulls serially).
                empty_target = Uint32(0)
                if generation > Uint32(0):
                    empty_target = generation * Uint32(2 * self.intermediate // 128)
                this_issued = Uint32(0)
                if empty_target == Uint32(0):
                    if elect_one_sync():
                        if cutlass.const_expr(self.eco and self.eco_trace):
                            if token_idx >= eco_fixed_tokens and token_idx < \
                                    eco_fixed_tokens + \
                                    Uint32(self.dispatch_sms * self.dispatch_warps):
                                self._eco_trace_ring(signals, 13, True)
                        bulk_g2s_1d(warp_pull + parity * Uint32(pull_words), src_x,
                                     Uint32(self.pull_bytes), warp_bar + parity)
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            warp_bar + parity, self.pull_bytes)
                    this_issued = Uint32(1)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.eco and self.eco_trace):
                    if tok_t0 != Uint64(0):
                        rep_tok_ns += read_globaltimer() - tok_t0
                # The in-flight token has now been remote for the whole resolve;
                # finish its SF/weight fetch, barrier wait, S2G and publish.
                pull_phase, pull_phase2 = self._pull_drain(
                    signals, warp_pull, warp_bar, pull_phase, pull_phase2,
                    pend, pend_issued, pend_par, pend_tidx, pend_pull, pend_pool,
                    pend_ring, pend_src_rank, pend_src_base, pend_src_token,
                    pend_topk_slot, pend_full_inc, pend_empty, eco_fixed_tokens,
                    lane)
                pend = Uint32(1)
                pend_issued = this_issued
                pend_par = parity
                pend_tidx = token_idx
                pend_pull = pull_token
                pend_pool = pool_token
                pend_ring = ring_block
                pend_src_rank = src_rank
                pend_src_base = src_base
                pend_src_token = src_token
                pend_topk_slot = topk_slot
                pend_full_inc = full_increment
                pend_empty = empty_target
                parity ^= Uint32(1)
                if cutlass.const_expr(self.eco and self.eco_trace):
                    if entry_t0 != Uint64(0):
                        rep_entry_ns += read_globaltimer() - entry_t0
                        entry_t0 = Uint64(0)
                token_idx += token_stride
                current_expert, expert_start, expert_end, pool_block_offset = \
                    advance_expert(
                        expert_counts, token_idx, current_expert, expert_start,
                        expert_end, pool_block_offset, lane, self.local_experts,
                        self.block_m)
                if cutlass.const_expr(self.eco_interleave and eco_phase == 0):
                    if current_expert < phase_bound and eco_num_fixed < Uint32(self.local_experts):
                        if ld_volatile_u32(signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_plan_ready']) // 4) == \
                                eco_epoch + Uint32(1):
                            il_early = Uint32(1)
                            il_tok = token_idx
                            il_exp = current_expert
                            il_es = expert_start
                            il_ee = expert_end
                            il_pbo = pool_block_offset
                            current_expert = Int32(self.local_experts)   # leave phase 0 now
        pull_phase, pull_phase2 = self._pull_drain(
            signals, warp_pull, warp_bar, pull_phase, pull_phase2,
            pend, pend_issued, pend_par, pend_tidx, pend_pull, pend_pool,
            pend_ring, pend_src_rank, pend_src_base, pend_src_token,
            pend_topk_slot, pend_full_inc, pend_empty, eco_fixed_tokens, lane)
        if cutlass.const_expr(self.eco and self.eco_trace):
            if entry_t0 != Uint64(0):
                rep_entry_ns += read_globaltimer() - entry_t0
            if elect_one_sync():
                base = signals.iterator + \
                    (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4
                entry = Uint32(ld_volatile_u64(base + 33 * 2)) % Uint32(64)
                ring = signals.iterator + \
                    (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace_ring']) // 4 + \
                    entry * Uint32(ECO_TRACE_RING_FIELDS) + Uint32(4)
                red_gpu_add_u32(ring, Uint32(rep_entry_ns))
                red_gpu_add_u32(ring + Uint32(8), Uint32(rep_tok_ns))

    @cute.jit
    def _role3_dispatch(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                        rank_idx: Uint32, tid: Uint32, lane: Uint32,
                        block: Uint32, recv_ptr: cute.Pointer, stats: cute.Tensor,
                        ready_epoch: Uint64, eco_fixed_slots: Uint32 = None) -> None:
        # Publish completed peer FC2 stores even with no routed tokens.
        # Epochs distinguish late old publications without resetting ready.
        if block == 0 and tid < self.num_ranks:
            dst = cute.make_ptr(
                Uint64, mapped_peer_base(peer_ptrs, Uint32(tid), self.num_ranks) +
                Uint64(SIGNAL_OFFSETS['combine_ready_grid_idx']) +
                Uint64(rank_idx * 8), cute.AddressSpace.gmem,
                assumed_align=8)
            st_release_sys_u64(dst, ready_epoch)
        # Dual-buffered receive sums: this replay consumes the generation
        # selected by the persistent parity word; the bump below flips it
        # for the next replay, and role 1 of the replay that retires this
        # generation clears it at entry.
        gen = Uint32(ready_epoch - Uint64(1)) & Uint32(1)
        recv_ptr = recv_ptr + gen * Uint32(self.local_experts * 2)
        # Role1 acquired every expert sum, including zero-count experts,
        # before publishing this summary. Normal graph completion dependencies
        # also retire every role2 ring access before this footer executes.
        pool_blocks = Uint32(signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4])
        if cutlass.const_expr(self.eco):
            # ECO: replica slots hold plan quotas, not native counts; total the padded slot blocks here
            eco_lane_blocks = Uint32(0)
            for eco_wave in cutlass.range_constexpr((self.local_experts + 31) // 32):
                eco_slot = Uint32(eco_wave * 32) + lane
                if eco_slot < Uint32(self.local_experts):
                    eco_lane_blocks += (self._eco_slot_total(
                        signals, rank_idx, eco_slot, eco_fixed_slots, recv_ptr,
                        Uint32(ready_epoch - Uint64(1))) + Uint32(self.block_m - 1)) // Uint32(self.block_m)
            pool_blocks = cute.arch.warp_redux_sync(eco_lane_blocks, 'add')
        # All CTAs use the immutable role1 epoch, so block0 may flip parity
        # without a grid-wide cache-completion barrier. The next dispatch
        # clears this bank only after the entire local finalize grid ends.
        # That clear precedes count publication; all-rank sum acquires in
        # the intervening call order it before any peer can publish into
        # the same bank again. Zero-count experts must join too.
        if block == Uint32(0) and tid == Uint32(0):
            red_gpu_add_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4,
                Uint32(1))
        warpgroup_reg_dealloc(48 if self.local_experts <= 64 else 96)

        if block == 0:
            send_off = SIGNAL_OFFSETS['expert_send_count'] // 4
            for expert_iter in cutlass.range(
                    Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                    128, unroll_full=True):
                expert_idx = Uint32(expert_iter)
                st_global_u64(signals.iterator + send_off + expert_idx * Uint32(2),
                              Uint64(0))
            if tid // Uint32(32) == Uint32(0):
                if elect_one_sync():
                    for locality in cutlass.range(2, unroll_full=True):
                        signals[SIGNAL_OFFSETS['l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['l2_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l1_task_count'] // 4 + locality] = Uint32(0)
                        signals[SIGNAL_OFFSETS['shared_l2_task_count'] // 4 + locality] = Uint32(0)
            cute.arch.sync_warp()
            shared_block = Uint32(tid)
            while shared_block < Uint32((self.capacity + 7) // 8):
                signals[SIGNAL_OFFSETS['shared_l2_full_count'] // 4 +
                        shared_block] = Uint32(0)
                shared_block += Uint32(128)
            cute.arch.sync_warp()
        if cutlass.const_expr(self.has_stats):
            local_expert = block * Uint32(128) + tid
            while local_expert < Uint32(self.local_experts):
                received = Uint32(ld_global_u64(recv_ptr + local_expert * Uint32(2)))
                if cutlass.const_expr(self.eco):
                    received = self._eco_slot_total(signals, rank_idx, local_expert, eco_fixed_slots,
                                                    recv_ptr, Uint32(ready_epoch - Uint64(1)))
                red_gpu_add_s32(stats.iterator + local_expert, Int32(received))
                local_expert += Uint32(self.finalize_sms * 128)

        # The padded expert ranges concatenate to [0,P). Modulo R, their
        # used slots are exactly [0,min(P,R)); clear each of those slots once.
        ring_block = block * Uint32(128) + tid
        while ring_block < min(pool_blocks, Uint32(self.ring_blocks)):
            signals[SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block] = Uint32(0)
            signals[SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block] = Uint32(0)
            st_global_u64(
                signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                ring_block * Uint32(2), Uint64(0))
            signals[SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block] = Uint32(0)
            ring_block += Uint32(self.finalize_sms * 128)

        # The former tail NVLink barrier (tag 2) is gone: the receive
        # signals are dual-buffered by recv_parity, so peers' next publish
        # targets the alternate generation and ordering is carried by the
        # all-expert sum acquires in the intervening call. Selected-peer
        # ready acquires protect each source's single-buffer inputs/indices.

    @cute.jit
    def _role3_combine(self, signals: cute.Tensor, rank_idx: Uint32,
                       num_tokens: Uint32, y_out: cute.Tensor,
                       block, physical_warp, lane,
                       combine_smem: cute.Pointer,
                       combine_bars: cute.Pointer, ready_epoch: Uint64) -> None:
        combine_chunks = self.combine_chunks
        chunk_bytes = self.combine_chunk_bytes
        u4_per_lane = self.combine_uint4_per_lane
        warpgroup_reg_alloc(self.epi_registers)
        epi_warp = Uint32(physical_warp - 4)
        ready = signals.iterator + SIGNAL_OFFSETS['combine_ready_grid_idx'] // 4
        if cutlass.const_expr(not self.finalize_output or
                              (self.eco and not self.eco_combine_probe and not self.eco_combine_exact)):
            # No token reduction exists to carry selected-peer acquires. ECO: a token's replica results come from
            # the copy's rank, not routed_expert // local_experts, so wait for every peer as before.
            peer = epi_warp
            while peer < Uint32(self.num_ranks):
                wait_sys_u64_eq(ready + peer * 2, ready_epoch, rank_idx, peer)
                peer += Uint32(COMBINE_WARPS)
            sync_aligned(COMBINE_WARPS * 32, 2)
        if cutlass.const_expr(self.finalize_output):
            # One warp owns a token chunk. Every routed slot of the chunk is
            # fetched into its own plane under a single barrier so the
            # reduction exposes one TMA latency instead of one per slot. The
            # output plane aliases slot 0 and is retired before the next
            # chunk overwrites it.
            combine = cute.recast_ptr(
                signals.iterator, dtype=Uint8) + \
                Uint64(self.combine_words) * Uint64(4)
            output = cute.recast_ptr(y_out.iterator, dtype=Uint8)
            routes = cute.recast_ptr(
                signals.iterator + self.topk_offset_words, dtype=Uint64)
            combine_slots = self.topk + (1 if self.shared_experts > 0 else 0)
            # Keep the historical reduction order: shared slot, then routes.
            slot_of = ([self.topk] if self.shared_experts > 0 else []) + \
                list(range(self.topk))
            store_smem = combine_smem + Uint32(epi_warp * chunk_bytes)
            store_words = cute.recast_ptr(store_smem, dtype=Uint32)
            bar = combine_bars + epi_warp
            phase = Uint32(0)
            token_chunk = epi_warp * Uint32(self.finalize_sms) + block
            if cutlass.const_expr(self.shared_experts > 0):
                if token_chunk < num_tokens * Uint32(combine_chunks):
                    _initialize_private_combine_bar(bar)
            while token_chunk < num_tokens * Uint32(combine_chunks):
                token = token_chunk // Uint32(combine_chunks)
                chunk = token_chunk % Uint32(combine_chunks)
                chunk_offset = chunk * Uint32(chunk_bytes)
                lane_valid = False
                routed_peer = Uint32(0)
                if lane < self.topk:
                    routed_expert = routes[token * Uint32(self.topk) + lane]
                    lane_valid = routed_expert < \
                        Uint64(self.num_ranks * self.local_experts)
                    if lane_valid:
                        routed_peer = Uint32(routed_expert) // Uint32(self.local_experts)
                        if cutlass.const_expr(self.eco_combine_exact):
                            routed_peer = Uint32(signals[(self.eco_base + ECO_SIGNAL_OFFSETS['eco_route_peer']) // 4 +
                                                         token * Uint32(self.topk) + lane])
                # All lanes vote; invalid routes never form a peer pointer.
                # Duplicate owners may acquire the same publication twice.
                peer_ready = not lane_valid
                if cutlass.const_expr(self.eco and not self.eco_combine_probe and not self.eco_combine_exact):
                    peer_ready = True   # every peer was acquired above
                wait_start = read_clock64()
                while cute.arch.vote_ballot_sync(peer_ready) != Uint32(0xffffffff):
                    if lane_valid and not peer_ready:
                        peer_ready = ld_acquire_sys_u64(
                            ready + routed_peer * Uint32(2)) == ready_epoch
                    if Uint64(read_clock64() - wait_start) >= Uint64(60 * 2000000000):
                        if not peer_ready:
                            cute.printf('GC token readiness timeout rank={} peer={}',
                                        rank_idx, routed_peer)
                            device_trap()
                # Transfer selected-lane acquires to the elected TMA loader.
                cute.arch.sync_warp()
                if cutlass.const_expr(self.shared_experts > 0):
                    if lane == self.topk:
                        lane_valid = True
                mask = Uint32(cute.arch.vote_ballot_sync(lane_valid))
                # Retire the previous chunk's output before any fetch plane
                # reuses the aliased slot-0 region, also when this chunk has
                # no valid route and only writes zeros. The wait is per-thread
                # and only the elected lane issued the store, so converge the
                # warp before any lane touches the plane.
                tma_store_wait_0()
                cute.arch.sync_warp()
                if mask != Uint32(0):
                    if elect_one_sync():
                        expect = Uint32(0)
                        for g in cutlass.range_constexpr(combine_slots):
                            if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                                expect += Uint32(chunk_bytes)
                        cute.arch.mbarrier_arrive_and_expect_tx(bar, expect)
                        for g in cutlass.range_constexpr(combine_slots):
                            if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                                src = combine + \
                                    (Uint64(slot_of[g]) * Uint64(self.capacity) +
                                     Uint64(token)) * Uint64(self.hidden * 2) + \
                                    Uint64(chunk_offset)
                                bulk_g2s_1d(
                                    combine_smem +
                                    (Uint32(slot_of[g]) * Uint32(COMBINE_WARPS) + epi_warp) *
                                    Uint32(chunk_bytes), src,
                                    Uint32(chunk_bytes), bar)
                    cute.arch.sync_warp()
                    cute.arch.mbarrier_wait(bar, phase)
                    phase ^= Uint32(1)
                reduced = cute.make_rmem_tensor((u4_per_lane * 8,), Float32)
                for index in cutlass.range_constexpr(u4_per_lane * 8):
                    reduced[index] = Float32(0)
                for g in cutlass.range_constexpr(combine_slots):
                    if ((mask >> Uint32(slot_of[g])) & Uint32(1)) != Uint32(0):
                        load_words = cute.recast_ptr(
                            combine_smem + (Uint32(slot_of[g]) * Uint32(COMBINE_WARPS) + epi_warp) *
                            Uint32(chunk_bytes), dtype=Uint32)
                        for j in cutlass.range_constexpr(u4_per_lane):
                            words = ld_shared_u32x4(
                                load_words + (j * 32 + lane) * 4)
                            for q in cutlass.range_constexpr(4):
                                packed = words[q]
                                reduced[j * 8 + q * 2] = \
                                    accumulate_bf16_into_f32(
                                        reduced[j * 8 + q * 2],
                                        Uint16(packed & Uint32(0xffff)))
                                reduced[j * 8 + q * 2 + 1] = \
                                    accumulate_bf16_into_f32(
                                        reduced[j * 8 + q * 2 + 1],
                                        Uint16(packed >> Uint32(16)))
                cute.arch.fence_view_async_shared()

                for j in cutlass.range_constexpr(u4_per_lane):
                    packed_words = cute.make_rmem_tensor((4,), Uint32)
                    for q in cutlass.range_constexpr(4):
                        packed_words[q] = pack_bf16_pair(
                            reduced[j * 8 + q * 2],
                            reduced[j * 8 + q * 2 + 1])
                    st_shared_u32x4(
                        store_words + (j * 32 + lane) * 4,
                        packed_words[0], packed_words[1],
                        packed_words[2], packed_words[3])
                cute.arch.sync_warp()
                if elect_one_sync():
                    tma_store_fence()
                    dst = output + Uint64(token) * \
                        Uint64(self.hidden * 2) + Uint64(chunk_offset)
                    bulk_s2g_1d(dst, store_smem, Uint32(chunk_bytes))
                    tma_store_arrive()
                cute.arch.sync_warp()
                token_chunk += Uint32(self.finalize_sms * COMBINE_WARPS)


    @cute.kernel
    def kernel(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
               rank_idx: Uint32, num_tokens: Uint32,
               tensor_maps=None, y_out=None, stats=None,
               activation_clamp=None, activation_alpha=None,
               tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
               tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
               tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
               tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
               tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
               tensor_map_16=None, tensor_map_17=None,
               tensor_map_18=None, tensor_map_19=None,
               tensor_map_20=None, tensor_map_21=None, eco_fixed_slots=None):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        physical_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32

        if cutlass.const_expr(self.role == 1):
            if cutlass.const_expr(self.eco_trace):
                if tid == 0:
                    self._eco_trace(signals, 0, 1)
                    if block == 0:
                        base = signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4
                        prev_start = ld_volatile_u64(base + 24 * 2)
                        self._eco_trace(signals, 24, 0)
                        call = atom_add_u64(base + 21 * 2, Uint64(1))
                        # Slot 30 is the counting-phase timestamp. Keep the ring's call index separate.
                        st_global_u64(base + 33 * 2, call)
                        ring = signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace_ring']) // 4 + \
                            (Uint32(call) % Uint32(64)) * Uint32(ECO_TRACE_RING_FIELDS)
                        for q in cutlass.range_constexpr(ECO_TRACE_RING_FIELDS // 4):
                            st_global_u32x4(ring + 4 * q, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        # field 11: this call's start - the previous call's start (the call period)
                        st_global_u32(ring + 11, Uint32(ld_volatile_u64(base + 24 * 2) - prev_start))
            # Count local routes and reserve distinct source-index spans. ECO overlaps count publication with
            # source-index writes; the grid and rank barriers join both before activation pulls.
            # Route each global expert to its owner rank and local expert.
            # Each local expert has an independent receive count and a padded
            # pool-block prefix; ring slots remain shared by all experts.
            dispatch_smem = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(
                dispatch_smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            expert_token_count = cute.recast_ptr(
                slab + self.shared_layout.expert_counts, dtype=Uint32)
            if cutlass.const_expr(self.eco):
                # ECO: per logical expert's canonical (lowest) copy slot
                copies_pk = cute.recast_ptr(
                    dispatch_smem.allocate(self.route_experts * 4, 16), dtype=Uint32)
            pull = cute.recast_ptr(slab + self.shared_layout.pull, dtype=Uint32)
            pull_bar = cute.recast_ptr(slab + self.shared_layout.dispatch_bars, dtype=Uint64)
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane))
            warpgroup_reg_dealloc(self.dispatch_registers)
            # Dual-buffered receive signals: replay N publishes and reads
            # generation parity & 1; role 3 flips the persistent parity
            # word after consuming each replay, and the replay that retires
            # a generation clears it here, so no cross-rank tail barrier is
            # needed.  Role 3 reads the epoch published below instead of a
            # grid barrier.
            parity = ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)
            retired = (parity + Uint32(1)) & Uint32(1)
            gen = parity & Uint32(1)
            if block == Uint32(0):
                if tid == Uint32(0):
                    # The normal role1 -> role3 graph completion dependency
                    # makes this plain store visible; role3 never polls it
                    # concurrently with this dispatch.
                    signals[SIGNAL_OFFSETS['role3_epoch'] // 4] = parity + Uint32(1)
                retired_sum = signals.iterator + \
                    SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                    retired * Uint32(self.local_experts * 2)
                for retired_iter in cutlass.range(
                        Uint32(tid), Uint32(self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    st_global_u64(
                        retired_sum + Uint32(retired_iter) * Uint32(2), Uint64(0))
                retired_recv = signals.iterator + \
                    SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                    retired * Uint32(self.num_ranks * self.local_experts * 2)
                for retired_iter in cutlass.range(
                        Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    st_global_u64(
                        retired_recv + Uint32(retired_iter) * Uint32(2), Uint64(0))
            recv_epoch = parity
            eco_num_fixed = Uint32(0)
            if cutlass.const_expr(self.eco):
                eco_num_fixed = Uint32(tensor_maps[self.eco_num_fixed_words + rank_idx])
                # Each publishing thread preloads its static slot words before counting/routing. They survive
                # in registers until publication, which then needs only the logical expert's send-count load.
                eco_slot_info = cute.make_rmem_tensor(
                    ((self.num_ranks * self.local_experts + self.dispatch_warps * 32 - 1) //
                     (self.dispatch_warps * 32),), Uint32)
                for wave in cutlass.range_constexpr(cute.size(eco_slot_info)):
                    slot_idx = Uint32(tid) + Uint32(wave * self.dispatch_warps * 32)
                    info = Uint32(0)
                    if block == self.dispatch_sms - 1 and slot_idx < Uint32(self.num_ranks * self.local_experts):
                        info = Uint32(tensor_maps[self.eco_slot_info_words + slot_idx])
                    eco_slot_info[wave] = info
                if block == self.dispatch_sms - 1 and tid == 0:
                    self._eco_trace(signals, 34, 0)
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            send_ptr = signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4
            routes = cute.make_tensor(
                cute.recast_ptr(signals.iterator + self.topk_offset_words, dtype=Uint64),
                cute.make_layout((self.capacity * self.topk,)))
            source_idx = cute.make_tensor(
                cute.recast_ptr(signals.iterator + SIGNAL_OFFSETS['size'] // 4,
                                dtype=Uint32),
                cute.make_layout((self.num_ranks * self.local_experts * self.capacity,)))
            # Reference read_topk_idx: adjacent lanes own adjacent topk
            # entries, so one warp processes floor(32 / topk) tokens per
            # iteration. Lanes past that whole-token group stay inactive.
            tokens_per_warp = 32 // self.topk
            active_route_lanes = tokens_per_warp * self.topk
            route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                tokens_per_warp)
            route_stride = self.dispatch_sms * self.dispatch_warps * tokens_per_warp
            if cutlass.const_expr(self.eco):
                # ECO: each logical expert's lowest copy (the canonical row holder; the
                # globally lowest copy is never passive) in smem. The plan role fans the
                # canonical row out to the other replica rows after the rank barrier, so
                # the routing loop below stores once per token.
                for pk_iter in cutlass.range(
                        Uint32(tid), Uint32(self.route_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    copies_pk[Uint32(pk_iter)] = Uint32(tensor_maps[
                        self.eco_copies_words + Uint32(pk_iter) * Uint32(16)]) & Uint32(0xffff)
            if cutlass.const_expr(self.eco_route_prep):
                # The producer already combined the expert totals with each dispatch CTA's
                # source-index start. Wait once, then initialize this CTA's cursors.
                route_ready = signals.iterator + (self.eco_base +
                                                  ECO_SIGNAL_OFFSETS['eco_counts_ready']) // 4
                if tid == 0:
                    while ld_acquire_gpu_u32(route_ready) != recv_epoch + Uint32(1):
                        pass
                    if cutlass.const_expr(self.eco_trace):
                        self._eco_trace(signals, 41, 0)
                cute.arch.sync_threads()
                route_offsets = signals.iterator + (self.eco_base +
                    ECO_SIGNAL_OFFSETS['eco_counts'] +
                    Uint32(4 * self.num_ranks * self.num_ranks * self.local_experts)) // 4 + \
                    block * Uint32(self.route_experts)
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.route_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    expert_token_count[expert_idx] = ld_volatile_u32(
                        route_offsets + expert_idx)
            else:
                for route_iter in cutlass.range(
                        route_base, num_tokens, route_stride, unroll_full=True):
                    route_base = Uint32(route_iter)
                    route_token = route_base + Uint32(lane // self.topk)
                    token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                    if lane < active_route_lanes and route_token < num_tokens:
                        expert = routes[token_topk]
                        if expert < Uint64(self.route_experts):
                            atom_shared_add_u32(expert_token_count + Uint32(expert),
                                                Uint32(1))
                    cute.arch.sync_warp()
            if cutlass.const_expr(self.eco):
                if lane == Uint32(0):
                    self._eco_trace(signals, 30, 2)
            cute.arch.sync_threads()
            if cutlass.const_expr(not self.eco_route_prep):
                # ECO: counts and source-slot spans are per logical expert
                for expert_iter in cutlass.range(
                        Uint32(tid), Uint32(self.route_experts),
                        self.dispatch_warps * 32, unroll_full=True):
                    expert_idx = Uint32(expert_iter)
                    old_send = atom_add_u64(
                        send_ptr + expert_idx * Uint32(2),
                        (Uint64(1) << 32) | Uint64(expert_token_count[expert_idx]))
                    expert_token_count[expert_idx] = Uint32(old_send)
            cute.arch.sync_threads()
            route_base = Uint32((block * self.dispatch_warps + physical_warp) *
                                tokens_per_warp)
            for route_iter in cutlass.range(
                    route_base, num_tokens, route_stride, unroll_full=True):
                route_base = Uint32(route_iter)
                route_token = route_base + Uint32(lane // self.topk)
                token_topk = route_base * Uint32(self.topk) + Uint32(lane)
                if lane < active_route_lanes and route_token < num_tokens:
                    expert = routes[token_topk]
                    if cutlass.const_expr(self.eco):
                        # ECO: the source index goes to the logical expert's lowest copy's row
                        # only — the canonical row. The plan role replicates it into the
                        # other copies' rows after the rank barrier (verbatim: every copy
                        # later pulls its quota span of the same canonical order).
                        if expert < Uint64(self.route_experts):
                            slot = atom_shared_add_u32(
                                expert_token_count + Uint32(expert), Uint32(1))
                            raw = copies_pk[Uint32(expert)]
                            dst_rank = raw // Uint32(self.local_experts)
                            dst_local = raw % Uint32(self.local_experts)
                            dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                            dst_index = cute.make_ptr(
                                Uint32, dst_base + Uint64(SIGNAL_OFFSETS['size']) +
                                Uint64(((dst_local * Uint32(self.num_ranks) + rank_idx) *
                                        Uint32(self.capacity) + slot) * Uint32(4)),
                                cute.AddressSpace.gmem)
                            dst_index[0] = token_topk
                            if cutlass.const_expr(self.eco_combine_exact > 0):
                                signals[(self.eco_base + ECO_SIGNAL_OFFSETS['eco_route_peer']) // 4 +
                                        token_topk] = slot
                    elif expert < Uint64(self.num_ranks * self.local_experts):
                        dst_rank = Uint32(expert) // Uint32(self.local_experts)
                        dst_local = Uint32(expert) % Uint32(self.local_experts)
                        slot = atom_shared_add_u32(
                            expert_token_count + Uint32(expert), Uint32(1))
                        dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                        dst_index = cute.make_ptr(
                            Uint32, dst_base + Uint64(SIGNAL_OFFSETS['size']) +
                            Uint64(((dst_local * Uint32(self.num_ranks) + rank_idx) *
                                    Uint32(self.capacity) +
                                    slot) *
                                   Uint32(4)), cute.AddressSpace.gmem)
                        dst_index[0] = token_topk
                cute.arch.sync_warp()
            if cutlass.const_expr(self.eco_combine_exact == 2):
                # this CTA's route slots are recorded: the plan CTA may derive their computing ranks
                cute.arch.sync_threads()
                if tid == Uint32(0):
                    st_release_u32(signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_routes_done']) // 4 +
                                   block, recv_epoch + Uint32(1), False)
            if cutlass.const_expr(self.eco_trace):
                if elect_one_sync():
                    self._eco_trace_ring(signals, 11, True)   # field 11: last dispatch warp finished routing
            if cutlass.const_expr(self.eco):
                if lane == Uint32(0):
                    self._eco_trace(signals, 31, 2)
            if cutlass.const_expr(self.eco):
                # Histograms are complete before each CTA writes source indices. The last CTA (typically
                # without source tokens at decode sizes) can publish fixed counts while the other CTAs route.
                # Wait on the histograms' existing arrival tags, not on source-index completion. The existing
                # grid + NVLink barrier below orders BOTH kinds of remote writes before any activation pull.
                if block == self.dispatch_sms - 1:
                    if tid == 0:
                        self._eco_trace(signals, 35, 0)
                    for wave in cutlass.range_constexpr(cute.size(eco_slot_info)):
                        slot_idx = Uint32(tid) + Uint32(wave * self.dispatch_warps * 32)
                        slot_info = eco_slot_info[wave]
                        if (slot_info >> Uint32(31)) != Uint32(0):
                            dst_rank = slot_idx // Uint32(self.local_experts)
                            dst_local = slot_idx % Uint32(self.local_experts)
                            dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                            count_ptr = signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 + \
                                (slot_info & Uint32(0x7fffffff)) * Uint32(2)
                            status = ld_volatile_u64(count_ptr)
                            while Uint32(status >> 32) != Uint32(self.dispatch_sms):
                                status = ld_volatile_u64(count_ptr)
                            remote_count = cute.make_ptr(
                                Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count']) +
                                Uint64(gen * Uint32(self.num_ranks * self.local_experts * 8)) +
                                Uint64((rank_idx * Uint32(self.local_experts) + dst_local) *
                                       Uint32(8)), cute.AddressSpace.gmem,
                                assumed_align=8)
                            st_global_u64(remote_count, status & Uint64(0xffffffff))
                            remote_sum = cute.make_ptr(
                                Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count_sum']) +
                                Uint64(gen * Uint32(self.local_experts * 8)) +
                                Uint64(dst_local * Uint32(8)),
                                cute.AddressSpace.gmem, assumed_align=8)
                            atom_sys_add_u64(remote_sum, status)
                    if tid == 0:
                        self._eco_trace(signals, 40, 0)
            grid_sync(signals, Uint32(block), Uint32(tid), self.dispatch_sms, 0)
            if cutlass.const_expr(not self.eco):
                if block == 0:
                    for expert_iter in cutlass.range(
                            Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                            self.dispatch_warps * 32, unroll_full=True):
                        expert_idx = Uint32(expert_iter)
                        dst_rank = expert_idx // Uint32(self.local_experts)
                        dst_local = expert_idx % Uint32(self.local_experts)
                        dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                        status = ld_global_u64(
                            signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 +
                            expert_idx * Uint32(2))
                        remote_count = cute.make_ptr(
                            Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count']) +
                            Uint64(gen * Uint32(self.num_ranks * self.local_experts * 8)) +
                            Uint64((rank_idx * Uint32(self.local_experts) + dst_local) *
                                   Uint32(8)), cute.AddressSpace.gmem,
                            assumed_align=8)
                        st_global_u64(remote_count, status & Uint64(0xffffffff))
                    # The grid acquire carries all route writes. Transfer
                    # them, plus every per-rank count store,
                    # to every publisher before the system release fence.
                    cute.arch.sync_threads()
                    fence_acq_rel_sys()
                    for expert_iter in cutlass.range(
                            Uint32(tid), Uint32(self.num_ranks * self.local_experts),
                            self.dispatch_warps * 32, unroll_full=True):
                        expert_idx = Uint32(expert_iter)
                        dst_rank = expert_idx // Uint32(self.local_experts)
                        dst_local = expert_idx % Uint32(self.local_experts)
                        dst_base = mapped_peer_base(peer_ptrs, dst_rank, self.num_ranks)
                        status = ld_global_u64(
                            signals.iterator + SIGNAL_OFFSETS['expert_send_count'] // 4 +
                            expert_idx * Uint32(2))
                        remote_sum = cute.make_ptr(
                            Uint64, dst_base + Uint64(SIGNAL_OFFSETS['expert_recv_count_sum']) +
                            Uint64(gen * Uint32(self.local_experts * 8)) +
                            Uint64(dst_local * Uint32(8)),
                            cute.AddressSpace.gmem, assumed_align=8)
                        atom_sys_add_u64(remote_sum, status)
                cute.arch.sync_threads()
                # Each per-expert acquire below imports every rank's route/count
                # stores, replacing the dispatch rank barrier. Include zero-token
                # experts: this join orders prior combine reads before new FC2
                # stores, and generation N's clear before N+2 can publish.
            if cutlass.const_expr(self.eco):
                if block == 0 and tid == 0:
                    self._eco_trace(signals, 36, 0)
                    if cutlass.const_expr(self.eco_rows_after_pub):
                        # The grid rendezvous covered all source-index and count writes. Gated rows start
                        # after those requests; no extra CTA barrier or fence is needed for this handoff.
                        st_release_u32(signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_pub_issued']) // 4,
                                       ld_volatile_u32(signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4) + Uint32(1),
                                       False)
            if cutlass.const_expr(self.eco):
                # ECO keeps the rank barrier: replica pulls read canonical source-index rows on the owner rank,
                # which no per-slot count acquire covers
                # All source-index writes precede the first rank barrier. The
                # signed signal and high-bit grid epoch persist between calls.
                nvlink_barrier(signals, peer_ptrs, Uint32(block), Uint32(tid),
                               self.num_ranks, self.dispatch_sms, False, True,
                               rank_idx, 1, self.nvlink_timeout_secs)
            if cutlass.const_expr(self.eco_trace):
                if block == 0 and tid == 0:
                    self._eco_trace(signals, 1, 0)
                    self._eco_trace_accum(signals, 20)
                    self._eco_trace_ring(signals, 0, False)
            # GCMoEScheduler::fetch_expert_recv_count: each lane caches
            # expert wave*32+lane, including zero for padded lanes.
            expert_counts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            # ECO: single-copy slot totals from the count table (complete after the barrier), replica slots
            # stay zero until the plan role publishes their quotas
            eco_epoch = Uint32(0)
            eco_fixed_tokens = Uint32(0)
            eco_plan_loaded = Uint32(0)
            if cutlass.const_expr(self.eco):
                eco_epoch = ld_volatile_u32(signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4)
                # Single-copy slot totals: the per-source count publication before the rank barrier, as
                # without ECO (no wait on the plan role); replica slots stay zero until the plan's quotas
                lane_tokens = Uint32(0)
                for wave in cutlass.range_constexpr(
                        (self.local_experts + 31) // 32):
                    slot_idx = Uint32(wave * 32) + lane
                    observed = Uint64(0)
                    if slot_idx < eco_num_fixed:
                        while Uint32(observed >> 32) != Uint32(self.dispatch_sms * self.num_ranks):
                            observed = ld_volatile_u64(recv_ptr + slot_idx * Uint32(2))
                    total = Uint32(0)
                    if slot_idx < eco_num_fixed:
                        total = Uint32(observed)
                    expert_counts[wave] = total
                    lane_tokens += total
                if cutlass.const_expr(self.eco_trace):
                    if tid == Uint32(0):
                        self._eco_trace(signals, 32, 0)
                eco_fixed_tokens = cute.arch.warp_redux_sync(lane_tokens, 'add')
            else:
                for wave in cutlass.range_constexpr(
                        (self.local_experts + 31) // 32):
                    expert_idx = Uint32(wave * 32) + lane
                    observed = Uint64(0)
                    if expert_idx < Uint32(self.local_experts):
                        wait_start = read_clock64()
                        while Uint32(observed >> 32) != Uint32(
                                self.dispatch_sms * self.num_ranks):
                            observed = ld_volatile_u64(
                                recv_ptr + expert_idx * Uint32(2))
                            if Uint64(read_clock64() - wait_start) >= Uint64(
                                    self.nvlink_timeout_secs * 2000000000):
                                if block == Uint32(0):
                                    cute.printf(
                                        'GC recv-count timeout role=1 rank={} expert={} gen={} observed_hi={} expected={}',
                                        rank_idx, expert_idx, gen, Uint32(observed >> 32),
                                        self.dispatch_sms * self.num_ranks)
                                device_trap()
                        observed = ld_acquire_sys_u64(
                            recv_ptr + expert_idx * Uint32(2))
                    expert_counts[wave] = Uint32(observed)
            cute.arch.sync_warp()
            if cutlass.const_expr(self.eco_trace):
                if elect_one_sync():
                    self._eco_trace_ring(signals, 12, True)   # field 12: last dispatch warp holds this rank's counts

            if block == Uint32(0) and physical_warp == Uint32(0):
                own_pool_blocks = Uint32(0)
                for wave in cutlass.range_constexpr(
                        (self.local_experts + 31) // 32):
                    own_pool_blocks += (expert_counts[wave] +
                                        Uint32(self.block_m - 1)) // Uint32(self.block_m)
                total_pool_blocks = cute.arch.warp_redux_sync(own_pool_blocks, 'add')
                if lane == Uint32(0):
                    signals[SIGNAL_OFFSETS['routed_pool_blocks'] // 4] = total_pool_blocks

            # One global warp cursor traverses concatenated, unpadded expert
            # tokens; the pool-block prefix advances by each expert's padded
            # M-block count. Only the TMA and final L1 release elect a lane.
            token_idx = Uint32(block) * Uint32(self.dispatch_warps) + Uint32(physical_warp)
            current_expert = Int32(-1)
            cached_rank_expert = Int32(-1)
            expert_start = Uint32(0)
            expert_end = Uint32(0)
            pool_block_offset = Uint32(0)
            pull_phase = Uint32(0)
            rank_waves = (self.num_ranks + 31) // 32
            stored_rank_count = cute.make_rmem_tensor((rank_waves,), Uint32)
            # ECO: replica slots' canonical-order starts, cached with the quotas.
            expert_starts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            for wave in cutlass.range_constexpr(cute.size(expert_starts)):
                expert_starts[wave] = Uint32(0)
            # ECO: replica slots' canonical copy as (logical_expert<<16 |
            # global_slot) — static placement data, resolved once here instead
            # of per slot boundary on the pull path.
            eco_canons = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            for wave in cutlass.range_constexpr(cute.size(eco_canons)):
                eco_canons[wave] = Uint32(0)
            if cutlass.const_expr(self.eco):
                for wave in cutlass.range_constexpr(cute.size(eco_canons)):
                    slot_idx = Uint32(wave * 32) + lane
                    if slot_idx >= eco_num_fixed and \
                            slot_idx < Uint32(self.local_experts):
                        _le = Uint32(ld_volatile_u32(
                            tensor_maps.iterator +
                            Uint32(self.eco_phy2log_words) +
                            rank_idx * Uint32(self.local_experts) +
                            slot_idx)) & Uint32(0xffff)
                        _gs = Uint32(tensor_maps[
                            Uint32(self.eco_copies_words) +
                            _le * Uint32(16)]) & Uint32(0xffff)
                        eco_canons[wave] = (_le << Uint32(16)) | _gs
            # ECO: the cursor enters the replica slots (after every single-copy token) only with the plan
            if cutlass.const_expr(self.eco):
                if eco_plan_loaded == Uint32(0) and token_idx >= eco_fixed_tokens:
                    self._eco_load_plan(signals, rank_idx, eco_epoch,
                                        eco_num_fixed, expert_counts, lane,
                                        expert_starts)
                    eco_plan_loaded = Uint32(1)
            current_expert, expert_start, expert_end, pool_block_offset = \
                advance_expert(
                    expert_counts, token_idx, current_expert, expert_start,
                    expert_end, pool_block_offset, lane, self.local_experts,
                    self.block_m)
            eco_peel_start = Uint32(0)
            eco_canon_rank = rank_idx
            eco_canon_slot = Uint32(0)
            eco_idx_ptr = source_idx.iterator
            slot_total = Uint32(0)
            if cutlass.const_expr(self.pull_chunks == 1 and
                                  self.pull_stages == 2 and self.pull_pipe):
                self._pull_pipeline(
                    signals, peer_ptrs, source_idx, tensor_maps, pull, pull_bar,
                    rank_idx, lane, gen, physical_warp, token_idx,
                    current_expert, expert_start, expert_end, pool_block_offset,
                    stored_rank_count, expert_counts, expert_starts, eco_canons,
                    eco_fixed_tokens, eco_num_fixed, eco_epoch, eco_plan_loaded)
                current_expert = Int32(self.local_experts)
            while current_expert < Int32(self.local_experts):
                if cached_rank_expert != current_expert:
                    cached_rank_expert = current_expert
                    if cutlass.const_expr(self.eco):
                        # The plan kernel materializes replica slots' per-source counts into the
                        # native receive-count rows, so every slot reads counts identically.
                        # Replica slots read their source indices from the logical expert's
                        # canonical (lowest) copy's rows on that copy's rank, not their own.
                        eco_peel_start = Uint32(0)
                        eco_slot_expert = Uint32(current_expert)
                        eco_canon_rank = rank_idx
                        eco_canon_slot = Uint32(current_expert)
                        # Per-slot index-row base: own slot's rows for fixed
                        # slots, the canonical owner's rows for replicas.
                        eco_idx_ptr = source_idx.iterator + \
                            Uint32(current_expert) * \
                            Uint32(self.num_ranks * self.capacity)
                        if Uint32(current_expert) >= eco_num_fixed:
                            eco_peel_start = cached_num_tokens(
                                expert_starts, current_expert, lane,
                                self.local_experts)
                            _pk = cached_num_tokens(
                                eco_canons, current_expert, lane,
                                self.local_experts)
                            _le = _pk >> Uint32(16)
                            _gs = _pk & Uint32(0xffff)
                            eco_canon_rank = _gs // Uint32(self.local_experts)
                            eco_canon_slot = _gs % Uint32(self.local_experts)
                            eco_idx_ptr = cute.make_ptr(
                                Uint32,
                                mapped_peer_base(peer_ptrs, eco_canon_rank,
                                                 self.num_ranks) +
                                Uint64(SIGNAL_OFFSETS['size']) +
                                Uint64(eco_canon_slot) *
                                Uint64(self.num_ranks * self.capacity * 4),
                                cute.AddressSpace.gmem)
                            if cutlass.const_expr(self.eco_recv_counts == 0):
                                eco_slot_expert = _le
                    for wave in cutlass.range_constexpr(rank_waves):
                        source_rank = Uint32(wave * 32) + lane
                        stored_rank_count[wave] = Uint32(0)
                        if source_rank < Uint32(self.num_ranks):
                            count_ptr = signals.iterator + \
                                SIGNAL_OFFSETS['expert_recv_count'] // 4 + \
                                gen * Uint32(self.num_ranks * self.local_experts * 2) + \
                                (source_rank * Uint32(self.local_experts) +
                                 Uint32(current_expert)) * Uint32(2)
                            if cutlass.const_expr(self.eco and self.eco_recv_counts == 0):
                                if Uint32(current_expert) >= eco_num_fixed:
                                    stored_rank_count[wave] = self._eco_table_count(
                                        signals, gen, source_rank, eco_slot_expert,
                                        (eco_epoch // Uint32(2)) % Uint32(3) + Uint32(1))
                                else:
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(count_ptr))
                            elif cutlass.const_expr(self.eco and self.eco_recv_counts == 2):
                                if Uint32(current_expert) >= eco_num_fixed:
                                    stored_rank_count[wave] = Uint32(
                                        ld_volatile_u64(count_ptr))
                                else:
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(count_ptr))
                            elif cutlass.const_expr(self.eco and self.eco_recv_counts == 3):
                                if Uint32(current_expert) >= eco_num_fixed:
                                    eco_ptr = signals.iterator + \
                                        self.eco_recv_count_base // 4 + \
                                        gen * Uint32(self.num_ranks *
                                                     self.local_experts * 2) + \
                                        (source_rank * Uint32(self.local_experts) +
                                         Uint32(current_expert)) * Uint32(2)
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(eco_ptr))
                                else:
                                    stored_rank_count[wave] = Uint32(
                                        ld_global_u64(count_ptr))
                            else:
                                stored_rank_count[wave] = Uint32(
                                    ld_global_u64(count_ptr))
                    if cutlass.const_expr(self.eco_spec):
                        lane_total = Uint32(0)
                        for wave in cutlass.range_constexpr(rank_waves):
                            lane_total += stored_rank_count[wave]
                        slot_total = cute.arch.warp_redux_sync(lane_total, 'add')
                received = expert_end - expert_start
                pull_token = token_idx - expert_start
                pool_block = pool_block_offset + pull_token // self.block_m
                pool_token = pool_block * Uint32(self.block_m) + pull_token % Uint32(self.block_m)
                ring_block = pool_block % self.ring_blocks
                generation = pool_block // self.ring_blocks

                # Each lane owns one source rank in each 32-rank wave.
                # Reduce across its register waves before the warp reduction.
                # A source's remaining count after the peeled rounds is max(count - peel_offset, 0); derive it
                # instead of mutating a copy, so the cached counts stay read-only for the expert's later tokens.
                if cutlass.const_expr(self.eco and self.eco_trace):
                    pt0 = read_globaltimer()
                peel_slot = Uint32(pull_token) + eco_peel_start
                peel_offset = Uint32(0)
                selected = False
                src_rank = Uint32(0)
                src_slot = Uint32(0)
                if cutlass.const_expr(self.eco_spec):
                    # Stale speculative spans can overrun this call's order: fold the
                    # overflow back into it (a rank-0 dummy hotspots every rank on the
                    # same remote address). slot_total == 0 has no order: rank-0 dummy.
                    if slot_total == Uint32(0):
                        selected = True
                    elif peel_slot >= slot_total:
                        peel_slot = peel_slot % slot_total
                while not selected:
                    lane_active = Uint32(0)
                    lane_minimum = Uint32(0xffffffff)
                    for wave in cutlass.range_constexpr(rank_waves):
                        count = stored_rank_count[wave] - min(stored_rank_count[wave], peel_offset)
                        lane_active += Uint32(1) if count > Uint32(0) else Uint32(0)
                        if count > Uint32(0):
                            lane_minimum = min(lane_minimum, count)
                    active = cute.arch.warp_redux_sync(lane_active, 'add')
                    minimum = cute.arch.warp_redux_sync(lane_minimum, 'min')
                    round_tokens = active * minimum
                    if peel_slot < round_tokens:
                        position = peel_slot % active
                        seen = Uint32(0)
                        for wave in cutlass.range_constexpr(rank_waves):
                            mask = Uint32(cute.arch.vote_ballot_sync(
                                stored_rank_count[wave] > peel_offset))
                            wave_active = Uint32(cute.arch.popc(mask))
                            if position >= seen and position < seen + wave_active:
                                src_rank = Uint32(wave * 32) + warp_nth_set_bit(
                                    mask, position - seen + Uint32(1))
                            seen += wave_active
                        src_slot = peel_offset + peel_slot // active
                        selected = True
                    else:
                        peel_slot -= round_tokens
                        peel_offset += minimum
                if cutlass.const_expr(self.eco and self.eco_trace):
                    pt1 = read_globaltimer()
                if cutlass.const_expr(self.eco):
                    # Fixed slots resolve to their own rows, replica slots to
                    # the canonical copy's rows — both via the per-slot base.
                    token_topk = Uint32(eco_idx_ptr[
                        Uint32(src_rank) * Uint32(self.capacity) + src_slot])
                else:
                    token_topk = source_idx[(Uint32(current_expert) * Uint32(self.num_ranks) +
                                             src_rank) * Uint32(self.capacity) + src_slot]
                if cutlass.const_expr(self.eco_spec):
                    token_topk = min(token_topk, Uint32(self.num_tokens) * Uint32(self.topk) - Uint32(1))
                if cutlass.const_expr(self.eco_route_verify):
                    if lane == Uint32(0):
                        st_global_u32(cute.make_ptr(
                            Uint32, mapped_peer_base(peer_ptrs, src_rank, self.num_ranks) +
                            Uint64(self.eco_base + ECO_SIGNAL_OFFSETS['eco_route_dbg']) + Uint64(token_topk) * Uint64(4),
                            cute.AddressSpace.gmem, assumed_align=4), rank_idx)
                src_token = token_topk // Uint32(self.topk)
                topk_slot = token_topk % Uint32(self.topk)
                if cutlass.const_expr(self.eco and self.eco_trace):
                    pt2 = read_globaltimer()
                if generation > 0:
                    empty_ptr = signals.iterator + \
                        SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block
                    target = generation * (2 * self.intermediate // 128)
                    while ld_acquire_gpu_u32(empty_ptr) < target:
                        pass
                if cutlass.const_expr(self.eco and self.eco_trace):
                    pt3 = read_globaltimer()
                src_base = mapped_peer_base(peer_ptrs, src_rank, self.num_ranks)
                src_x = cute.make_ptr(
                    Uint32, src_base + Uint64(self.input_x_words * 4) +
                    Uint64(src_token) * Uint64(self.hidden),
                    cute.AddressSpace.gmem, assumed_align=16)
                dst_x = cute.recast_ptr(
                    cute.recast_ptr(signals.iterator, dtype=Uint8) +
                    Uint64(self.l1_x_words) * Uint64(4) +
                    Uint64(pool_token % self.ring_tokens) * Uint64(self.hidden),
                    dtype=Uint32)
                warp_pull = pull + physical_warp * self.pull_stages * \
                    (self.pull_bytes // 4)
                warp_bar = pull_bar + physical_warp * self.pull_stages
                if elect_one_sync():
                    if cutlass.const_expr(self.eco and self.eco_trace):
                        if token_idx >= eco_fixed_tokens and token_idx < eco_fixed_tokens + \
                                Uint32(self.dispatch_sms * self.dispatch_warps):
                            self._eco_trace_ring(signals, 13, True)
                    for chunk in cutlass.range_constexpr(self.pull_chunks):
                        byte_words = chunk * (self.pull_bytes // 4)
                        bulk_g2s_1d(warp_pull, src_x + byte_words,
                                     Uint32(self.pull_bytes), warp_bar)
                        cute.arch.mbarrier_arrive_and_expect_tx(warp_bar, self.pull_bytes)
                        if chunk + 1 < self.pull_chunks:
                            cute.arch.mbarrier_wait(warp_bar, pull_phase)
                            pull_phase ^= Uint32(1)
                            bulk_s2g_1d(dst_x + byte_words, warp_pull,
                                         Uint32(self.pull_bytes))
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.eco and self.eco_trace):
                    pt4 = read_globaltimer()

                sf_row = ring_block * ((self.block_m + 127) // 128 * 128) + \
                    ((pull_token % self.block_m) & ~127) + \
                    ((pull_token % self.block_m) & 31) * 4 + \
                    (((pull_token % self.block_m) >> 5) & 3)
                remote_sf_base = src_base + Uint64(self.input_sf_words * 4) + \
                    Uint64(src_token) * Uint64(self.hidden // 128 * 4)
                remote_weight = cute.make_ptr(
                    Uint32, src_base + Uint64(self.input_weight_words * 4) +
                    Uint64(token_topk) * Uint64(4), cute.AddressSpace.gmem)
                weight = remote_weight[0]
                if cutlass.const_expr((self.hidden // 128) % 4 == 0):
                    sf_vectors = self.hidden // 512
                    for q_round in cutlass.range(
                            (sf_vectors + 31) // 32, unroll_full=True):
                        q = q_round * 32 + lane
                        if q < sf_vectors:
                            remote_vec = cute.make_ptr(
                                Uint32, remote_sf_base + Uint64(q * Uint32(16)),
                                cute.AddressSpace.gmem, assumed_align=16)
                            values = ld_global_u32x4(remote_vec)
                            for component in cutlass.range_constexpr(4):
                                sf_col = q * 4 + component
                                if cutlass.const_expr(self.wide_l1_sf):
                                    l1_sf_bytes = cute.recast_ptr(
                                        signals.iterator, dtype=Uint8) + \
                                        Uint64(self.l1_sf_words) * Uint64(4)
                                    sf_word = Uint64(sf_col) * \
                                        Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                    cute.recast_ptr(
                                        l1_sf_bytes + sf_word * Uint64(4),
                                        dtype=Uint32)[0] = values[component]
                                else:
                                    signals[self.l1_sf_words +
                                            sf_col * self.sf_ring_tokens + sf_row] = \
                                        values[component]
                else:
                    for sf_round in cutlass.range(
                            (self.hidden // 128 + 31) // 32, unroll_full=True):
                        sf_col = sf_round * 32 + lane
                        if sf_col < self.hidden // 128:
                            remote_word = cute.make_ptr(
                                Uint32, remote_sf_base + Uint64(sf_col * Uint32(4)),
                                cute.AddressSpace.gmem, assumed_align=4)
                            if cutlass.const_expr(self.wide_l1_sf):
                                l1_sf_bytes = cute.recast_ptr(
                                    signals.iterator, dtype=Uint8) + \
                                    Uint64(self.l1_sf_words) * Uint64(4)
                                sf_word = Uint64(sf_col) * \
                                    Uint64(self.sf_ring_tokens) + Uint64(sf_row)
                                cute.recast_ptr(
                                    l1_sf_bytes + sf_word * Uint64(4),
                                    dtype=Uint32)[0] = remote_word[0]
                            else:
                                signals[self.l1_sf_words +
                                        sf_col * self.sf_ring_tokens + sf_row] = \
                                    remote_word[0]
                cute.arch.sync_warp()
                if cutlass.const_expr(self.eco and self.eco_trace):
                    pt5 = read_globaltimer()
                if elect_one_sync():
                    signals[self.l1_weight_words + pool_token % self.ring_tokens] = weight
                    signals[self.metadata_words + pool_token * 3] = src_rank
                    signals[self.metadata_words + pool_token * 3 + 1] = src_token
                    signals[self.metadata_words + pool_token * 3 + 2] = topk_slot
                    # The last remote G2S overlapped weight/SF work. Drain
                    # it, store the chunk, then publish W1 after S2G finishes.
                    cute.arch.mbarrier_wait(warp_bar, pull_phase)
                    pull_phase ^= Uint32(1)
                    final_byte_words = (self.pull_chunks - 1) * (self.pull_bytes // 4)
                    bulk_s2g_1d(dst_x + final_byte_words, warp_pull,
                                 Uint32(self.pull_bytes))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
                    is_last_token = token_idx == expert_end - Uint32(1)
                    full_increment = Uint32(self.block_m) - (pull_token % self.block_m) \
                        if is_last_token else Uint32(1)
                    red_release_gpu_add_u32(
                        signals.iterator + SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block,
                        Uint32(full_increment))
                    if cutlass.const_expr(self.eco and self.eco_trace):
                        if token_idx >= eco_fixed_tokens and token_idx < eco_fixed_tokens + \
                                Uint32(self.dispatch_sms * self.dispatch_warps):
                            self._eco_trace_ring(signals, 14, True)
                cute.arch.sync_warp()
                if cutlass.const_expr(self.eco and self.eco_trace):
                    # serial-path per-token phases (ns sums): trace slots 44.. replica tokens, 52.. single-copy
                    pt6 = read_globaltimer()
                    if elect_one_sync():
                        pbase = signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4 + \
                            (Uint32(44) if Uint32(current_expert) >= eco_num_fixed else Uint32(52)) * Uint32(2)
                        atom_add_u64(pbase, Uint64(1))
                        atom_add_u64(pbase + 2, pt1 - pt0)
                        atom_add_u64(pbase + 4, pt2 - pt1)
                        atom_add_u64(pbase + 6, pt3 - pt2)
                        atom_add_u64(pbase + 8, pt4 - pt3)
                        atom_add_u64(pbase + 10, pt5 - pt4)
                        atom_add_u64(pbase + 12, pt6 - pt5)
                    cute.arch.sync_warp()
                token_idx += Uint32(self.dispatch_sms * self.dispatch_warps)
                if cutlass.const_expr(self.eco):
                    if eco_plan_loaded == Uint32(0) and token_idx >= eco_fixed_tokens:
                        self._eco_load_plan(signals, rank_idx, eco_epoch,
                                            eco_num_fixed, expert_counts, lane,
                                            expert_starts)
                        eco_plan_loaded = Uint32(1)
                current_expert, expert_start, expert_end, pool_block_offset = \
                    advance_expert(
                        expert_counts, token_idx, current_expert, expert_start,
                        expert_end, pool_block_offset, lane, self.local_experts,
                        self.block_m)
            if cutlass.const_expr(self.eco_trace):
                if elect_one_sync():
                    self._eco_trace(signals, 8, 2)
                    self._eco_trace_ring(signals, 6, True)
            if cutlass.const_expr(self.eco_combine_exact == 1):
                self._eco_route_peers(signals, tensor_maps, routes, rank_idx, num_tokens, lane,
                                      Uint32(block * self.dispatch_warps + physical_warp), gen, eco_epoch)
        elif cutlass.const_expr(self.role == 2):
            if physical_warp == 0:
                # Each descriptor has its own aligned by-value kernel
                # parameter, in the configuration's map order.
                prefetch_tensormap(tensor_maps.iterator)
                prefetch_tensormap(tensor_map_1.iterator)
                prefetch_tensormap(tensor_map_2.iterator)
                prefetch_tensormap(tensor_map_3.iterator)
                prefetch_tensormap(tensor_map_4.iterator)
                prefetch_tensormap(tensor_map_5.iterator)
                prefetch_tensormap(tensor_map_6.iterator)
                prefetch_tensormap(tensor_map_7.iterator)
                prefetch_tensormap(tensor_map_8.iterator)
                prefetch_tensormap(tensor_map_9.iterator)
                prefetch_tensormap(tensor_map_10.iterator)
                prefetch_tensormap(tensor_map_11.iterator)
                prefetch_tensormap(tensor_map_12.iterator)
                prefetch_tensormap(tensor_map_13.iterator)
                prefetch_tensormap(tensor_map_14.iterator)
                prefetch_tensormap(tensor_map_15.iterator)
                prefetch_tensormap(tensor_map_16.iterator)
                prefetch_tensormap(tensor_map_17.iterator)
                if cutlass.const_expr(self.use_small_tile_path):
                    prefetch_tensormap(tensor_map_18.iterator)
                    prefetch_tensormap(tensor_map_19.iterator)
                    prefetch_tensormap(tensor_map_20.iterator)
                    prefetch_tensormap(tensor_map_21.iterator)
            smem = cutlass.utils.SmemAllocator()
            slab = cute.recast_ptr(
                smem.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
            full = cute.recast_ptr(slab + self.shared_layout.task_full, dtype=Uint64)
            empty = cute.recast_ptr(slab + self.shared_layout.task_empty, dtype=Uint64)
            task = cute.recast_ptr(slab + self.shared_layout.tasks, dtype=Uint32)
            cd_smem = slab + self.shared_layout.cd
            data_a = slab + self.shared_layout.data_a
            data_b = slab + self.shared_layout.data_b
            data_sfa = cute.recast_ptr(slab + self.shared_layout.data_sfa, dtype=Uint32)
            data_sfb = cute.recast_ptr(slab + self.shared_layout.data_sfb, dtype=Uint32)
            epi_amax = cute.recast_ptr(slab + self.shared_layout.amax, dtype=Float32)
            data_full = cute.recast_ptr(slab + self.shared_layout.data_full, dtype=Uint64)
            data_empty = cute.recast_ptr(slab + self.shared_layout.data_empty, dtype=Uint64)
            accum_full = cute.recast_ptr(slab + self.shared_layout.accum_full, dtype=Uint64)
            accum_empty = cute.recast_ptr(slab + self.shared_layout.accum_empty, dtype=Uint64)
            tptr = cute.recast_ptr(slab + self.shared_layout.tmem_ptr, dtype=Uint32)
            cta_rank, _, _ = cute.arch.block_in_cluster_idx()
            self._initialize_shared_storage(slab, Uint32(physical_warp), Uint32(lane))
            # Warpgroup-uniform deallocation/allocation. The first WG
            # runs load/MMA/scheduler; the other two run epilogue.
            if physical_warp < 4:
                warpgroup_reg_dealloc(self.non_epi_registers)
            else:
                warpgroup_reg_alloc(self.epi_registers)
            iteration = Uint32(0)
            # Dual-buffered receive sums: poll the generation selected by
            # the persistent parity word (volatile read, fresh from L2).
            gen = ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['recv_parity'] // 4) & Uint32(1)
            recv_ptr = signals.iterator + SIGNAL_OFFSETS['expert_recv_count_sum'] // 4 + \
                gen * Uint32(self.local_experts * 2)
            l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['l1_task_count'] // 4
            l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['l2_task_count'] // 4
            shared_l1_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l1_task_count'] // 4
            shared_l2_count_ptr = signals.iterator + SIGNAL_OFFSETS['shared_l2_task_count'] // 4
            expert_counts = cute.make_rmem_tensor(
                ((self.local_experts + 31) // 32,), Uint32)
            if cta_rank == 0 and physical_warp == 3:
                if cutlass.const_expr(self.eco_trace):
                    if elect_one_sync():
                        self._eco_trace(signals, 9, 1)
                schedule_tasks(
                    task, full, empty, recv_ptr, l1_count_ptr, l2_count_ptr,
                    shared_l1_count_ptr, shared_l2_count_ptr, expert_counts,
                    iteration, Int32(lane), Int32(block + self.cta_base), num_tokens,
                    self.block_m,
                    self.compute_sms,
                    self.dispatch_sms,
                    self.hidden,
                    self.intermediate,
                    self.l1_clusters,
                    self.l2_clusters,
                    self.local_experts,
                    self.num_ranks,
                    self.shared_experts,
                    self.eco,
                    signals.iterator,
                    rank_idx,
                    self.eco_trace,
                    self.eco_base,
                    eco_fixed_slots,
                    self.eco_spec,
                )
                if cutlass.const_expr(self.eco_trace):
                    if elect_one_sync():
                        self._eco_trace(signals, 13, 2)
            # The same two-slot task stream drives both TMA producer warps
            # and the leader MMA warp. All three consume every valid phase.
            if physical_warp == 0:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        expert_more = Uint32(task_words[task_stage * 8 + 2]) >> Uint32(31)
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        block_idx = pool_block % Uint32(self.ring_blocks)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        # The MMA B operand consumes N/2 rows from each CTA.
                        # Keep the original stage pitch and full shared/fallback maps.
                        activation_rows = Uint32(self.block_m // 2)
                        if cutlass.const_expr(self.use_small_tile_path):
                            if is_shared == Uint32(0) and valid_m <= Uint32(32):
                                activation_rows = Uint32(16)
                                activation_map = tensor_map_18.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_19.iterator
                        if cutlass.const_expr(self.use_small_tile_path and self.block_m > 48):
                            if is_shared == Uint32(0) and valid_m > Uint32(32) and valid_m <= Uint32(48):
                                activation_rows = Uint32(24)
                                activation_map = tensor_map_20.iterator
                                if task_phase == Uint32(GCBlockPhase.Linear2):
                                    activation_map = tensor_map_21.iterator
                        # Keep this pair generic for the entire K loop.
                        # The conversion is full-warp, before elected TMA.
                        activation_map = generic_tensormap_ptr(activation_map)
                        activation_sf_map = generic_tensormap_ptr(activation_sf_map)
                        if task_phase == Uint32(GCBlockPhase.Linear1):
                            l1_full_ptr = signals.iterator + \
                                SIGNAL_OFFSETS['l1_full_count'] // 4 + block_idx
                            required = self.block_m * \
                                (pool_block // Uint32(self.ring_blocks) + Uint32(1))
                            if cutlass.const_expr(self.eco and self.eco_trace):
                                input_wait_begin = read_globaltimer()
                            while ld_acquire_gpu_u32(l1_full_ptr) != required:
                                pass
                            if cutlass.const_expr(self.eco_trace and not self.eco):
                                # plain GEMM timeline: first / last L1 task start (inputs ready) per call
                                if elect_one_sync():
                                    self._eco_trace_ring(signals, 13, True)   # inverted max = earliest
                                    self._eco_trace_ring(signals, 3, True)
                            if cutlass.const_expr(self.eco and self.eco_trace):
                                input_wait_ns = Uint32(read_globaltimer() - input_wait_begin)
                                if local_expert >= eco_fixed_slots:
                                    if elect_one_sync():
                                        base = signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4
                                        entry = Uint32(ld_volatile_u64(base + 33 * 2)) % Uint32(64)
                                        atom_max_u32_global(
                                            signals.iterator + (self.eco_base + ECO_SIGNAL_OFFSETS['eco_trace_ring']) // 4 +
                                            entry * Uint32(ECO_TRACE_RING_FIELDS) + 15, input_wait_ns)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                shared_full_ptr = signals.iterator + \
                                    SIGNAL_OFFSETS['shared_l2_full_count'] // 4 + block_idx
                                while ld_acquire_gpu_u32(shared_full_ptr) != \
                                        self.l1_clusters * self.shared_experts * 2:
                                    pass
                        # GCL2KBlockDependency belongs to this task, so
                        # the observed XOR-normalized mask survives all
                        # K128 waits and is reset for the next task.
                        mask_ptr = signals.iterator + \
                            SIGNAL_OFFSETS['l2_full_mask'] // 4 + \
                            (pool_block % Uint32(self.ring_blocks)) * 2
                        generation = pool_block // Uint32(self.ring_blocks)
                        expected = Uint64(0) if (generation & Uint32(1)) != Uint32(0) \
                            else Uint64(self.l1_full_mask)
                        pending = Uint64(self.l1_full_mask)
                        if cutlass.const_expr(self.eco_trace and not self.eco):
                            if task_phase == Uint32(GCBlockPhase.Linear2):   # first / last L2 task start per call
                                if elect_one_sync():
                                    self._eco_trace_ring(signals, 14, True)   # inverted max = earliest
                                    self._eco_trace_ring(signals, 7, True)
                        k_block = Uint32(0)
                        while k_block < shape_k // Uint32(128):
                            if task_phase == Uint32(GCBlockPhase.Linear2):
                                required_mask = Uint64(3) << (k_block * Uint32(2))
                                while (pending & required_mask) != Uint64(0):
                                    pending = ld_acquire_gpu_u64(mask_ptr) ^ expected
                            load_mixed_k128_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + data_stage * (self.block_m // 2 * 128),
                                data_sfa + data_stage * self.sf_block_m,
                                data_b + data_stage * (128 * 128),
                                data_sfb + data_stage * 128,
                                data_full + data_stage, data_empty + data_stage,
                                k_block, data_phase, block_idx,
                                n_cluster * Uint32(2) + cta_rank,
                                local_expert, valid_m, activation_rows, shape_k,
                                is_shared, cta_rank, 0,
                                self.block_m, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.shared_weight_cache_hint,
                                self.routed_weight_linger, expert_more)
                            data_stage = Uint32(0) if data_stage == \
                                Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                            k_block += Uint32(1)
                    task_iter += Uint32(1)
            elif physical_warp == 1:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        pool_block = Uint32(task_words[task_stage * 8 + 4])
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        expert_more = Uint32(task_words[task_stage * 8 + 2]) >> Uint32(31)
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        n_cluster = Uint32(task_words[task_stage * 8 + 3])
                        local_expert = Uint32(task_words[task_stage * 8 + 1])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        block_idx = pool_block % Uint32(self.ring_blocks)
                        if cutlass.const_expr(self.shared_experts > 0):
                            if is_shared != Uint32(0):
                                block_idx = pool_block
                        activation_map = tensor_maps.iterator
                        activation_sf_map = tensor_map_1.iterator
                        weight_map = tensor_map_2.iterator
                        weight_sf_map = tensor_map_3.iterator
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            activation_map = tensor_map_5.iterator
                            activation_sf_map = tensor_map_6.iterator
                            weight_map = tensor_map_7.iterator
                            weight_sf_map = tensor_map_8.iterator
                        if cutlass.const_expr(self.shared_experts > 0):
                            if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                activation_map = tensor_map_9.iterator
                                activation_sf_map = tensor_map_10.iterator
                                weight_map = tensor_map_11.iterator
                                weight_sf_map = tensor_map_12.iterator
                            elif task_phase == Uint32(GCBlockPhase.SharedLinear2):
                                activation_map = tensor_map_14.iterator
                                activation_sf_map = tensor_map_15.iterator
                                weight_map = tensor_map_16.iterator
                                weight_sf_map = tensor_map_17.iterator
                        weight_map = generic_tensormap_ptr(weight_map)
                        weight_sf_map = generic_tensormap_ptr(weight_sf_map)
                        k_block = Uint32(0)
                        while k_block < shape_k // Uint32(128):
                            # The weight loader ignores the activation geometry.
                            load_mixed_k128_tma(
                                activation_map, activation_sf_map,
                                weight_map, weight_sf_map,
                                data_a + data_stage * (self.block_m // 2 * 128),
                                data_sfa + data_stage * self.sf_block_m,
                                data_b + data_stage * (128 * 128),
                                data_sfb + data_stage * 128,
                                data_full + data_stage, data_empty + data_stage,
                                k_block, data_phase, block_idx,
                                n_cluster * Uint32(2) + cta_rank,
                                local_expert, valid_m, Uint32(0), shape_k,
                                is_shared, cta_rank, 1,
                                self.block_m, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.shared_weight_cache_hint,
                                self.routed_weight_linger, expert_more)
                            data_stage = Uint32(0) if data_stage == \
                                Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                            k_block += Uint32(1)
                    task_iter += Uint32(1)
            if cta_rank == 0 and physical_warp == 2:
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                # Create the two swapped K-major data descriptor bases
                # once for the persistent leader MMA warp.
                weight_desc_base = kmajor_smem_desc(
                    data_b, Uint32(1024), True, Uint32(0))
                activation_desc_base = kmajor_smem_desc(
                    data_a, Uint32(1024), True, Uint32(0))
                data_desc_hi = (weight_desc_base >> Uint64(32)) << Uint64(32)
                sf_desc_hi = kmajor_smem_desc_from_start(
                    Uint32(0), Uint32(128), False)
                task_iter = Uint32(0)
                data_stage = Uint32(0)
                data_phase = Uint32(0)
                running = True
                while running:
                    task_stage = task_iter % Uint32(2)
                    cute.arch.mbarrier_wait(full + task_stage,
                                            (task_iter // Uint32(2)) % Uint32(2))
                    task_phase = task_words[task_stage * 8]
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    if running:
                        valid_m = Uint32(task_words[task_stage * 8 + 5])
                        shape_k = Uint32(task_words[task_stage * 8 + 7])
                        is_shared = Uint32(0)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_shared = Uint32(1) if task_phase >= \
                                Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                        num_k_blocks = shape_k // Uint32(128)
                        weight_format = Uint32(5)
                        if cutlass.const_expr(self.shared_experts > 0):
                            weight_format = select_weight_format(is_shared)
                        aligned_n = (valid_m + Uint32(15)) & Uint32(~15)
                        instr_desc_base = blockscaled_instr_desc_hi(
                            aligned_n, weight_format, Uint32(0), Uint32(0))
                        accum_stage = task_iter % Uint32(self.accum_stages)
                        accum_phase = (task_iter // Uint32(self.accum_stages)) % Uint32(2)
                        cute.arch.mbarrier_wait(accum_empty + accum_stage,
                                                accum_phase ^ Uint32(1))
                        tcgen05_after_thread_sync()
                        # Unroll K128 MMA stages by the configured group size.
                        for k_iter in cutlass.range(
                                Int32(0), Int32(num_k_blocks),
                                Int32(1), unroll=4):
                            k_block = Uint32(k_iter)
                            cute.arch.mbarrier_wait(data_full + data_stage,
                                                    data_phase)
                            # Keep descriptor layout bits fixed and
                            # advance only the 16-byte SMEM address field.
                            weight_desc_start = Uint32(
                                Uint32(weight_desc_base) + data_stage * Uint32(1024))
                            activation_desc_start = Uint32(
                                Uint32(activation_desc_base) + data_stage *
                                Uint32((self.block_m // 2 * 128) // 16))
                            issue_mixed_k128(
                                weight_desc_start, activation_desc_start,
                                data_sfb, data_sfa, data_stage,
                                data_desc_hi, sf_desc_hi,
                                accum_stage * Uint32(self.block_m),
                                Uint32(self.accum_stages * self.block_m +
                                       self.sfa_tmem_cols),
                                Uint32(self.accum_stages * self.block_m),
                                data_empty + data_stage, accum_full + accum_stage,
                                instr_desc_base, k_block,
                                Uint32(k_block == num_k_blocks - Uint32(1)),
                                is_shared, self.sf_block_m,
                                self.shared_weight_gran_k,
                                self.shared_experts > 0,
                                self.use_small_tile_path)
                            data_stage = Uint32(0) if data_stage == \
                                Uint32(self.math_stages - 1) else data_stage + Uint32(1)
                            data_phase ^= Uint32(data_stage == Uint32(0))
                    task_iter += Uint32(1)
                # The terminal None task increments task_iter once. Drain
                # the final valid accumulator stage before leaving the
                # producer warp, as the reference does after its loop.
                if task_iter > Uint32(1):
                    last_valid_iter = task_iter - Uint32(2)
                    last_stage = last_valid_iter % Uint32(self.accum_stages)
                    last_phase = (last_valid_iter // Uint32(self.accum_stages)) % Uint32(2)
                    cute.arch.mbarrier_wait(accum_empty + last_stage,
                                            last_phase)
            if physical_warp >= 4:
                if tptr[0] != Uint32(0):
                    device_trap()
                task_words = cute.make_tensor(task, cute.make_layout((16,)))
                iteration = Uint32(0)
                running = True
                while running:
                    stage = iteration % 2
                    generation_phase = (iteration // 2) % 2
                    cute.arch.mbarrier_wait(full + stage, generation_phase)
                    task_phase = task_words[stage * 8]
                    n_cluster = Uint32(task_words[stage * 8 + 3])
                    pool_block = Uint32(task_words[stage * 8 + 4])
                    valid_m = Uint32(task_words[stage * 8 + 5])
                    if task_phase != Uint32(GCBlockPhase.None_):
                        accum_stage = iteration % Uint32(self.accum_stages)
                        accum_phase = (iteration // Uint32(self.accum_stages)) % Uint32(2)
                        cute.arch.mbarrier_wait(accum_full + accum_stage,
                                                accum_phase)
                        tcgen05_after_thread_sync()
                        release_task_info(empty, Int32(stage))
                        # Match the reference's lane-0 exchange so valid-M
                        # controls the epilogue uniformly within each warp.
                        valid_m = cute.arch.shuffle_sync(valid_m, 0)
                        if physical_warp == 4 and \
                                task_phase == Uint32(GCBlockPhase.Linear2):
                            if elect_one_sync():
                                # The final MMA commit completed every K128
                                # input read. The BF16 peer scatter below uses
                                # TMEM, so L1 may now reuse the input ring.
                                ring_block = pool_block % Uint32(self.ring_blocks)
                                red_gpu_add_u32(
                                    signals.iterator +
                                    SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block,
                                    Uint32(1))
                        if task_phase == Uint32(GCBlockPhase.Linear2):
                            # The reference joins every epilogue warp after
                            # the elected routed L2 ring release.
                            cute.arch.sync_warp()
                        is_l1 = task_phase == Uint32(GCBlockPhase.Linear1)
                        if cutlass.const_expr(self.shared_experts > 0):
                            is_l1 = is_l1 or task_phase == Uint32(GCBlockPhase.SharedLinear1)
                        if is_l1:
                            # W1 epilogue: the 16dp register mapping below
                            # predicates stores with this task's valid rows.
                            is_shared_l1 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l1 = Uint32(1) if task_phase == \
                                    Uint32(GCBlockPhase.SharedLinear1) else Uint32(0)
                            ring_block = pool_block % Uint32(self.ring_blocks)
                            generation = pool_block // Uint32(self.ring_blocks)
                            if not is_shared_l1:
                                l2_empty_ptr = signals.iterator + \
                                    SIGNAL_OFFSETS['l2_empty_count'] // 4 + ring_block
                                while ld_acquire_gpu_u32(l2_empty_ptr) != \
                                        generation * Uint32(self.l2_clusters * 2):
                                    pass
                            output_block = ring_block
                            if is_shared_l1:
                                output_block = pool_block
                            route_weights = cute.recast_ptr(
                                signals.iterator + self.l1_weight_words,
                                dtype=Float32)
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            wg_start_m = warpgroup * Uint32(self.block_m // 2)
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > Uint32(self.block_m // 2):
                                    valid_wg_rows = Uint32(self.block_m // 2)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l1 - 1)) // \
                                Uint32(self.store_m_l1)
                            n_block = n_cluster * Uint32(2) + cta_rank
                            cd_wg = cd_smem + warpgroup * Uint32(
                                2 * self.store_m_l1 * 64)
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l1 // 8, 8), Float32)
                            stored_cached_weight = Float32(1)
                            clamp_value = activation_clamp
                            if cutlass.const_expr(activation_clamp is None):
                                # Preserve the old direct diagnostic call
                                # while production supplies the FP32 ABI arg.
                                clamp_value = Float32(self.activation_clamp)
                            if valid_stores == Uint32(0):
                                _release_w1_tmem(accum_empty, accum_stage)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        base = accum_stage * Uint32(self.block_m) + \
                                            wg_start_m + Uint32(atom * 8)
                                        _load_w1_atom(raw_values, 0, atom, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w1_tmem(accum_empty, accum_stage)
                            for store in cutlass.range_constexpr(
                                    (self.block_m // 2) // self.store_m_l1):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l1 // 8):
                                            base = accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l1 + atom * 8)
                                            _load_w1_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l1 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage)
                                    # BM240 overlaps the next register
                                    # block's loads with this block's math.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            for atom in cutlass.range_constexpr(
                                                    self.store_m_l1 // 8):
                                                next_base = accum_stage * Uint32(self.block_m) + \
                                                    wg_start_m + Uint32(
                                                        (store + 1) * self.store_m_l1 + atom * 8)
                                                _load_w1_atom(
                                                    raw_values, (store + 1) % 2,
                                                    atom, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w1_tmem(accum_empty, accum_stage)
                                    raw_buffer = store % 2 if self.block_m == 240 else 0
                                    activation_values = cute.make_rmem_tensor(
                                        (self.store_m_l1 // 8, 4), Float32)
                                    reduced_amax = cute.make_rmem_tensor(
                                        (self.store_m_l1 // 8, 2), Float32)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        j = store * (self.store_m_l1 // 8) + atom
                                        if not is_shared_l1 and j % 4 == 0:
                                            if self.block_m // 2 % 32 == 0 or \
                                                    Uint32(j * 8 + lane) < \
                                                    Uint32(self.block_m // 2):
                                                weight_token = ring_block * \
                                                    Uint32(self.block_m) + wg_start_m + \
                                                    Uint32(j * 8 + lane)
                                                stored_cached_weight = \
                                                    route_weights[weight_token]
                                        source_lane = Uint32((j * 8) % 32) + \
                                            (Uint32(lane) % Uint32(4)) * Uint32(2)
                                        weight0 = cute.arch.shuffle_sync(
                                            stored_cached_weight, source_lane)
                                        weight1 = cute.arch.shuffle_sync(
                                            stored_cached_weight, source_lane + Uint32(1))
                                        value0, value1 = _swiglu_pair(
                                            raw_values[raw_buffer, atom, 0],
                                            raw_values[raw_buffer, atom, 1],
                                            raw_values[raw_buffer, atom, 2],
                                            raw_values[raw_buffer, atom, 3],
                                            weight0, weight1, clamp_value,
                                            Float32(1), self.has_activation_clamp,
                                            False, self.fast_math)
                                        value2, value3 = _swiglu_pair(
                                            raw_values[raw_buffer, atom, 4],
                                            raw_values[raw_buffer, atom, 5],
                                            raw_values[raw_buffer, atom, 6],
                                            raw_values[raw_buffer, atom, 7],
                                            weight0, weight1, clamp_value,
                                            Float32(1), self.has_activation_clamp,
                                            False, self.fast_math)
                                        activation_values[atom, 0] = value0
                                        activation_values[atom, 1] = value1
                                        activation_values[atom, 2] = value2
                                        activation_values[atom, 3] = value3
                                        amax0 = warp_group4_max(
                                            fmax_f32(
                                                fmax_f32(Float32(0), abs_f32(value0)),
                                                abs_f32(value2)))
                                        amax1 = warp_group4_max(
                                            fmax_f32(
                                                fmax_f32(Float32(0), abs_f32(value1)),
                                                abs_f32(value3)))
                                        reduced_amax[atom, 0] = amax0
                                        reduced_amax[atom, 1] = amax1
                                        if lane < 4:
                                            amax_offset = epi_warp * \
                                                Uint32(self.store_m_l1) + \
                                                Uint32(atom * 8 + lane * 2)
                                            epi_amax[amax_offset] = amax0
                                            epi_amax[amax_offset + 1] = amax1
                                        cute.arch.sync_warp()
                                    tma_store_wait_1()
                                    sync_epilogue_warpgroup(warpgroup)
                                    tma_stage = store % 2
                                    cd_stage = cd_wg + Uint32(
                                        tma_stage * self.store_m_l1 * 64)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l1 // 8):
                                        partner_offset = (epi_warp ^ Uint32(1)) * \
                                            Uint32(self.store_m_l1) + \
                                            Uint32(atom * 8) + \
                                            (Uint32(lane) % Uint32(4)) * Uint32(2)
                                        sf_amax0 = fmax_f32(
                                            reduced_amax[atom, 0],
                                            epi_amax[partner_offset])
                                        sf_amax1 = fmax_f32(
                                            reduced_amax[atom, 1],
                                            epi_amax[partner_offset + 1])
                                        sf_exp0 = ue8m0_fp8_scale_exp(sf_amax0)
                                        sf_exp1 = ue8m0_fp8_scale_exp(sf_amax1)
                                        inv0 = ue8m0_fp8_inv(sf_exp0)
                                        inv1 = ue8m0_fp8_inv(sf_exp1)
                                        upper0, upper1 = cute.arch.mul_packed_f32x2(
                                            (activation_values[atom, 0],
                                             activation_values[atom, 1]), (inv0, inv1), rnd='rn', ftz=False)
                                        lower0, lower1 = cute.arch.mul_packed_f32x2(
                                            (activation_values[atom, 2],
                                             activation_values[atom, 3]), (inv0, inv1), rnd='rn', ftz=False)
                                        packed = _w1_pack_fp8x4(
                                            upper0, upper1, lower0, lower1)
                                        row = Uint32(lane)
                                        smem_offset = Uint32(atom * 8 * 64) + \
                                            row * Uint32(64) + \
                                            (warp_in_wg ^ (row // Uint32(2))) * Uint32(16)
                                        stmatrix_fp8x4_trans_late(
                                            cd_stage, smem_offset, packed)
                                        if warp_in_wg % Uint32(2) == Uint32(0) and lane < 4:
                                            output_sf_words = Uint64(self.l2_sf_words)
                                            sf_stride = Uint32(self.sf_ring_tokens)
                                            if is_shared_l1:
                                                output_sf_words = Uint64(
                                                    self.shared_l2_sf_words)
                                                sf_stride = Uint32(
                                                    self.shared_sf_tokens)
                                            l2_sf_bytes = cute.recast_ptr(
                                                signals.iterator, dtype=Uint8) + \
                                                output_sf_words * Uint64(4)
                                            k_sf = n_block * Uint32(2) + \
                                                warp_in_wg // Uint32(2)
                                            token_base = wg_start_m + Uint32(
                                                store * self.store_m_l1 + atom * 8)
                                            assume_u32_lt(
                                                token_base, Uint32(self.block_m))
                                            sf_token = output_block * Uint32(self.sf_block_m) + \
                                                (token_base & Uint32(~127)) + \
                                                (token_base & Uint32(31)) * Uint32(4) + \
                                                ((token_base >> Uint32(5)) & Uint32(3)) + \
                                                Uint32(lane * 8)
                                            if cutlass.const_expr(
                                                    self.wide_routed_w1_sf or
                                                    self.wide_shared_w1_sf):
                                                sf_addr = Uint64(k_sf // Uint32(4)) * \
                                                    (Uint64(sf_stride) * Uint64(4)) + \
                                                    Uint64(sf_token) * Uint64(4) + \
                                                    Uint64(k_sf % Uint32(4))
                                                l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                l2_sf_bytes[sf_addr + Uint64(16)] = \
                                                    Uint8(sf_exp1)
                                            else:
                                                sf_addr = (k_sf // Uint32(4)) * \
                                                    (sf_stride * Uint32(4)) + \
                                                    sf_token * Uint32(4) + k_sf % Uint32(4)
                                                l2_sf_bytes[sf_addr] = Uint8(sf_exp0)
                                                l2_sf_bytes[sf_addr + Uint32(16)] = \
                                                    Uint8(sf_exp1)
                                        cute.arch.sync_warp()
                                    sync_epilogue_warpgroup(warpgroup)
                                    if warp_in_wg == Uint32(0):
                                        with cute.arch.elect_one():
                                            output_map = tensor_map_4.iterator
                                            if is_shared_l1:
                                                output_map = tensor_map_13.iterator
                                            tma_store_fence()
                                            tile_s2g_2d(
                                                output_map, cd_stage,
                                                n_block * Uint32(64),
                                                output_block * Uint32(self.block_m) +
                                                wg_start_m + Uint32(store * self.store_m_l1))
                                            tma_store_arrive()
                                    cute.arch.sync_warp()
                            tma_store_wait_0()
                        else:
                            # Source-contract W2 vector epilogue. The
                            # 16dp register-to-value bijection still
                            # requires validM>8 device verification.
                            is_shared_l2 = Uint32(0)
                            if cutlass.const_expr(self.shared_experts > 0):
                                is_shared_l2 = Uint32(
                                    task_phase == Uint32(GCBlockPhase.SharedLinear2))
                            epi_warp = Uint32(physical_warp - 4)
                            warpgroup = epi_warp // Uint32(4)
                            warp_in_wg = epi_warp % Uint32(4)
                            wg_start_m = warpgroup * Uint32(self.block_m // 2)
                            valid_wg_rows = Uint32(0)
                            if valid_m > wg_start_m:
                                valid_wg_rows = valid_m - wg_start_m
                                if valid_wg_rows > Uint32(self.block_m // 2):
                                    valid_wg_rows = Uint32(self.block_m // 2)
                            valid_stores = (valid_wg_rows + Uint32(self.store_m_l2 - 1)) // \
                                Uint32(self.store_m_l2)
                            n_block = n_cluster * Uint32(2) + cta_rank
                            cd_wg = cd_smem + warpgroup * Uint32(
                                self.store_m_l2 * 128 * 2)
                            raw_values = cute.make_rmem_tensor(
                                (2 if self.block_m == 240 else 1,
                                 self.store_m_l2 // 8, 8), Float32)
                            if valid_stores == Uint32(0):
                                _release_w2_tmem(accum_empty, accum_stage)
                            if cutlass.const_expr(self.block_m == 240):
                                if valid_stores > Uint32(0):
                                    base = accum_stage * Uint32(self.block_m) + \
                                        wg_start_m
                                    _load_w2_atom(raw_values, 0, 0, base)
                                    if valid_stores == Uint32(1):
                                        cute.arch.fence_view_async_tmem_load()
                                        _release_w2_tmem(accum_empty, accum_stage)
                            for store in cutlass.range_constexpr(
                                    (self.block_m // 2) // self.store_m_l2):
                                if Uint32(store) < valid_stores:
                                    if cutlass.const_expr(self.block_m != 240):
                                        for atom in cutlass.range_constexpr(
                                                self.store_m_l2 // 8):
                                            base = accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32(store * self.store_m_l2 + atom * 8)
                                            _load_w2_atom(raw_values, 0, atom, base)
                                            if atom == self.store_m_l2 // 8 - 1 and \
                                                    Uint32(store + 1) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage)
                                    # Wait for the current atom before
                                    # issuing the BM240 next-store prefetch.
                                    cute.arch.fence_view_async_tmem_load()
                                    if cutlass.const_expr(self.block_m == 240):
                                        if Uint32(store + 1) < valid_stores:
                                            next_base = accum_stage * Uint32(self.block_m) + \
                                                wg_start_m + Uint32((store + 1) * 8)
                                            _load_w2_atom(
                                                raw_values, (store + 1) % 2,
                                                0, next_base)
                                            if Uint32(store + 2) == valid_stores:
                                                cute.arch.fence_view_async_tmem_load()
                                                _release_w2_tmem(accum_empty, accum_stage)
                                    # One metadata record for each lane's
                                    # row in every 8-row atom.
                                    src_metadata = ()
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        token_in_block = wg_start_m + \
                                            Uint32(store * self.store_m_l2 + atom * 8) + \
                                            warp_in_wg * Uint32(2) + Uint32(lane // 16)
                                        pool_token = pool_block * Uint32(self.block_m) + \
                                            token_in_block
                                        record = _load_w2_source_metadata(
                                            signals.iterator + self.metadata_words,
                                            token_in_block, valid_m, pool_token,
                                            rank_idx, is_shared_l2, self.topk,
                                            self.shared_experts > 0)
                                        src_metadata += (record,)
                                    cute.arch.sync_warp()
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        if atom == 0 and store > 0:
                                            sync_epilogue_warpgroup(warpgroup)
                                        row = Uint32(lane % 8)
                                        col = (epi_warp % Uint32(2)) * Uint32(4) + \
                                            Uint32(lane // 8)
                                        smem_store_offset = (warp_in_wg // Uint32(2)) * \
                                            Uint32(self.store_m_l2 * 128) + \
                                            Uint32(atom * 8 * 128) + row * Uint32(128) + \
                                            (col ^ row) * Uint32(16)
                                        buffer = store % 2 if self.block_m == 240 else 0
                                        stmatrix_bf16x8_trans(
                                            cd_wg + smem_store_offset,
                                            pack_bf16_pair(raw_values[buffer, atom, 0],
                                                            raw_values[buffer, atom, 1]),
                                            pack_bf16_pair(raw_values[buffer, atom, 2],
                                                            raw_values[buffer, atom, 3]),
                                            pack_bf16_pair(raw_values[buffer, atom, 4],
                                                            raw_values[buffer, atom, 5]),
                                            pack_bf16_pair(raw_values[buffer, atom, 6],
                                                            raw_values[buffer, atom, 7]))
                                    sync_epilogue_warpgroup(warpgroup)
                                    for atom in cutlass.range_constexpr(
                                            self.store_m_l2 // 8):
                                        row_in_atom = warp_in_wg * Uint32(2) + \
                                            Uint32(lane // 16)
                                        row_in_store = Uint32(atom * 8) + row_in_atom
                                        token_in_block = wg_start_m + \
                                            Uint32(store * self.store_m_l2) + row_in_store
                                        if token_in_block < valid_m:
                                            bank_group = Uint32(lane % 8)
                                            half = (Uint32(lane) % Uint32(16)) // Uint32(8)
                                            smem_load_offset = half * \
                                                Uint32(self.store_m_l2 * 128) + \
                                                row_in_store * Uint32(128) + \
                                                (bank_group ^ row_in_atom) * Uint32(16)
                                            words = ld_shared_u32x4(
                                                cd_wg + smem_load_offset)
                                            token_bytes = Uint64(self.hidden * 2)
                                            slot_bytes = Uint64(self.capacity) * token_bytes
                                            channel_bytes = n_block * Uint32(256) + \
                                                (Uint32(lane) % Uint32(16)) * Uint32(16)
                                            dst_bytes = (Uint64(src_metadata[atom][2]) * slot_bytes +
                                                         Uint64(src_metadata[atom][1]) * token_bytes +
                                                         Uint64(channel_bytes))
                                            dst = cute.make_ptr(
                                                Uint32, mapped_peer_base(
                                                    peer_ptrs, src_metadata[atom][0],
                                                    self.num_ranks) +
                                                Uint64(self.combine_words * 4) + \
                                                dst_bytes,
                                                cute.AddressSpace.gmem,
                                                assumed_align=16)
                                            st_global_u32x4(
                                                dst, words[0], words[1],
                                                words[2], words[3])
                        sync_aligned(256, 2)
                    is_w1_release = task_phase == Uint32(GCBlockPhase.Linear1)
                    if cutlass.const_expr(self.shared_experts > 0):
                        is_w1_release = is_w1_release or \
                            task_phase == Uint32(GCBlockPhase.SharedLinear1)
                    if is_w1_release:
                        if physical_warp == 4:
                            if elect_one_sync():
                                ring_block = pool_block % self.ring_blocks
                                n_block = n_cluster * 2 + cta_rank
                                if task_phase == Uint32(GCBlockPhase.Linear1):
                                    red_release_gpu_xor_u64(
                                        signals.iterator + SIGNAL_OFFSETS['l2_full_mask'] // 4 +
                                        ring_block * 2,
                                        Uint64(1) << n_block)
                                    red_gpu_add_u32(
                                        signals.iterator + SIGNAL_OFFSETS['l1_empty_count'] // 4 +
                                        ring_block, Uint32(1))
                                else:
                                    if cutlass.const_expr(self.shared_experts > 0):
                                        if task_phase == Uint32(GCBlockPhase.SharedLinear1):
                                            red_release_gpu_add_u32(
                                                signals.iterator +
                                                SIGNAL_OFFSETS['shared_l2_full_count'] // 4 +
                                                pool_block, Uint32(1))
                        cute.arch.sync_warp()
                    running = task_phase != Uint32(GCBlockPhase.None_)
                    iteration = Uint32(iteration + Uint32(1))
                # The same full epilogue warp in both CTAs retires TMEM.
                if physical_warp == 4:
                    tmem_free_2cta(tptr[0], Uint32(self.tmem_cols))
            if cutlass.const_expr(self.eco_trace):
                if elect_one_sync():
                    self._eco_trace(signals, 16, 2)
                    self._eco_trace_ring(signals, 8, True)  # field 8: last role2 CTA exit
        else:
            # Both graph workers precede role 3. Fused finalization uses
            # warp-owned barriers; legacy paths retain the whole-CTA join.
            combine_smem = cute.recast_ptr(signals.iterator, dtype=Uint8)
            combine_bars = cute.recast_ptr(signals.iterator, dtype=Uint64)
            if cutlass.const_expr(self.finalize_output):
                # One fetch plane per combine slot; the output plane aliases
                # slot 0. Both live inside the role3 reusable prefix.
                combine_alloc = cutlass.utils.SmemAllocator()
                combine_smem = cute.recast_ptr(
                    combine_alloc.allocate(self.shared_layout.field_end, 1024), dtype=Uint8)
                combine_bars = cute.recast_ptr(
                    combine_smem + self.shared_layout.combine_bars, dtype=Uint64)
                if cutlass.const_expr(self.shared_experts > 0):
                    cute.arch.griddepcontrol_wait()
                else:
                    self._initialize_shared_storage(
                        combine_smem, Uint32(physical_warp), Uint32(lane))
            # Role 1's epoch remains immutable throughout this role-3 launch.
            # Each warp caches it independently; the footer advances recv_parity.
            ready_epoch = Uint64(ld_volatile_u32(
                signals.iterator + SIGNAL_OFFSETS['role3_epoch'] // 4))
            if cutlass.const_expr(not (self.finalize_output and self.shared_experts > 0)):
                sync_unaligned(FINALIZE_THREADS, 1)
            if cutlass.const_expr(self.eco_trace):
                if tid == 0:
                    self._eco_trace(signals, 14, 1)
                    if block == 0:
                        self._eco_trace_accum(signals, 22)
                        # role3 starts once every role2 CTA has exited: this rank's routed GEMM completion
                        self._eco_trace_ring(signals, 1, False)
            recv_off = SIGNAL_OFFSETS['expert_recv_count_sum'] // 4
            recv_ptr = signals.iterator + recv_off
            if physical_warp < 4:
                self._role3_dispatch(signals, peer_ptrs, rank_idx,
                                     Uint32(tid), Uint32(lane), Uint32(block),
                                     recv_ptr, stats, ready_epoch, eco_fixed_slots)
            else:
                self._role3_combine(signals, rank_idx, num_tokens, y_out,
                                    block, physical_warp, lane,
                                    combine_smem, combine_bars, ready_epoch)
            if cutlass.const_expr(self.eco_trace):
                if elect_one_sync():
                    self._eco_trace(signals, 15, 2)
                    self._eco_trace_ring(signals, 9, True)  # field 9: last role3 warp done
            if cutlass.const_expr(self.eco_trace):
                if block == 0 and tid == 0:
                    self._eco_trace_accum(signals, 23)

    @cute.jit
    def __call__(self, signals: cute.Tensor, peer_ptrs: cute.Tensor,
                 rank_idx: Uint32, num_tokens: Uint32,
                 tensor_maps=None, y_out=None, stats=None,
                 activation_clamp=None, activation_alpha=None,
                 tensor_map_1=None, tensor_map_2=None, tensor_map_3=None,
                 tensor_map_4=None, tensor_map_5=None, tensor_map_6=None,
                 tensor_map_7=None, tensor_map_8=None, tensor_map_9=None,
                 tensor_map_10=None, tensor_map_11=None, tensor_map_12=None,
                 tensor_map_13=None, tensor_map_14=None, tensor_map_15=None,
                 tensor_map_16=None, tensor_map_17=None,
                 tensor_map_18=None, tensor_map_19=None,
                 tensor_map_20=None, tensor_map_21=None, eco_fixed_slots=None):
        if cutlass.const_expr(self.role == 1):
            if cutlass.const_expr(self.eco):
                # ECO: the layer's EcoMeta (int32) rides in the role's unused `tensor_maps` formal
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                            tensor_maps, None, None,
                            activation_clamp, activation_alpha).launch(
                    grid=(self.dispatch_sms, 1, 1),
                    block=(self.dispatch_warps * 32, 1, 1),
                    max_number_threads=(self.dispatch_warps * 32, 1, 1),
                    min_blocks_per_mp=1)
            elif cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens).launch(
                    grid=(self.dispatch_sms, 1, 1),
                    block=(self.dispatch_warps * 32, 1, 1),
                    max_number_threads=(self.dispatch_warps * 32, 1, 1),
                    min_blocks_per_mp=1)
            else:
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                            None, None, None,
                            activation_clamp, activation_alpha).launch(
                    grid=(self.dispatch_sms, 1, 1),
                    block=(self.dispatch_warps * 32, 1, 1),
                    max_number_threads=(self.dispatch_warps * 32, 1, 1),
                    min_blocks_per_mp=1)
        elif cutlass.const_expr(self.role == 2):
            if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                            tensor_maps, None, None, None, None,
                            tensor_map_1, tensor_map_2, tensor_map_3,
                            tensor_map_4, tensor_map_5, tensor_map_6,
                            tensor_map_7, tensor_map_8, tensor_map_9,
                            tensor_map_10, tensor_map_11, tensor_map_12,
                            tensor_map_13, tensor_map_14, tensor_map_15,
                            tensor_map_16, tensor_map_17,
                            tensor_map_18, tensor_map_19,
                            tensor_map_20, tensor_map_21, eco_fixed_slots).launch(
                    grid=(self.compute_sms, 1, 1), block=(384, 1, 1),
                    max_number_threads=(384, 1, 1),
                    cluster=(2, 1, 1), min_blocks_per_mp=1)
            else:
                self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                            None, None,
                            activation_clamp, activation_alpha,
                            tensor_map_1, tensor_map_2, tensor_map_3,
                            tensor_map_4, tensor_map_5, tensor_map_6,
                            tensor_map_7, tensor_map_8, tensor_map_9,
                            tensor_map_10, tensor_map_11, tensor_map_12,
                            tensor_map_13, tensor_map_14, tensor_map_15,
                            tensor_map_16, tensor_map_17,
                            tensor_map_18, tensor_map_19,
                            tensor_map_20, tensor_map_21, eco_fixed_slots).launch(
                    grid=(self.compute_sms, 1, 1), block=(384, 1, 1),
                    max_number_threads=(384, 1, 1),
                    cluster=(2, 1, 1), min_blocks_per_mp=1)
        else:
            if cutlass.const_expr(self.finalize_output):
                if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                                y_out, stats, eco_fixed_slots=eco_fixed_slots).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
                else:
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens, tensor_maps,
                                y_out, stats,
                                activation_clamp, activation_alpha, eco_fixed_slots=eco_fixed_slots).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
            else:
                if cutlass.const_expr(activation_clamp is None and activation_alpha is None):
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                                eco_fixed_slots=eco_fixed_slots).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)
                else:
                    self.kernel(signals, peer_ptrs, rank_idx, num_tokens,
                                None, None, None,
                                activation_clamp, activation_alpha, eco_fixed_slots=eco_fixed_slots).launch(
                        grid=(self.finalize_sms, 1, 1),
                        block=(FINALIZE_THREADS, 1, 1),
                        max_number_threads=(FINALIZE_THREADS, 1, 1), min_blocks_per_mp=1)

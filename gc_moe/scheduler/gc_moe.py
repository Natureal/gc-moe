"""Two-CTA GC task handoff and the four task phase identifiers.

The 32-byte task lives in each CTA's SMEM. The scheduler warp in the leader
CTA publishes both copies with transaction-counted DSM stores; the two
epilogue warpgroups per CTA retire a slot through the leader empty barrier.
"""

from __future__ import annotations

from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Uint64

from ..common.math import warp_inclusive_sum
from ..layout.gc_moe import (ECO_SIGNAL_OFFSETS, ECO_TRACE_RING_FIELDS, MAX_EXPERTS,
                             SIGNAL_OFFSETS)
from ..ptx.ld_st import (
    atom_add_u64,
    atom_max_u32_global,
    ld_acquire_gpu_u32,
    ld_volatile_u32,
    red_global_max_u64,
    red_global_min_u64,
    red_gpu_add_u32,
    ld_volatile_u64,
    warp_wait_count_u64,
    st_global_u32,
    atom_add_u32,
    st_async_cluster_u32x4,
)
from ..ptx.utils import (
    elect_one_sync,
    read_globaltimer,
    warp_first_set_bit,
)


class GCBlockPhase(IntEnum):
    None_ = 0
    Linear1 = 1
    Linear2 = 2
    SharedLinear1 = 3
    SharedLinear2 = 4


TASK_WORDS = 8
TASK_BYTES = 32
SCHEDULE_STAGES = 2
TASK_EMPTY_CONSUMERS = 512


@cute.jit
def publish_task32(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                   stage: Int32, peer_cta: Int32,
                   block_phase: Uint32, local_expert: Uint32,
                   m_block: Uint32, n_cluster: Uint32,
                   pool_block: Uint32, valid_m: Uint32,
                   shape_n: Uint32, shape_k: Uint32) -> None:
    cute.arch.mbarrier_arrive_and_expect_tx(full_ptr + stage, TASK_BYTES, peer_cta)
    base = stage * TASK_WORDS
    st_async_cluster_u32x4(task_ptr, full_ptr + stage, peer_cta, base,
                  block_phase, local_expert, m_block, n_cluster)
    st_async_cluster_u32x4(task_ptr, full_ptr + stage, peer_cta, base + 4,
                  pool_block, valid_m, shape_n, shape_k)


@cute.jit
def release_task_info(empty_ptr: cute.Pointer, stage: Int32) -> None:
    cute.arch.fence_acq_rel_cta()
    cute.arch.mbarrier_arrive(empty_ptr + stage, peer_cta_rank_in_cluster=0)


@cute.jit
def claim_task(task_count_ptr: cute.Pointer, lane: Int32) -> Uint32:
    task_idx = Uint32(0)
    if elect_one_sync():
        task_idx = atom_add_u32(task_count_ptr, Uint32(1))
    return cute.arch.shuffle_sync(task_idx, 0)


@cute.jit
def wait_task_slot(empty_ptr: cute.Pointer, iteration: Uint32) -> None:
    """Wait for the next task slot before entering its claim loop."""
    stage = Int32(iteration % 2)
    parity = Int32((iteration // 2) % 2)
    cute.arch.mbarrier_wait(empty_ptr + stage, parity ^ 1)


@cute.jit
def publish_iteration(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                      iteration: Uint32,
                      lane: Int32, block_phase: Uint32,
                      local_expert: Uint32,
                      m_block: Uint32, n_cluster: Uint32,
                      pool_block: Uint32, valid_m: Uint32,
                      shape_n: Uint32, shape_k: Uint32) -> None:
    stage = Int32(iteration % 2)
    if lane < 2:
        publish_task32(task_ptr, full_ptr, stage, lane,
                       block_phase, local_expert, m_block, n_cluster,
                       pool_block, valid_m, shape_n, shape_k)
    cute.arch.sync_warp()


@cute.jit
def publish_shared_phase(task_ptr: cute.Pointer, full_ptr: cute.Pointer,
                         empty_ptr: cute.Pointer, counter_ptr: cute.Pointer,
                         iteration: Uint32, lane: Int32,
                         cluster_idx: Uint32,
                         phase: Uint32, num_tokens: Uint32,
                         block_m: cutlass.Constexpr,
                         n_clusters: cutlass.Constexpr,
                         num_compute_sms: cutlass.Constexpr,
                         shape_n: cutlass.Constexpr,
                         shape_k: cutlass.Constexpr) -> Uint32:
    """Reference shared_mainloop: static one task/cluster, else global claims."""
    m_blocks = (num_tokens + Uint32(block_m - 1)) // Uint32(block_m)
    total = m_blocks * Uint32(n_clusters)
    if total <= Uint32(num_compute_sms // 2):
        wait_task_slot(empty_ptr, iteration)
        if cluster_idx < Uint32(total):
            m_block = cluster_idx // Uint32(n_clusters)
            n_cluster = cluster_idx % Uint32(n_clusters)
            valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
            publish_iteration(task_ptr, full_ptr, iteration,
                              lane, phase, Uint32(0), m_block, n_cluster, m_block,
                              valid_m, Uint32(shape_n), Uint32(shape_k))
            iteration += Uint32(1)
            # Match the reference's wait before the static path's failed claim.
            wait_task_slot(empty_ptr, iteration)
    else:
        scheduling = True
        while scheduling:
            wait_task_slot(empty_ptr, iteration)
            task_idx = claim_task(counter_ptr, lane)
            if task_idx >= Uint32(total):
                scheduling = False
            else:
                m_block = task_idx // Uint32(n_clusters)
                n_cluster = task_idx % Uint32(n_clusters)
                valid_m = min(Uint32(block_m), Uint32(num_tokens) - m_block * Uint32(block_m))
                publish_iteration(task_ptr, full_ptr, iteration,
                                  lane, phase, Uint32(0), m_block, n_cluster, m_block,
                                  valid_m, Uint32(shape_n), Uint32(shape_k))
                iteration += Uint32(1)
    return iteration


@cute.jit
def locate_routed_pool_block(counts: cute.Tensor,
                             pool_block: Uint32, lane: Uint32,
                             local_experts: cutlass.Constexpr,
                             block_m: cutlass.Constexpr):
    """Reference GCMoEScheduler::create_task from per-lane count registers."""
    expert = Uint32(0)
    m_block = Uint32(0)
    valid_m = Uint32(0)
    block_offset = Uint32(0)
    for wave in cutlass.range_constexpr((local_experts + 31) // 32):
        expert_idx = Uint32(wave * 32) + lane
        received = counts[wave]
        blocks = (received + Uint32(block_m - 1)) // Uint32(block_m)
        inclusive = warp_inclusive_sum(blocks, lane)
        lane_offset = block_offset + inclusive - blocks
        is_owner = expert_idx < Uint32(local_experts) and \
            pool_block >= lane_offset and pool_block < lane_offset + blocks
        owner_mask = Uint32(cute.arch.vote_ballot_sync(is_owner))
        if owner_mask != Uint32(0):
            owner_lane = warp_first_set_bit(owner_mask)
            local_m = pool_block - lane_offset
            local_valid = min(Uint32(block_m),
                              received - local_m * Uint32(block_m))
            expert = cute.arch.shuffle_sync(expert_idx, owner_lane)
            m_block = cute.arch.shuffle_sync(local_m, owner_lane)
            valid_m = cute.arch.shuffle_sync(local_valid, owner_lane)
        block_offset += cute.arch.shuffle_sync(inclusive, 31)
    return expert, m_block, valid_m


@cute.jit
def cached_num_tokens(counts: cute.Tensor, expert: Int32,
                      lane: Uint32,
                      local_experts: cutlass.Constexpr) -> Uint32:
    """GCMoEScheduler::get_num_tokens from expert wave*32+lane registers."""
    selected = Uint32(0)
    for wave in cutlass.range_constexpr((local_experts + 31) // 32):
        if Uint32(expert) == Uint32(wave * 32) + lane:
            selected = counts[wave]
    return cute.arch.shuffle_sync(selected, Uint32(expert) % Uint32(32))


@cute.jit
def advance_expert(counts: cute.Tensor, token_idx: Uint32,
                   current_expert: Int32, expert_start: Uint32,
                   expert_end: Uint32, pool_block_offset: Uint32,
                   lane: Uint32, local_experts: cutlass.Constexpr,
                   block_m: cutlass.Constexpr):
    """Advance the unpadded token interval and its padded pool prefix."""
    while token_idx >= expert_end and current_expert < Int32(local_experts):
        current_expert += Int32(1)
        if current_expert < Int32(local_experts):
            pool_block_offset += (expert_end - expert_start +
                                  Uint32(block_m - 1)) // Uint32(block_m)
            expert_start = expert_end
            expert_end += cached_num_tokens(
                counts, current_expert, lane, local_experts)
    return current_expert, expert_start, expert_end, pool_block_offset


@cute.jit
def eco_trace_point(signals: cute.Pointer, slot: cutlass.Constexpr, is_max: cutlass.Constexpr,
                    enabled: cutlass.Constexpr, eco_base: cutlass.Constexpr) -> None:
    """ECO timeline trace (elected lane): min or max of globaltimer into signals 'eco_trace'[slot]"""
    if cutlass.const_expr(enabled):
        if elect_one_sync():
            ptr = signals + ((eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) + slot * 8) // 4
            if cutlass.const_expr(is_max):
                red_global_max_u64(ptr, read_globaltimer())
            else:
                red_global_min_u64(ptr, read_globaltimer())


@cute.jit
def eco_trace_ring_point(signals: cute.Pointer, field: cutlass.Constexpr, block: Int32,
                         enabled: cutlass.Constexpr, eco_base: cutlass.Constexpr) -> None:
    """ECO per-call trace ring (block 0's elected lane): ns since this call's role1 start into `field`"""
    if cutlass.const_expr(enabled):
        if block == Int32(0):
            if elect_one_sync():
                base = signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4
                elapsed = Uint32(read_globaltimer() - ld_volatile_u64(base + 24 * 2))
                entry = Uint32(ld_volatile_u64(base + 33 * 2)) % Uint32(64)
                st_global_u32(signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_trace_ring']) // 4 + entry * Uint32(ECO_TRACE_RING_FIELDS) + field, elapsed)


@cute.jit
def eco_wait_plan(signals: cute.Pointer, epoch: Uint32, trace: cutlass.Constexpr,
                  eco_base: cutlass.Constexpr, dispatch: cutlass.Constexpr) -> None:
    """Acquire the plan; trace first-read misses separately from trace/acquire overhead."""
    ready = signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_plan_ready']) // 4
    if cutlass.const_expr(trace):
        begin = read_globaltimer()
        observed = ld_acquire_gpu_u32(ready)
        blocked = Uint32(observed != epoch + Uint32(1))
        while observed != epoch + Uint32(1):
            observed = ld_acquire_gpu_u32(ready)
        duration = Uint32(read_globaltimer() - begin)
        blocked = cute.arch.warp_redux_sync(blocked, 'max')
        duration = cute.arch.warp_redux_sync(duration, 'max')
        if elect_one_sync():
            base = signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_trace']) // 4
            entry = Uint32(ld_volatile_u64(base + 33 * 2)) % Uint32(64)
            ring = signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_trace_ring']) // 4 + \
                entry * Uint32(ECO_TRACE_RING_FIELDS)
            if cutlass.const_expr(dispatch):
                red_global_min_u64(base + 6 * 2, begin)
                # Invert minima so zero-initialized ring entries support a max reduction.
                atom_max_u32_global(ring + 8, Uint32(0xffffffff) -
                                    Uint32(begin - ld_volatile_u64(base + 24 * 2)))
                atom_add_u64(base + 27 * 2, Uint64(duration))
                atom_add_u64(base + 29 * 2, Uint64(1))
                if blocked != Uint32(0):
                    red_gpu_add_u32(ring + 9, Uint32(1))
                    atom_max_u32_global(ring + 10, duration)
            else:
                red_global_min_u64(base + 11 * 2, begin)
                atom_add_u64(base + 26 * 2, Uint64(duration))
                atom_add_u64(base + 28 * 2, Uint64(1))
                if blocked != Uint32(0):
                    red_gpu_add_u32(ring + 11, Uint32(1))
                    atom_max_u32_global(ring + 12, duration)
    else:
        while ld_acquire_gpu_u32(ready) != epoch + Uint32(1):
            pass


@cute.jit
def eco_load_replicas(signals: cute.Pointer, rank_idx: Uint32, epoch: Uint32,
                      num_fixed: Uint32, expert_counts: cute.Tensor, lane: Int32,
                      local_experts: cutlass.Constexpr,
                      block_m: cutlass.Constexpr, eco_base: cutlass.Constexpr,
                      num_ranks: cutlass.Constexpr = 1,
                      trace: cutlass.Constexpr = False, block: Int32 = None,
                      spec: cutlass.Constexpr = False) -> Uint32:
    """ECO: wait for this call's plan, cache the replica slots' quotas, return the routed M-block total"""
    if cutlass.const_expr(not spec):
        eco_wait_plan(signals, epoch, trace, eco_base, False)
    eco_trace_point(signals, 12, True, trace, eco_base)
    qbank = epoch & Uint32(1)
    if cutlass.const_expr(spec):
        qbank = (epoch + Uint32(1)) & Uint32(1)
    lane_m_blocks = Uint32(0)
    for wave in cutlass.range_constexpr((local_experts + 31) // 32):
        slot_idx = Uint32(wave * 32 + lane)
        if slot_idx >= num_fixed and slot_idx < Uint32(local_experts):
            expert_counts[wave] = ld_volatile_u32(
                signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_quota']) // 4 +
                qbank * Uint32(num_ranks * local_experts
                               if 2 * num_ranks * local_experts <= MAX_EXPERTS else 0) +
                rank_idx * Uint32(local_experts) + slot_idx)
        lane_m_blocks += (expert_counts[wave] + Uint32(block_m - 1)) // Uint32(block_m)
    return cute.arch.warp_redux_sync(lane_m_blocks, 'add')


@cute.jit
def schedule_tasks(task: cute.Pointer, full: cute.Pointer, empty: cute.Pointer,
                   recv_ptr: cute.Pointer, l1_count_ptr: cute.Pointer,
                   l2_count_ptr: cute.Pointer, shared_l1_count_ptr: cute.Pointer,
                   shared_l2_count_ptr: cute.Pointer, expert_counts: cute.Tensor,
                   iteration: Uint32, lane: Int32, block: Int32, num_tokens: Uint32,
                   block_m: cutlass.Constexpr,
                   compute_sms: cutlass.Constexpr,
                   dispatch_sms: cutlass.Constexpr,
                   hidden: cutlass.Constexpr,
                   intermediate: cutlass.Constexpr,
                   l1_clusters: cutlass.Constexpr,
                   l2_clusters: cutlass.Constexpr,
                   local_experts: cutlass.Constexpr,
                   num_ranks: cutlass.Constexpr,
                   shared_experts: cutlass.Constexpr,
                   eco: cutlass.Constexpr = False,
                   signals: cute.Pointer = None,
                   rank_idx: Uint32 = None,
                   eco_trace: cutlass.Constexpr = False,
                   eco_base: cutlass.Constexpr = 0,
                   eco_fixed_slots: Uint32 = None,
                   eco_spec: cutlass.Constexpr = False,
) -> None:
    """Publish shared and interleaved routed tasks, then the end sentinel.

    ECO: the single-copy slots (native receive counts, independent of the plan) are scheduled first; a claim beyond
    them waits for the plan role and extends the routed total with the replica slots' quotas."""
    shared_m_blocks = (num_tokens + Uint32(block_m - 1)) // \
        Uint32(block_m)
    shared_l2_early = (shared_m_blocks * Uint32(l2_clusters) <=
                       Uint32(compute_sms))
    if cutlass.const_expr(shared_experts > 0):
        shared_l1_clusters = l1_clusters * shared_experts
        iteration = publish_shared_phase(
            task, full, empty, shared_l1_count_ptr, iteration,
            Int32(lane), Uint32(block // 2),
            Uint32(GCBlockPhase.SharedLinear1),
            num_tokens, block_m, shared_l1_clusters,
            compute_sms, 2 * intermediate * shared_experts,
            hidden)
        if shared_l2_early:
            iteration = publish_shared_phase(
                task, full, empty, shared_l2_count_ptr, iteration,
                Int32(lane), Uint32(block // 2),
                Uint32(GCBlockPhase.SharedLinear2),
                num_tokens, block_m, l2_clusters,
                compute_sms, hidden,
                intermediate * shared_experts)
    # Each lane owns experts wave*32+lane in registers, as in
    # GCMoEScheduler. The count cache survives routed L1/L2 task
    # claims, without an E-dependent shared prefix allocation.
    eco_epoch = Uint32(0)
    eco_num_fixed = Uint32(0)
    eco_plan_loaded = Uint32(0)
    if cutlass.const_expr(eco):
        eco_epoch = ld_volatile_u32(signals + SIGNAL_OFFSETS['recv_parity'] // 4)
        eco_num_fixed = eco_fixed_slots
        for wave in cutlass.range_constexpr(
                (local_experts + 31) // 32):
            slot_idx = Uint32(wave * 32 + lane)
            expert_counts[wave] = warp_wait_count_u64(
                recv_ptr + slot_idx * Uint32(2), Uint32(1) if slot_idx < eco_num_fixed else Uint32(0),
                dispatch_sms * num_ranks)
        eco_trace_point(signals, 10, False, eco_trace, eco_base)
        eco_trace_ring_point(signals, 4, block, eco_trace, eco_base)
    else:
        for wave in cutlass.range_constexpr(
                (local_experts + 31) // 32):
            local_expert = Uint32(wave * 32 + lane)
            count = Uint64(0)
            if local_expert < Uint32(local_experts):
                while Uint32(count >> 32) != Uint32(
                        dispatch_sms * num_ranks):
                    count = ld_volatile_u64(recv_ptr + local_expert * 2)
            expert_counts[wave] = Uint32(count)
        eco_trace_ring_point(signals, 4, block, eco_trace, eco_base)
    cute.arch.sync_warp()
    lane_m_blocks = Uint32(0)
    for wave in cutlass.range_constexpr(
            (local_experts + 31) // 32):
        received = expert_counts[wave]
        own_blocks = (received + Uint32(block_m - 1)) // \
            Uint32(block_m)
        lane_m_blocks += own_blocks
    routed_m_blocks = cute.arch.warp_redux_sync(
        lane_m_blocks, 'add')
    # Match GCMoEScheduler::get_gc_num_l1_warmup_waves and
    # get_next_task for one locality domain: enough L1 work is
    # issued before the first L2 wave, then one L1 follows each
    # successful L2 claim until the L1 queue is exhausted.
    num_clusters = compute_sms // 2
    first_l2_m_blocks = (num_clusters + l2_clusters - 1) // l2_clusters
    first_l2_warmup = (first_l2_m_blocks * l1_clusters +
                       num_clusters - 1) // num_clusters
    interleave_diff = max(l1_clusters - l2_clusters, 0)
    l1_warmup = Uint32(0)
    if routed_m_blocks > 0:
        interleave_warmup = (l1_clusters +
                             (routed_m_blocks - 1) * interleave_diff +
                             num_clusters - 1) // num_clusters + 1
        total_l1_waves = (routed_m_blocks * l1_clusters +
                          num_clusters - 1) // num_clusters
        l1_warmup = min(total_l1_waves,
                        max(Uint32(first_l2_warmup), interleave_warmup))
    if cutlass.const_expr(eco):
        # An empty fixed prefix must claim L1 first, using the existing quota-loading path below. Starting
        # with L2 can make every scheduler wait on L1 tasks nobody has issued. Avoid another inlined copy of
        # eco_load_replicas here: it increases scheduler register pressure and spills under the 40-reg cap.
        if routed_m_blocks == Uint32(0):
            l1_warmup = Uint32(first_l2_warmup)
    l1_done = Uint32(0xffffffff)
    # Typed before the loop for both count-loading paths (CuTe DSL joins need one type)
    local_expert = Uint32(0)
    scheduling = True
    while scheduling:
        wait_task_slot(empty, iteration)
        fetching = True
        while fetching:
            if l1_warmup != l1_done and l1_warmup > 0:
                l1_warmup -= Uint32(1)
                task_idx = claim_task(l1_count_ptr, Int32(lane))
                if cutlass.const_expr(eco):
                    if eco_plan_loaded == Uint32(0) and task_idx >= routed_m_blocks * l1_clusters:
                        routed_m_blocks = eco_load_replicas(
                            signals, rank_idx, eco_epoch, eco_num_fixed, expert_counts, Int32(lane),
                            local_experts, block_m, eco_base, num_ranks, eco_trace, block, eco_spec)
                        eco_plan_loaded = Uint32(1)
                if task_idx >= routed_m_blocks * l1_clusters:
                    l1_warmup = l1_done
                else:
                    pool_block = task_idx // l1_clusters
                    n_cluster = task_idx % l1_clusters
                    local_expert, m_block, valid_m = \
                        locate_routed_pool_block(
                            expert_counts, pool_block, Uint32(lane),
                            local_experts, block_m)
                    publish_iteration(task, full, iteration, Int32(lane),
                                      Uint32(GCBlockPhase.Linear1), local_expert,
                                      m_block, n_cluster, pool_block, valid_m,
                                      Uint32(2 * intermediate), Uint32(hidden))
                    iteration = Uint32(iteration + Uint32(1))
                    fetching = False
                    if cutlass.const_expr(eco):
                        # ECO: take the plan as soon as it is published (non-blocking poll after a publish, while
                        # the consumers are busy), not at the claim that first needs a replica slot
                        if eco_plan_loaded == Uint32(0):
                            if cutlass.const_expr(eco_spec) or \
                                    ld_volatile_u32(signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_plan_ready']) // 4) == \
                                    eco_epoch + Uint32(1):
                                routed_m_blocks = eco_load_replicas(
                                    signals, rank_idx, eco_epoch, eco_num_fixed, expert_counts, Int32(lane),
                                    local_experts, block_m, eco_base, num_ranks, eco_trace, block, eco_spec)
                                eco_plan_loaded = Uint32(1)
            else:
                task_idx = claim_task(l2_count_ptr, Int32(lane))
                if cutlass.const_expr(eco):
                    if eco_plan_loaded == Uint32(0) and task_idx >= routed_m_blocks * l2_clusters:
                        routed_m_blocks = eco_load_replicas(
                            signals, rank_idx, eco_epoch, eco_num_fixed, expert_counts, Int32(lane),
                            local_experts, block_m, eco_base, num_ranks, eco_trace, block, eco_spec)
                        eco_plan_loaded = Uint32(1)
                if task_idx >= routed_m_blocks * l2_clusters:
                    scheduling = False
                    fetching = False
                else:
                    if l1_warmup != l1_done:
                        l1_warmup = Uint32(1)
                    pool_block = task_idx // l2_clusters
                    n_cluster = task_idx % l2_clusters
                    local_expert, m_block, valid_m = \
                        locate_routed_pool_block(
                            expert_counts, pool_block, Uint32(lane),
                            local_experts, block_m)
                    required_l1 = (pool_block + 1) * l1_clusters
                    while ld_volatile_u32(l1_count_ptr) < required_l1:
                        pass
                    publish_iteration(task, full, iteration, Int32(lane),
                                      Uint32(GCBlockPhase.Linear2), local_expert,
                                      m_block, n_cluster, pool_block, valid_m,
                                      Uint32(hidden), Uint32(intermediate))
                    iteration = Uint32(iteration + Uint32(1))
                    fetching = False
                    if cutlass.const_expr(eco):
                        # ECO: take the plan as soon as it is published (non-blocking poll after a publish, while
                        # the consumers are busy), not at the claim that first needs a replica slot
                        if eco_plan_loaded == Uint32(0):
                            if cutlass.const_expr(eco_spec) or \
                                    ld_volatile_u32(signals + (eco_base + ECO_SIGNAL_OFFSETS['eco_plan_ready']) // 4) == \
                                    eco_epoch + Uint32(1):
                                routed_m_blocks = eco_load_replicas(
                                    signals, rank_idx, eco_epoch, eco_num_fixed, expert_counts, Int32(lane),
                                    local_experts, block_m, eco_base, num_ranks, eco_trace, block, eco_spec)
                                eco_plan_loaded = Uint32(1)
    if cutlass.const_expr(shared_experts > 0):
        if not shared_l2_early:
            iteration = publish_shared_phase(
                task, full, empty, shared_l2_count_ptr, iteration,
                Int32(lane), Uint32(block // 2),
                Uint32(GCBlockPhase.SharedLinear2),
                num_tokens, block_m, l2_clusters,
                compute_sms, hidden,
                intermediate * shared_experts)
    wait_task_slot(empty, iteration)
    publish_iteration(task, full, iteration, Int32(lane),
                      Uint32(GCBlockPhase.None_), Uint32(0),
                      Uint32(0), Uint32(0),
                      Uint32(0), Uint32(0), Uint32(0), Uint32(0))

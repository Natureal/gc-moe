#pragma once

#include <cstdint>

#include "eco_compat.cuh"

namespace deep_gemm::layout {

// ECO (EPLB-placed Capped Online split) metadata: int32, one table per MoE layer, built by the host
// (`rtp_llm/models_py/modules/eco.py`). The placement is the official EPLB one; every rank's local slots are
// `[single-copy experts | replica copies]`. Layout:
//   header[kHeaderSize]: magic, E, P, R, kMaxCopies, cost coefficients (active, blocks, tokens), #placement rounds,
//                        #relief rounds, active cap extra, plan block M, kMaxRounds, reserved
//   copies[E][kMaxCopies]   : int16 physical slots of every logical expert ascending (-1 padded); bit 15 marks a
//                             passive copy (a lower copy of the expert sits on the same rank), two per int32
//   phy2log[P]              : logical expert of each physical slot
//   num_fixed[R]            : single-copy slots of each rank (a prefix of its local slots)
//   rounds[kMaxRounds][32]  : placement round words of lane `l` (ranks `l` and `l + 32`): two 11-bit slots (0x7ff:
//                             none) and the lowest lane planning the same expert in bits 22-26 (idle lanes: own);
//                             within a round no two experts share a rank
//   pairs[kMaxPairsPerSlot * P]: `p << 16 | c` for every ordered pair of non-passive copies (p, c) of a replicated
//                             expert on distinct ranks, 0xffffffff padded; count in the header
//   unit section (int32 offset in header[15]): the plan role's static tables (`gc_moe/eco/unit_ref.py`,
//                             `unit_section`). The plan role reads only copies, phy2log, num_fixed and this section.
struct EcoMeta {
    static constexpr int kMagic = 0x45434F31;
    static constexpr uint32_t kHeaderSize = 16;
    static constexpr uint32_t kMaxCopies = 32;
    static constexpr uint32_t kMaxRounds = 64;
    static constexpr uint32_t kNumLanes = 32;
    static constexpr uint32_t kMaxPairsPerSlot = 4;
    static constexpr int16_t kPassiveBit = static_cast<int16_t>(0x8000);

    const int* base;
    uint32_t num_logical_experts, num_physical_experts, num_ranks;

    EcoMeta() = default;

    CUTLASS_HOST_DEVICE EcoMeta(const int* base, const uint32_t& num_logical_experts,
                                const uint32_t& num_physical_experts, const uint32_t& num_ranks):
        base(base), num_logical_experts(num_logical_experts),
        num_physical_experts(num_physical_experts), num_ranks(num_ranks) {}

    CUTLASS_HOST_DEVICE static uint32_t get_num_ints(const uint32_t& num_logical_experts,
                                                     const uint32_t& num_physical_experts,
                                                     const uint32_t& num_ranks) {
        return kHeaderSize + num_logical_experts * kMaxCopies / 2 + num_physical_experts + num_ranks +
               kMaxRounds * kNumLanes + kMaxPairsPerSlot * num_physical_experts;
    }

    CUTLASS_HOST_DEVICE uint32_t get_copies_offset() const { return kHeaderSize; }
    CUTLASS_HOST_DEVICE uint32_t get_phy2log_offset() const { return kHeaderSize + num_logical_experts * kMaxCopies / 2; }
    CUTLASS_HOST_DEVICE uint32_t get_num_fixed_offset() const { return get_phy2log_offset() + num_physical_experts; }
    CUTLASS_HOST_DEVICE uint32_t get_rounds_offset() const { return get_num_fixed_offset() + num_ranks; }
    CUTLASS_HOST_DEVICE uint32_t get_pairs_offset() const { return get_rounds_offset() + kMaxRounds * kNumLanes; }

#if defined(__CUDACC__) or defined(__CLION_IDE__)
    CUTLASS_DEVICE int get_header(const uint32_t& idx) const { return __ldg(base + idx); }

    CUTLASS_DEVICE int get_phy2log(const uint32_t& physical_idx) const {
        return __ldg(base + get_phy2log_offset() + physical_idx);
    }

    CUTLASS_DEVICE uint32_t get_num_fixed(const uint32_t& rank_idx) const {
        return static_cast<uint32_t>(__ldg(base + get_num_fixed_offset() + rank_idx));
    }
#endif
};

} // namespace deep_gemm::layout

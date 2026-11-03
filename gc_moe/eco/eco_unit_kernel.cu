// ECO plan role of GC MoE: one CTA per rank computes the identical split plan of every rank while the dispatch role
// pulls the single-copy experts' tokens and the compute role runs their GEMMs. Bit-exact mirror of
// `gc_moe/eco/unit_ref.py`.
//
// Units: an expert's n tokens are ceil(n / bm) units (the last one partial); rank time = wb * blocks + wa * active
// slots + tokens (wb = wa = 0: tokens alone). A copy is passive (never planned) when a lower copy of the same expert
// sits on the same rank. (1) Start: warm, the previous plan's units per copy (`state`: its quotas and counts), only the
// experts whose unit count changed add / remove units; cold, an even split rounded to units. (2) Auction rounds
// (`eco_unit_auction.cuh`): sources above the mean offer unit moves, receivers accept a prefix up to the mean + one
// unit; the first UNIT_HOP2 rounds also offer two-hop paths.
//
// Compile-time configuration (nvcc -D): ECO_P physical slots, ECO_R ranks, ECO_E logical experts, ECO_THREADS (576),
// the signal field offsets (OFF_*), UNIT_BM (block M), UNIT_WB / UNIT_WA (block / active-slot weights, tokens 1),
// UNIT_NR / UNIT_NRC (warm / cold auction rounds), UNIT_HOP2 (two-hop rounds).
//
// Warp roles: warps 0..15 plan (named barrier 1, 512 threads); warps 1..15 first gather the count table (barrier 2,
// 480 threads); warp 0 and the two index warps 16, 17 stage the static tables (barrier 4, 96 threads), then the index
// warps write the replicated experts' source indices into every active copy's rank (barrier 3, 64 threads).

#include <cstdint>

#include "include/eco_meta.cuh"

using namespace deep_gemm;

static constexpr uint32_t kP = ECO_P, kR = ECO_R, kE = ECO_E, kS = kP / kR, kThreads = ECO_THREADS;
static constexpr uint32_t kPlanWarps = 16, kPlanThreads = 32 * kPlanWarps, kIndexWarps = 2;
static constexpr uint32_t kFirstIndexWarp = kPlanWarps, kHelpers = kPlanThreads - 32;
static_assert(kThreads == kPlanThreads + 32 * kIndexWarps, "Unit plan role: 16 plan warps and 2 index warps");
static_assert(kP % kR == 0 and kR <= 64 and kP <= 2048 and kP > kE and kE % 8 == 0, "Unit plan geometry");
static constexpr uint32_t kMaxG = kP - kE;                                  // replicated experts (one extra slot each)
static constexpr uint32_t kMaxBig = (kP - kE) / 4 + 1;                      // experts with five or more active copies
static constexpr uint32_t kBigPerWarp = (kMaxBig + kPlanWarps - 1) / kPlanWarps;
static constexpr uint32_t kMaxCopy = kR < 32 ? kR : 32;                     // active copies: at most one per rank
static constexpr uint32_t kMaxPairs = kMaxG * kMaxCopy < 2048 ? kMaxG * kMaxCopy : 2048;   // host-checked (unit_section)
static constexpr int kBM = UNIT_BM, kWB = UNIT_WB, kWA = UNIT_WA;
static_assert(kMaxG <= kPlanThreads, "One thread per small replicated expert");

// Named barriers count warps: converge the warp first (inline asm is opaque to the compiler's reconvergence)
__device__ __forceinline__ void plan_sync() { __syncwarp(); asm volatile("bar.sync 1, 512;" ::: "memory"); }
__device__ __forceinline__ uint32_t plan_sync_count(const bool& pred) {
    __syncwarp();
    uint32_t c;
    asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %1, 0;\n bar.red.popc.u32 %0, 1, 512, p;\n}"
                 : "=r"(c) : "r"(static_cast<uint32_t>(pred)) : "memory");
    return c;
}
__device__ __forceinline__ void helper_sync() { __syncwarp(); asm volatile("bar.sync 2, 480;" ::: "memory"); }
__device__ __forceinline__ void index_sync() { __syncwarp(); asm volatile("bar.sync 3, 64;" ::: "memory"); }
__device__ __forceinline__ void stage_sync() { __syncwarp(); asm volatile("bar.sync 4, 96;" ::: "memory"); }

__device__ __forceinline__ void st_release_gpu(uint32_t* ptr, const uint32_t& value) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}
__device__ __forceinline__ void st_release_sys(uint32_t* ptr, const uint32_t& value) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(value) : "memory");
}
// Polling load: `asm volatile`, so the compiler cannot hoist it out of a spin loop (a plain `__ldcv` can be)
__device__ __forceinline__ uint4 ld_volatile_v4(const uint4* ptr) {
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr) : "memory");
    return v;
}
// Timeline trace point (globaltimer ns) into the signals' `eco_trace` slots (ECO_TRACE builds)
__device__ __forceinline__ void trace_point(uint8_t* buffer, const uint32_t& slot) {
#if defined(ECO_TRACE) and ECO_TRACE
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    reinterpret_cast<uint64_t*>(buffer + OFF_ECO_TRACE)[slot] = now;
#endif
}

// Rank time of a slot holding y full units and pp (0/1) partial units of rr tokens
__device__ __forceinline__ int pcost(const int& yy, const int& pp, const int& rr) {
    const int b = yy + pp;
    return kWB * b + (b > 0 ? kWA : 0) + kBM * yy + pp * rr;
}

extern "C" __global__ void __launch_bounds__(ECO_THREADS, 1)
eco_plan_kernel(const int* meta, uint8_t* buffer, uint16_t* state, int* stats, const uint64_t* peers,
                uint32_t rank, uint32_t num_tokens) {
    // Count gathering
    __shared__ alignas(16) uint16_t counts[kE];
    __shared__ alignas(16) uint32_t route_count[kE];
    // Static tables: active copies expert-major ascending in `cslot`; per slot: logical expert, active copy count of
    // its expert (0: passive), start of its expert's list; replicated experts split into small (<= 4 copies) / big
    __shared__ alignas(16) int phy[kP], clen[kP], cst[kP], cslot[kP], ecnt[kE], eoff[kE];
    __shared__ alignas(16) uint64_t ogids[2 * kMaxG];   // replicated expert ids (op 1), then the auction's offers
    int* const gids = reinterpret_cast<int*>(ogids);
    __shared__ alignas(16) int roff[(kR + 1 + 3) / 4 * 4];
    int* const small_ids = gids;
    int* const big_ids = gids + kMaxG;
    __shared__ alignas(16) uint32_t plist[kMaxPairs];   // per rank (slot << 16 | copy) pairs, ascending slots then copies
    // Plan state: tokens (warm: the previous quotas before the start), full units, partial flag, partial size
    __shared__ alignas(16) int q[kP], y[kP], pf[kP], rm[kP];
    __shared__ int T0[kR], T[kR];
    __shared__ uint32_t Bk[kR];
    __shared__ int dk2[kR], dd1[kR], de1[kR], dr2[kR], ds1b[kR], ds2b[kR];
    __shared__ int s_ns, s_nb, s_warm, s_prev;
    __shared__ int s_noff, cnt[kP];             // cnt: units per slot (full + partial)
    __shared__ int din2[kR], dip[kR];           // decoded best outflow: receiver time increase, partial flag
    __shared__ int rcnt[kR], rfill[kR], rbs[kR];   // offers per receiver, its segment fill and base
    __shared__ int Ts[kR];                      // one-hop rounds: the round's rank times (moves apply concurrently)
#if ECO_R <= 32
    __shared__ alignas(16) uint64_t ckey2[kP];  // one-hop rounds: the second candidate key buffer
#endif
    __shared__ uint32_t imin[kR];               // per intermediate rank: its two-hop offers' min (source time desc, s0)
    __shared__ int rfirst[kR];                  // per rank: first active replicated slot
    __shared__ alignas(16) uint64_t ckey[kP];   // per slot: the source candidate's sort key (~0: none)
    __shared__ int csh[kP];                     // per slot: the candidate's shed (source time decrease)
    __shared__ int oadd[2 * kMaxG];
    __shared__ alignas(16) uint16_t pcnt[kE];   // the previous call's counts (unit-space warm start)

    const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    // Debug (stats != nullptr): cycles since kernel entry at milestones, stats[kP / kR + 16 ...]
    const long long t_entry = clock64();
    const auto milestone = [&](const uint32_t& slot) {
        if (stats != nullptr)
            atomicAdd(stats + kP / kR + slot, static_cast<int>((clock64() - t_entry) / 16));
    };
    auto* signals = reinterpret_cast<uint32_t*>(buffer);
    const uint32_t epoch = *reinterpret_cast<volatile uint32_t*>(signals + OFF_EPOCH / 4);
    const uint32_t gen = epoch & 1u;
    const layout::EcoMeta online_meta(meta, ECO_E, ECO_P, ECO_R);
    if (tid == 0)
        trace_point(buffer, 3);
    const auto* routes = reinterpret_cast<const int64_t*>(buffer + OFF_TOPK);
    const uint32_t num_routes = num_tokens * ECO_TOPK;

    // ---- static tables and the warm start (warp 0 + index warps, while the counts warps gather) ----------------
    if (warp == 0 or warp >= kFirstIndexWarp) {
        const uint32_t sid = warp == 0 ? tid : 32 + tid - kPlanThreads;
        // Host-precomputed static tables: the meta's unit section (offset in header[15], `unit_ref.unit_section`)
        const int* us = meta + online_meta.get_header(15);
        constexpr uint32_t kOffSmall = 4, kOffBig = kOffSmall + kMaxG, kOffSlot = kOffBig + kMaxG;
        constexpr uint32_t kOffExp = kOffSlot + kP, kOffCslot = kOffExp + kE, kOffRoff = kOffCslot + kP;
        constexpr uint32_t kOffPairs = kOffRoff + (kR + 1 + 3) / 4 * 4;
        static_assert(kP % 8 == 0 and kE % 8 == 0 and kMaxG % 4 == 0, "Vector staging");
        const auto* us4 = reinterpret_cast<const int4*>(us);
        const auto* p2l4 = reinterpret_cast<const int4*>(meta + online_meta.get_phy2log_offset());
        constexpr uint32_t kSlotV = kP / 4, kExpV = kE / 4, kGV = 2 * kMaxG / 4, kRoffV = (kR + 1 + 3) / 4;
        constexpr uint32_t kSI = (kSlotV + 95) / 96, kEI = (kExpV + 95) / 96, kGI = (kGV + 95) / 96, kRI = (kRoffV + 95) / 96;
        constexpr uint32_t kQV = kP / 8, kNV = kE / 8, kQI = (kQV + 95) / 96, kNI = (kNV + 95) / 96;
        // All loads first (independent 16-byte loads in flight together), then the shared stores
        int4 wv[kSI], cv[kSI], pv[kSI], ev[kEI], gv[kGI], rv[kRI];
        uint4 qv[kQI], nv[kNI];
        #pragma unroll
        for (uint32_t i = 0; i < kSI; ++ i) {
            const uint32_t v = sid + 96 * i;
            if (v < kSlotV) {
                wv[i] = __ldg(us4 + kOffSlot / 4 + v); cv[i] = __ldg(us4 + kOffCslot / 4 + v); pv[i] = __ldg(p2l4 + v);
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < kEI; ++ i)
            if (sid + 96 * i < kExpV)
                ev[i] = __ldg(us4 + kOffExp / 4 + sid + 96 * i);
        #pragma unroll
        for (uint32_t i = 0; i < kGI; ++ i)
            if (sid + 96 * i < kGV)
                gv[i] = __ldg(us4 + kOffSmall / 4 + sid + 96 * i);
        #pragma unroll
        for (uint32_t i = 0; i < kRI; ++ i)
            if (sid + 96 * i < kRoffV)
                rv[i] = __ldg(us4 + kOffRoff / 4 + sid + 96 * i);
        if (state != nullptr) {
            const auto* st4 = reinterpret_cast<const uint4*>(state);
            #pragma unroll
            for (uint32_t i = 0; i < kQI; ++ i)
                if (sid + 96 * i < kQV)
                    qv[i] = st4[sid + 96 * i];
            #pragma unroll
            for (uint32_t i = 0; i < kNI; ++ i)
                if (sid + 96 * i < kNV)
                    nv[i] = st4[kQV + sid + 96 * i];
        }
        const int4 hdr = __ldg(us4);
        const int npairs = hdr.z;
        #pragma unroll
        for (uint32_t i = 0; i < kSI; ++ i) {
            const uint32_t v = sid + 96 * i;
            if (v < kSlotV) {
                const int w[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++ k) {
                    clen[4 * v + k] = w[k] & 0xff; cst[4 * v + k] = w[k] >> 8;
                }
                reinterpret_cast<int4*>(cslot)[v] = cv[i]; reinterpret_cast<int4*>(phy)[v] = pv[i];
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < kEI; ++ i) {
            const uint32_t v = sid + 96 * i;
            if (v < kExpV) {
                const int w[4] = {ev[i].x, ev[i].y, ev[i].z, ev[i].w};
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++ k) {
                    ecnt[4 * v + k] = w[k] & 0xff; eoff[4 * v + k] = w[k] >> 8;
                }
            }
        }
        #pragma unroll
        for (uint32_t i = 0; i < kGI; ++ i)
            if (sid + 96 * i < kGV)
                reinterpret_cast<int4*>(gids)[sid + 96 * i] = gv[i];
        #pragma unroll
        for (uint32_t i = 0; i < kRI; ++ i)
            if (sid + 96 * i < kRoffV)
                reinterpret_cast<int4*>(roff)[sid + 96 * i] = rv[i];
        for (int v0 = static_cast<int>(sid); v0 < (npairs + 3) / 4; v0 += 96 * 4) {   // four pair vectors in flight
            int4 pp[4];
            #pragma unroll
            for (int b = 0; b < 4; ++ b)
                if (v0 + 96 * b < (npairs + 3) / 4)
                    pp[b] = __ldg(us4 + kOffPairs / 4 + v0 + 96 * b);
            #pragma unroll
            for (int b = 0; b < 4; ++ b)
                if (v0 + 96 * b < (npairs + 3) / 4)
                    reinterpret_cast<int4*>(plist)[v0 + 96 * b] = pp[b];
        }
        if (sid < kR) {
            T0[sid] = 0; T[sid] = 0; Bk[sid] = ~0u;
        }
        if (sid == 0) {
            s_ns = hdr.x; s_nb = hdr.y; s_prev = 0;
        }
        if (state != nullptr) {   // warm start: the previous call's quotas into q, its counts into pcnt
            int part = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kQI; ++ i) {
                const uint32_t v = sid + 96 * i;
                if (v < kQV) {
                    const uint32_t w[4] = {qv[i].x, qv[i].y, qv[i].z, qv[i].w};
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++ k) {
                        q[8 * v + 2 * k] = static_cast<int>(w[k] & 0xffffu); q[8 * v + 2 * k + 1] = static_cast<int>(w[k] >> 16);
                    }
                }
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNI; ++ i) {
                const uint32_t v = sid + 96 * i;
                if (v < kNV) {
                    const uint32_t w[4] = {nv[i].x, nv[i].y, nv[i].z, nv[i].w};
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++ k) {
                        part += static_cast<int>((w[k] & 0xffffu) + (w[k] >> 16));
                        reinterpret_cast<uint32_t*>(pcnt)[4 * v + k] = w[k];
                    }
                }
            }
            part = __reduce_add_sync(~0u, part);
            if (lane == 0)
                atomicAdd(&s_prev, part);
        }
        if (sid == 0)
            milestone(24);
        stage_sync();
        if (sid == 0)
            milestone(25);
        const bool warm = state != nullptr and s_prev > 0;   // warm start iff the previous call left counts
        if (sid == 0)
            s_warm = warm;
        stage_sync();
        if (sid == 0)
            milestone(22);
    }

    if (warp >= kFirstIndexWarp)   // replicated experts' indices: written by the dispatch role's routing phase
        return;

    if (warp >= 1) {
        // ---- counts (warps 1..15): this rank's row into every peer, then every source's row summed ------------
        constexpr uint32_t kVectors = kE / 8;
        const uint32_t helper = tid - 32;
        for (uint32_t e = helper; e < kE; e += kHelpers)
            route_count[e] = 0;
        helper_sync();
        for (uint32_t i = helper; i < num_routes; i += kHelpers) {
            const auto e = static_cast<uint64_t>(__ldg(routes + i));
            if (e < kE)
                atomicAdd(route_count + e, 1u);
        }
        helper_sync();
        if (helper == 0)
            milestone(16);
        // Entries `(tag << 14) | count`, eight per 16-byte store
        const uint32_t tag = (epoch / 2) % 3 + 1, tag2 = (tag << 14) | (tag << 30);
        const auto pack = [&](const uint32_t& e) { return tag2 | route_count[e] | (route_count[e + 1] << 16); };
        for (uint32_t v = helper; v < kR * kVectors; v += kHelpers) {
            const uint32_t peer = v / kVectors, first = (v % kVectors) * 8;
            *reinterpret_cast<uint4*>(peers[peer] + OFF_ECO_COUNTS + ((gen * kR + rank) * kP + first) * 2) =
                make_uint4(pack(first), pack(first + 2), pack(first + 4), pack(first + 6));
        }
        if (helper == 0)
            milestone(17);
        // A thread owns one 8-expert column (16-byte loads, polled until all eight entries carry this call's tag)
        // over a strided subset of the sources
        const auto* table = reinterpret_cast<const uint4*>(buffer + OFF_ECO_COUNTS + gen * kR * kP * 2);
        constexpr uint32_t kSplit = kHelpers / kVectors > 0 ? kHelpers / kVectors : 1;
        helper_sync();
        for (uint32_t e = helper; e < kE; e += kHelpers)
            route_count[e] = 0;
        helper_sync();
        for (uint32_t item = helper; item < kVectors * kSplit; item += kHelpers) {
            const uint32_t v = item % kVectors, part = item / kVectors;
            uint32_t sum[8] = {};
            constexpr uint32_t kTagMask = (3u << 14) | (3u << 30);
            const auto stale = [&](const uint4& w) {
                return ((w.x & kTagMask) != tag2) | ((w.y & kTagMask) != tag2) | ((w.z & kTagMask) != tag2) |
                       ((w.w & kTagMask) != tag2);
            };
            constexpr uint32_t kBatchSrc = 8;
            for (uint32_t src0 = part; src0 < kR; src0 += kSplit * kBatchSrc) {
                uint4 w[kBatchSrc];
                #pragma unroll
                for (uint32_t b = 0; b < kBatchSrc; ++ b) {
                    const uint32_t src = src0 + b * kSplit;
                    w[b] = src < kR ? __ldcg(table + src * (kP / 8) + v) : make_uint4(tag2, tag2, tag2, tag2);
                }
                #pragma unroll
                for (uint32_t b = 0; b < kBatchSrc; ++ b) {
                    const uint32_t src = src0 + b * kSplit;
                    while (stale(w[b]))
                        w[b] = ld_volatile_v4(table + src * (kP / 8) + v);
                    const uint32_t ws[4] = {w[b].x, w[b].y, w[b].z, w[b].w};
                    #pragma unroll
                    for (uint32_t k = 0; k < 4; ++ k)
                        sum[2 * k] += ws[k] & 0x3fffu, sum[2 * k + 1] += (ws[k] >> 16) & 0x3fffu;
                }
            }
            #pragma unroll
            for (uint32_t k = 0; k < 8; ++ k)
                atomicAdd(route_count + v * 8 + k, sum[k]);
        }
        if (helper == 0)
            milestone(18);
        helper_sync();
        for (uint32_t e = helper; e < kE; e += kHelpers)
            counts[e] = static_cast<uint16_t>(route_count[e]);
        // This rank's single-copy slot totals for the compute role (its scheduler starts on them before the rank
        // barrier, like plain GC's receive counts)
        helper_sync();
        constexpr uint32_t kLocal = kP / kR;
        const uint32_t num_fixed = static_cast<uint32_t>(online_meta.get_num_fixed(rank));
        for (uint32_t j = helper; j < num_fixed; j += kHelpers)
            signals[OFF_ECO_SLOT_COUNT / 4 + j] = counts[online_meta.get_phy2log(rank * kLocal + j)];
        if (helper == 0)
            signals[OFF_ECO_NUM_FIXED / 4] = num_fixed;
        helper_sync();
        if (helper == 0) {
            __threadfence();
            st_release_gpu(signals + OFF_ECO_COUNTS_READY / 4, epoch + 1);
            milestone(19);
        }
    }

    // ---- plan (warps 0..15) ------------------------------------------------------------------------------------
    plan_sync();
    if (tid == 0)
        trace_point(buffer, 4);
    const bool warm = s_warm != 0;
    const int NS = s_ns, NB = s_nb;
    // Single-copy slots whole (their units too); passive slots empty
    for (uint32_t s = tid; s < kP; s += kPlanThreads) {
        const int n = counts[phy[s]];
        rm[s] = n > 0 ? n - ((n + kBM - 1) / kBM - 1) * kBM : 0;
        if (clen[s] <= 1) {
            const bool single = clen[s] == 1;
            const int v = single ? n : 0;
            q[s] = v;
            y[s] = single and n > 0 ? (n + kBM - 1) / kBM - 1 : 0;
            pf[s] = single and n > 0;
            if (warm and single and n > 0)   // warm start: the rank-time estimate T0, single-copy part
                atomicAdd(&T0[s / kS], pcost(y[s], pf[s], rm[s]));
        }
    }
    // Small replicated experts: one thread each (registers); big ones: a warp each (lane = copy)
    const bool sm_on = tid < static_cast<uint32_t>(NS);
    const int se = sm_on ? small_ids[tid] : 0, sk = sm_on ? ecnt[se] : 0, so = sm_on ? eoff[se] : 0;
    const int stot = sm_on ? counts[se] : 0;
    int sp[4], sq[4];
    #pragma unroll
    for (int j = 0; j < 4; ++ j)
        sp[j] = j < sk ? cslot[so + j] : 0;
    int bp[kBigPerWarp], bq[kBigPerWarp], bk[kBigPerWarp], btot[kBigPerWarp];
    bool bvalid[kBigPerWarp], bon[kBigPerWarp];
    #pragma unroll
    for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
        const int g = static_cast<int>(warp + kPlanWarps * i);
        bvalid[i] = g < NB;
        const int e = bvalid[i] ? big_ids[g] : 0;
        bk[i] = bvalid[i] ? ecnt[e] : 0;
        bon[i] = static_cast<int>(lane) < bk[i];
        bp[i] = bon[i] ? cslot[eoff[e] + lane] : 0;
        btot[i] = bvalid[i] ? counts[e] : 0;
    }
    if (warm) {
        // Unit-space warm start (`UnitPlanner._warm`): (1) the previous plan's units per copy (q: previous quotas, pcnt:
        // previous counts; the partial unit's owner is the first copy whose quota is congruent to the previous partial
        // size), the estimate T0 with the new partial sizes; (2) the experts whose unit count changed add units to the
        // copies with the lowest estimate (+ own additions) or remove them from the highest (+ own removals); the
        // partial stays with its owner while it holds a unit, else goes to the holding copy with the highest estimate
        constexpr int kUnit = kWB + kBM;
        int su[4] = {0, 0, 0, 0}, sown = -1;
        if (sm_on) {
            if (stot > 0) {
                const int npv = pcnt[se], kp = (npv + kBM - 1) / kBM;
                if (kp > 0) {
                    const int remp = npv - (kp - 1) * kBM;
                    #pragma unroll
                    for (int j = 0; j < 4; ++ j) {
                        const int qs = j < sk ? q[sp[j]] : 0;
                        if (j < sk and sown < 0 and qs >= remp and (qs - remp) % kBM == 0)
                            sown = j;
                    }
                    if (sown >= 0) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++ j)
                            su[j] = j < sk ? (q[sp[j]] - (j == sown ? remp : 0)) / kBM + (j == sown) : 0;
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                if (j < sk) {
                    const int yy = sown >= 0 ? su[j] - (j == sown) : 0, pp = j == sown;
                    y[sp[j]] = yy; pf[sp[j]] = pp;
                    atomicAdd(&T0[sp[j] / kS], pcost(yy, pp, stot > 0 ? stot - ((stot + kBM - 1) / kBM - 1) * kBM : 0));
                }
        }
        int bu[kBigPerWarp], bown[kBigPerWarp];
        #pragma unroll
        for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
            bu[i] = 0; bown[i] = -1;
            if (not bvalid[i])
                continue;
            if (btot[i] > 0) {
                const int be = __shfl_sync(~0u, bon[i] ? phy[bp[i]] : 0, 0);
                const int npv = pcnt[be], kp = (npv + kBM - 1) / kBM;
                if (kp > 0) {
                    const int remp = npv - (kp - 1) * kBM, qs = bon[i] ? q[bp[i]] : 0;
                    const uint32_t bal = __ballot_sync(~0u, bon[i] and qs >= remp and (qs - remp) % kBM == 0);
                    bown[i] = bal ? __ffs(bal) - 1 : -1;
                    if (bown[i] >= 0 and bon[i])
                        bu[i] = (qs - (static_cast<int>(lane) == bown[i] ? remp : 0)) / kBM + (static_cast<int>(lane) == bown[i]);
                }
            }
            if (bon[i]) {
                const int yy = bown[i] >= 0 ? bu[i] - (static_cast<int>(lane) == bown[i]) : 0;
                const int pp = static_cast<int>(lane) == bown[i];
                y[bp[i]] = yy; pf[bp[i]] = pp;
                atomicAdd(&T0[bp[i] / kS], pcost(yy, pp, btot[i] > 0 ? btot[i] - ((btot[i] + kBM - 1) / kBM - 1) * kBM : 0));
            }
        }
        plan_sync();
        if (sm_on and stot > 0) {
            int add[4] = {0, 0, 0, 0}, tr[4];
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                tr[j] = j < sk ? T0[sp[j] / kS] : 0;
            int delta = (stot + kBM - 1) / kBM - (su[0] + su[1] + su[2] + su[3]);
            for (; delta > 0; -- delta) {
                int jb = -1, vb = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++ j) {
                    const int v = tr[j] + add[j] * kUnit;
                    if (j < sk and (jb < 0 or v < vb)) {
                        jb = j; vb = v;
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; ++ j)
                    if (j == jb) {
                        ++ su[j]; ++ add[j];
                    }
            }
            for (; delta < 0; ++ delta) {
                int jb = -1, vb = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++ j) {
                    const int v = tr[j] + add[j] * kUnit;
                    if (j < sk and su[j] > 0 and (jb < 0 or v > vb)) {
                        jb = j; vb = v;
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; ++ j)
                    if (j == jb) {
                        -- su[j]; -- add[j];
                    }
            }
            bool keep = false;
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                keep |= j == sown and su[j] > 0;
            if (not keep) {
                int jb = -1, vb = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++ j)
                    if (j < sk and su[j] > 0 and (jb < 0 or tr[j] > vb)) {
                        jb = j; vb = tr[j];
                    }
                sown = jb;
            }
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                if (j < sk) {
                    y[sp[j]] = su[j] - (j == sown); pf[sp[j]] = j == sown;
                }
        }
        #pragma unroll
        for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
            if (not bvalid[i] or btot[i] == 0)
                continue;
            const int tr = bon[i] ? T0[bp[i] / kS] : 0;
            int add = 0, u = bu[i];
            int delta = (btot[i] + kBM - 1) / kBM - __reduce_add_sync(~0u, bon[i] ? u : 0);
            for (; delta > 0; -- delta) {
                const uint32_t mk = __reduce_min_sync(~0u, bon[i] ? ((static_cast<uint32_t>(tr + add * kUnit) << 5) | lane) : ~0u);
                if (lane == (mk & 31)) {
                    ++ u; ++ add;
                }
            }
            for (; delta < 0; ++ delta) {
                const uint32_t mk = __reduce_max_sync(~0u, (bon[i] and u > 0) ? ((static_cast<uint32_t>(tr + add * kUnit) << 5) | (31 - lane)) : 0u);
                if (lane == 31 - (mk & 31)) {
                    -- u; -- add;
                }
            }
            int own = bown[i];
            if (not __any_sync(~0u, static_cast<int>(lane) == own and u > 0)) {
                const uint32_t mk = __reduce_max_sync(~0u, (bon[i] and u > 0) ? ((static_cast<uint32_t>(tr) << 5) | (31 - lane)) : 0u);
                own = 31 - static_cast<int>(mk & 31);
            }
            if (bon[i]) {
                y[bp[i]] = u - (static_cast<int>(lane) == own); pf[bp[i]] = static_cast<int>(lane) == own;
            }
        }
        plan_sync();
    } else {   // cold: an even split rounded to units
        const int b0 = sk ? stot / sk : 0;
        #pragma unroll
        for (int j = 0; j < 4; ++ j)
            sq[j] = j < sk ? b0 + (j == 0 ? stot - b0 * sk : 0) : 0;
        #pragma unroll
        for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
            bq[i] = 0;
            if (bon[i]) {
                const int bb = btot[i] / bk[i];
                bq[i] = bb + (lane == 0 ? btot[i] - bb * bk[i] : 0);
            }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++ j)
            if (j < sk)
                q[sp[j]] = sq[j];
        #pragma unroll
        for (uint32_t i = 0; i < kBigPerWarp; ++ i)
            if (bon[i])
                q[bp[i]] = bq[i];
        plan_sync();
        // Rank times of the token split (rounding keys)
        for (uint32_t s = tid; s < kP; s += kPlanThreads) {
            const int x = q[s];
            if (x > 0)
                atomicAdd(&T0[s / kS], kWB * ((x + kBM - 1) / kBM) + kWA + x);
        }
        plan_sync();
        // Load-aware rounding to units: round up the copies with the smallest (rank time * bm - leftover * (wb + bm));
        // the partial unit goes to the rounded-up copy with the smallest leftover
        if (sm_on) {
            const int kk = (stot + kBM - 1) / kBM;
            int g[4], key[4], left = kk;
            #pragma unroll
            for (int j = 0; j < 4; ++ j) {
                g[j] = j < sk ? sq[j] / kBM : 0;
                left -= g[j];
                key[j] = j < sk ? T0[sp[j] / kS] * kBM - (sq[j] % kBM) * (kWB + kBM) : 0x7fffffff;
            }
            int jp = -1, bv = 0, gp[4];
            #pragma unroll
            for (int j = 0; j < 4; ++ j) {
                int rk = 0;
                #pragma unroll
                for (int i2 = 0; i2 < 4; ++ i2)
                    rk += (i2 != j and i2 < sk and (key[i2] < key[j] or (key[i2] == key[j] and i2 < j)));
                gp[j] = g[j] + (j < sk and rk < left);
                const int v = gp[j] * kBM - sq[j];
                if (j < sk and gp[j] > 0 and (jp < 0 or v > bv)) {
                    jp = j; bv = v;
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                if (j < sk) {
                    y[sp[j]] = gp[j] - (j == jp); pf[sp[j]] = j == jp;
                }
        }
        #pragma unroll
        for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
            if (not bvalid[i])
                continue;
            const int kk = (btot[i] + kBM - 1) / kBM, g = bon[i] ? bq[i] / kBM : 0;
            const int left = kk - __reduce_add_sync(~0u, g);
            const int key = bon[i] ? T0[bp[i] / kS] * kBM - (bq[i] % kBM) * (kWB + kBM) : 0;
            int sel = 0;
            for (int t_ = 0; t_ < left; ++ t_) {
                const uint32_t mk = __reduce_min_sync(~0u, (bon[i] and not sel) ? ((static_cast<uint32_t>(key) << 5) | lane) : 0xffffffffu);
                sel |= (mk & 31) == lane;
            }
            const int gp = g + sel;
            const uint32_t km = __reduce_max_sync(~0u, (bon[i] and gp > 0) ? ((static_cast<uint32_t>(gp * kBM - bq[i] + kBM) << 5) | (31 - lane)) : 0u);
            const int jp = km ? 31 - static_cast<int>(km & 31) : -1;
            if (bon[i]) {
                y[bp[i]] = gp - (static_cast<int>(lane) == jp); pf[bp[i]] = static_cast<int>(lane) == jp;
            }
        }
        plan_sync();
    }
    for (uint32_t s = tid; s < kP; s += kPlanThreads) {
        const int c = pcost(y[s], pf[s], rm[s]);
        cnt[s] = y[s] + pf[s];
        if (c)
            atomicAdd(&T[s / kS], c);
    }
    plan_sync();
    if (tid == 0)
        trace_point(buffer, 17);

    // ---- auction rounds -----------------------------------------------------------------------------------------
    const auto mv = [&](const int& a, const int& b) {   // one unit (the partial if `a` owns it) from slot a to slot b
        const int ya = y[a], pa = pf[a], ra = rm[a], yb = y[b], pb = pf[b];
        if (pa) {
            T[a / kS] += pcost(ya, 0, ra) - pcost(ya, 1, ra); T[b / kS] += pcost(yb, pb + 1, ra) - pcost(yb, pb, ra);
            pf[a] = 0; pf[b] = 1;
        } else {
            T[a / kS] += pcost(ya - 1, 0, ra) - pcost(ya, 0, ra); T[b / kS] += pcost(yb + 1, pb, ra) - pcost(yb, pb, ra);
            y[a] = ya - 1; y[b] = yb + 1;
        }
    };
#include "eco_unit_auction.cuh"
    plan_sync();
    if (tid == 0) {
        milestone(21);
        trace_point(buffer, 18);
    }

    // ---- publish: quotas, canonical starts within each expert's copies (ascending slots) -------------------------
    // Quotas (full units * bm + the partial unit) computed where they are read: no rewrite pass and barrier
    const auto quota = [&](const int& c) { return y[c] * kBM + pf[c] * rm[c]; };
    auto* out_quota = signals + OFF_ECO_QUOTA / 4;
    auto* out_start = signals + OFF_ECO_QUOTA_START / 4;
    for (uint32_t p = tid; p < kP; p += kPlanThreads) {
        const int o = cst[p], k = ecnt[phy[p]];
        int start = 0;
        for (int j = 0; j < k; ++ j) {   // active copies ascend: the copies below p (passive copies hold nothing)
            const int c = cslot[o + j];
            if (c >= static_cast<int>(p))
                break;
            start += quota(c);
        }
        out_quota[p] = static_cast<uint32_t>(quota(static_cast<int>(p)));
        out_start[p] = static_cast<uint32_t>(start);
    }
    plan_sync();
    if (tid == 0) {
        __threadfence();
        st_release_gpu(signals + OFF_ECO_PLAN_READY / 4, epoch + 1);
        trace_point(buffer, 5);
        milestone(20);
#if defined(ECO_TRACE) and ECO_TRACE
        auto* slots = reinterpret_cast<unsigned long long*>(buffer + OFF_ECO_TRACE);
        const auto since = slots[5] - *reinterpret_cast<volatile unsigned long long*>(slots + 24);
        atomicAdd(slots + 25, since);
        const auto entry = static_cast<uint32_t>(*reinterpret_cast<volatile unsigned long long*>(slots + 30)) % 64;
        reinterpret_cast<uint32_t*>(buffer + OFF_ECO_TRACE_RING)[entry * 8 + 3] = static_cast<uint32_t>(since);
#endif
    }
    // ---- next call's warm state (after the release, off the critical path): this plan's quotas of the replicated
    // slots, then the counts
    if (state != nullptr) {
        for (uint32_t p = tid; p < kP; p += kPlanThreads)
            state[p] = static_cast<uint16_t>(clen[p] >= 2 ? quota(static_cast<int>(p)) : 0);
        for (uint32_t e = tid; e < kE; e += kPlanThreads)
            state[kP + e] = counts[e];
    }
}

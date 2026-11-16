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
// Warp roles: warps 0..15 plan (named barrier 1, 512 threads); warps 1..15 gather the count table (barrier 2,
// 480 threads) and stage the unit tables / reconstruct phy while polling; the two index warps 16, 17 fan this
// rank's packed count row out to every peer (barrier 5, 544 threads) and return. With dispatch-gated rows
// (OFF_ECO_PUB_ISSUED) the send waits for this rank's dispatch count publication; otherwise rows go out first.
// With ECO_SPLIT_COUNTS, CTA 0 counts/sends with all 18 warps; CTA 1 stages/receives/plans with
// warps 0..15. A two-SM cluster keeps both CTAs inside the existing planner reservation.
// Neither the count exchange nor planning gates single-copy dispatch/compute.

#include <cstdint>

#ifndef UNIT_HYDRO
#define UNIT_HYDRO 0   // 1: stateless block-peak waterfill + augmenting repair replaces warm/cold + auction
#endif
#ifndef UNIT_HDBG
#define UNIT_HDBG 0    // 1: bound every serial loop; write a stall marker to stats[kP/kR + 60..63]
#endif

#include "include/eco_meta.cuh"

using namespace deep_gemm;

static constexpr uint32_t kP = ECO_P, kR = ECO_R, kE = ECO_E, kS = kP / kR, kThreads = ECO_THREADS;
#if ECO_ROUTE_PREP
static_assert(ECO_SPLIT_COUNTS, "route preparation needs the producer CTA");
static constexpr uint32_t kDispatchSms = ECO_DISPATCH_SMS, kDispatchWarps = ECO_DISPATCH_WARPS;
static constexpr uint32_t kTokensPerDispatchWarp = 32 / ECO_TOPK;
#endif
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
#if UNIT_HYDRO
static constexpr int kMaxU = 2 * kP;      // tracked movable block units (overflow: plan degrades to greedy)
#ifndef UNIT_HREFINE
#define UNIT_HREFINE 8
#endif
#ifndef UNIT_HAUG
#define UNIT_HAUG 64
#endif
#ifndef UNIT_HSPLIT
#define UNIT_HSPLIT 0  // 1: an expert's tokens spread evenly over its nb units (first n % nb carry +1), not bm + tail
#endif
#ifndef UNIT_HPROF
#define UNIT_HPROF 0   // 1: per-segment clocks of the greedy expert loop into hsc[14..19] (debug)
#endif
#ifndef UNIT_HCF
#define UNIT_HCF 3     // experts with >= UNIT_HCF units deal their picks in closed form (ballot per level)
#endif
#ifndef UNIT_HCFLV
#define UNIT_HCFLV 3   // levels the closed-form deal resolves at once (deeper water: per-unit fallback)
#endif
#ifndef UNIT_HORD
#define UNIT_HORD 0    // 1: greedy order = fewest copies first, then hottest (most constrained first)
#endif
#ifndef UNIT_HLB
#define UNIT_HLB 1     // 1: relocate / BFS stop once the peak sits on max(ceil(total / R), max floor)
#endif
static constexpr int kCfLv = UNIT_HCFLV;
static constexpr int kHydRefine = UNIT_HREFINE, kHydAug = UNIT_HAUG, kHydDepth = 8;
#endif

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
__device__ __forceinline__ void rows_sync() { __syncwarp(); asm volatile("bar.sync 5, 544;" ::: "memory"); }
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
// Sys-scope acquire for a remote peer's row-tail release flag: orders this thread's later loads of the row data
__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t* ptr) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}
// Timeline trace point (globaltimer ns) into the signals' `eco_trace` slots (ECO_TRACE builds)
__device__ __forceinline__ uint64_t trace_point(uint8_t* buffer, const uint32_t& slot,
                                              const int* sync_state = nullptr) {
    uint64_t now = 0;
#if defined(ECO_TRACE) and ECO_TRACE
    if (sync_state != nullptr) {
        // BAR.SYNC.DEFER_BLOCKING can let a clock read run before the barrier actually completes. A shared
        // load followed by a dependent predicate makes the timer wait too; sync_state is the shared 0/1 flag.
        const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(sync_state));
        asm volatile("{ .reg .u32 ready; .reg .pred valid; "
                     "ld.volatile.shared.u32 ready, [%1]; setp.le.u32 valid, ready, 1; "
                     "@valid mov.u64 %0, %%globaltimer; }"
                     : "+l"(now) : "r"(addr) : "memory");
    } else {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
    }
    reinterpret_cast<uint64_t*>(buffer + OFF_ECO_TRACE)[slot] = now;
#endif
    return now;
}

// Rank time of a slot holding y full units and pp (0/1) partial units of rr tokens
__device__ __forceinline__ int pcost(const int& yy, const int& pp, const int& rr) {
    const int b = yy + pp;
    return kWB * b + (b > 0 ? kWA : 0) + kBM * yy + pp * rr;
}

#if ECO_SPLIT_COUNTS
// Each sender writes self-tagged payloads to the receiving rank's local table. The split variant
// uses a producer CTA in the same two-SM cluster as the planner, without an inter-CTA handshake.
__device__ __forceinline__ void send_count_rows(uint8_t* buffer, const uint64_t* peers,
                                               const uint32_t* row, uint32_t rank, uint32_t epoch,
                                               uint32_t worker, uint32_t workers) {
#ifdef OFF_ECO_PUB_ISSUED
    if (worker % 32 == 0)
        while (ld_acquire_sys_u32(reinterpret_cast<uint32_t*>(buffer + OFF_ECO_PUB_ISSUED)) != epoch + 1) {}
    __syncwarp();
#endif
    if (worker == 0)
        trace_point(buffer, 37);
    constexpr uint32_t kVectors = kE / 12;
    const uint32_t gen = epoch & 1u, taghi = ((epoch / 2) % 3 + 1) << 30;
    const auto pack3 = [&](uint32_t e) {
        return taghi | min(row[e], 1023u) | (min(row[e + 1], 1023u) << 10) |
               (min(row[e + 2], 1023u) << 20);
    };
    for (uint32_t v = worker; v < kR * kVectors; v += workers) {
        uint32_t peer = v / kVectors;
        const uint32_t first = (v % kVectors) * 12;
        *reinterpret_cast<uint4*>(peers[peer] + OFF_ECO_COUNTS + (gen * kR + rank) * kP * 2 +
                                 (v % kVectors) * 16) =
            make_uint4(pack3(first), pack3(first + 3), pack3(first + 6), pack3(first + 9));
    }
    if (worker == 0)
        trace_point(buffer, 38);
}

#endif
extern "C" __global__ void __launch_bounds__(ECO_THREADS, 1)
#if ECO_SPLIT_COUNTS
__cluster_dims__(2, 1, 1)
#endif
eco_plan_kernel(const int* meta, uint8_t* buffer, uint16_t* state, int* stats, const uint64_t* peers,
                uint32_t rank, uint32_t num_tokens) {
    // Count gathering
#if ECO_SPLIT_COUNTS
    // Direct accumulation avoids a second helper barrier and narrowing copy.
    __shared__ alignas(16) uint32_t counts[kE];
#else
    __shared__ alignas(16) uint16_t counts[kE];
#endif
    __shared__ alignas(16) uint32_t route_count[kE];
#if ECO_ROUTE_PREP
    // One local routing pass produces both this rank's expert totals and every dispatch CTA's
    // source-index start. The dispatch partition consumes the resulting prefix table.
    __shared__ alignas(16) uint32_t cta_count[kDispatchSms * kE];
#endif
#if !ECO_SPLIT_COUNTS
    __shared__ alignas(16) uint32_t sums[kE];   // global per-expert totals (count scan; separate so polling can start while other threads still write rows)
#endif
    // Static tables: active copies expert-major ascending in `cslot`; per slot: logical expert, active copy count of
    // its expert (0: passive), start of its expert's list; replicated experts split into small (<= 4 copies) / big
    __shared__ alignas(16) int phy[kP], clen[kP], cst[kP], cslot[kP], ecnt[kE], eoff[kE];
    __shared__ alignas(16) uint64_t ogids[2 * kMaxG < kP ? kP : 2 * kMaxG];   // replicated expert ids (op 1), then the auction's offers
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
    __shared__ volatile int s_state_ready;
    __shared__ int s_noff, cnt[kP];             // cnt: units per slot (full + partial)
    __shared__ int din2[kR], dip[kR];           // decoded best outflow: receiver time increase, partial flag
    __shared__ int rcnt[kR], rfill[kR], rbs[kR];   // offers per receiver, its segment fill and base
    __shared__ int Ts[kR];                      // one-hop rounds: the round's rank times (moves apply concurrently)
    static constexpr int kEmW = (kP / kR + 31) / 32;
    __shared__ uint32_t emask[kR * kEmW];       // per rank: bitmask of slots that got a candidate key this round
    __shared__ uint16_t elist[kP];              // per rank: the emitted source slots, compacted (dedup via emask)
    __shared__ int ecntr[kR];                   // per rank: emitted count
#if ECO_R <= 32
    __shared__ alignas(16) uint64_t ckey2[kP];  // one-hop rounds: the second candidate key buffer
#endif
    __shared__ uint32_t imin[kR];               // per intermediate rank: its two-hop offers' min (source time desc, s0)
    __shared__ int rfirst[kR];                  // per rank: first active replicated slot
    __shared__ alignas(16) uint64_t ckey[kP];   // per slot: the source candidate's sort key (~0: none)
    __shared__ int csh[kP];                     // per slot: the candidate's shed (source time decrease)
    __shared__ int oadd[2 * kMaxG < kP ? kP : 2 * kMaxG];
    __shared__ alignas(16) uint16_t pcnt[kE];   // the previous call's counts (unit-space warm start)
#if UNIT_HYDRO
    // Hydro plan state (replaces the warm/cold start + auction), aliased onto the auction's shared
    // arrays, which are all dead in this mode: per-rank block loads, the movable experts' units
    // (placement order, bit15 = the expert's partial unit), per-rank unit ordering sequences
    // (augmenting moves take dense tail positions, matching the reference's remove + append order),
    // and the multi-source BFS tables. `bedge[r]` is the u32 order key (fseq << 16 | unit position
    // << 5 | copy) of the best edge into rank r — native ATOMS.MIN.32; the winning edge's unit and
    // slot are decoded from the key itself via the queue (frk) and the unit lists (ilist/ioff).
    static_assert(kMaxU <= 2 * kP and kMaxPairs >= kMaxU + 2 * kR, "Hydro scratch must fit the auction buffers");
    static_assert(kMaxU <= 2 * (kP > 2 * kMaxG ? kP : 2 * kMaxG), "ilist must fit ogids");
    static_assert(kR * kEmW >= 30, "hsc scalars (hsc[0..29]) must fit emask");
    static_assert(kE <= (2 * kMaxG < kP ? kP : 2 * kMaxG), "epk must fit oadd");
    int* const hload = T;
    int* const ucnt = dk2;
    int* const mord = csh;
    auto* const hplaced = reinterpret_cast<uint16_t*>(cnt);
    int* const upos = reinterpret_cast<int*>(plist);
    uint32_t* const bedge = reinterpret_cast<uint32_t*>(plist + kMaxU); // u32 order keys: native ATOMS.MIN
    int* const useq = de1;                                              // per-rank monotone append stamp
    int* const frk = T0;                                                // BFS queue: queue index -> rank
    int* const bpar = din2;
    int* const bdep = dip;
    int* const bparu = dr2;
    int* const fseq = ds1b;
    int* const hsc = reinterpret_cast<int*>(emask);   // the loop scalars below
    // volatile: lane0 writes these inside divergent `if (lane == 0)` regions and the other lanes read
    // them after __syncwarp — register promotion would let them read a stale zero
    volatile int& s_nm = hsc[0], & s_nu = hsc[1];
    volatile int& s_fmax = hsc[24], & s_fsum = hsc[25], & s_lb = hsc[26], & s_lbx = hsc[27];
    // even split: base tokens of one unit of slot s's expert (a flagged unit carries one more)
    const auto ueven = [&](const int& sl) { const int ne = counts[phy[sl]]; return ne / ((ne + kBM - 1) / kBM); };
    int* const epk = oadd;                             // per-expert packs (oadd is free pre-epilog)
    uint64_t* const erm = ogids;                       // per-expert rank masks of the copies (0: duplicate ranks)
    int* const clane = reinterpret_cast<int*>(ckey);           // flattened per-(expert, lane) copy slots
#endif

    const uint32_t tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    // Debug (stats != nullptr): cycles since kernel entry at milestones, stats[kP / kR + 16 ...]
    const long long t_entry = clock64();
    const auto milestone = [&](const uint32_t& slot) {
        if (stats != nullptr)
            *reinterpret_cast<volatile int*>(stats + kP / kR + slot) =
                static_cast<int>((clock64() - t_entry) / 16);
    };
    auto* signals = reinterpret_cast<uint32_t*>(buffer);
    const uint32_t epoch = *reinterpret_cast<volatile uint32_t*>(signals + OFF_EPOCH / 4);
    const uint32_t gen = epoch & 1u;

    // Replica slots pull through u64 receive-count rows materialized from the assembled count matrix:
    // each index-warp lane owns one source rank and copies this rank's replica experts' counts as soon
    // as their words arrive tagged. ECO_RECV_MODE < 3 targets the native expert_recv_count region;
    // mode 3 uses the dedicated ECO-region rows instead (the native area measured ~3 us slower for
    // these reads on L20B: it shares the routing-flooded L2 space with the per-token counters).
    // The rows land before the plan acquire; the two generation banks keep stale reads impossible.
    const auto fill_replica_counts = [&] {
        const layout::EcoMeta online_meta(meta, ECO_E, ECO_P, ECO_R);
        const uint32_t nf = online_meta.get_num_fixed(rank);
        if (nf >= kS)
            return;
        const uint32_t taghi = ((epoch / 2u) % 3u + 1u) << 30;
        const uint32_t iw = tid - kFirstIndexWarp * 32u;
        if (iw >= kR)
            return;
        const auto* row = reinterpret_cast<const volatile uint32_t*>(
            buffer + OFF_ECO_COUNTS + gen * kR * kP * 2u + iw * kP * 2u);
        constexpr uint32_t kOutOff = ECO_RECV_MODE == 3 ? OFF_ECO_RECV_COUNTS : OFF_RECV_COUNT;
        auto* out = reinterpret_cast<uint64_t*>(buffer + kOutOff) +
                    (static_cast<uint64_t>(gen) * kR + iw) * kS;
        for (uint32_t l = nf; l < kS; ++ l) {
            const uint32_t e = static_cast<uint32_t>(online_meta.get_phy2log(rank * kS + l));
            const auto* w = row + e / 3u;
            uint32_t word = *w, backoff = 32u;
            while ((word & 0xc0000000u) != taghi) {
                __nanosleep(backoff);
                backoff = min(backoff * 2u, 512u);
                word = *w;
            }
            out[l] = static_cast<uint64_t>((word >> ((e % 3u) * 10u)) & 0x3ffu);
        }
    };
    // Replica slots carry no source-index rows: the dispatch role writes each logical
    // expert's indices into its lowest (canonical) copy's rows only, and every replica's
    // pull reads that canonical row directly on its owner rank — complete once all
    // dispatches exit the rank barrier. No post-barrier fan-out is needed at all.
#if ECO_SPLIT_COUNTS
    if (blockIdx.x == 0) {
#if ECO_ROUTE_PREP
        for (uint32_t i = tid; i < kDispatchSms * kE; i += kThreads)
            cta_count[i] = 0;
#endif
        for (uint32_t e = tid; e < kE; e += kThreads)
            route_count[e] = 0;
        __syncthreads();
        for (uint32_t i = tid; i < num_tokens * ECO_TOPK; i += kThreads) {
            const auto e = static_cast<uint64_t>(__ldg(reinterpret_cast<const int64_t*>(buffer + OFF_TOPK) + i));
            if (e < kE) {
                atomicAdd(route_count + e, 1u);
#if ECO_ROUTE_PREP
                // Dispatch assigns consecutive route chunks round-robin over (CTA, warp).
                const uint32_t chunk = i / (ECO_TOPK * kTokensPerDispatchWarp);
                const uint32_t cta = (chunk % (kDispatchSms * kDispatchWarps)) / kDispatchWarps;
                atomicAdd(cta_count + e * kDispatchSms + cta, 1u);
#endif
            }
        }
        __syncthreads();
        if (tid == 0)
            milestone(16);
#if ECO_ROUTE_PREP
        // Rows issue first at the full producer width; dispatch's own staging covers the offset
        // table build below, so its route_ready wait never blocks.
        send_count_rows(buffer, peers, route_count, rank, epoch, tid, kThreads);
        if (tid == 0)
            milestone(17);
        // Every producer thread prefixes one expert's per-CTA counts (consecutive words), writes
        // each dispatch CTA's source-index start and the final send_count status (all CTAs arrived).
        auto* route_offsets = reinterpret_cast<uint32_t*>(buffer + OFF_ECO_ROUTE_OFFSETS);
        auto* send_count = reinterpret_cast<uint64_t*>(buffer + OFF_SEND_COUNT);
        for (uint32_t e = tid; e < kE; e += kThreads) {
            uint32_t start = 0;
            for (uint32_t cta = 0; cta < kDispatchSms; ++ cta) {
                const uint32_t n = cta_count[e * kDispatchSms + cta];
                route_offsets[cta * kE + e] = start;
                start += n;
            }
            send_count[e] = (static_cast<uint64_t>(kDispatchSms) << 32) | start;
        }
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            st_release_gpu(signals + OFF_ECO_ROUTE_READY / 4, epoch + 1);
            trace_point(buffer, 42);
        }
        return;
#else
        send_count_rows(buffer, peers, route_count, rank, epoch, tid, kThreads);
        if (tid == 0)
            milestone(17);
        return;
#endif
    }
    // The producer occupies the other SM of this cluster. These two warps have no sending work here;
    // they publish this rank's replica slots' native-format count rows instead.
    if (warp >= kFirstIndexWarp) {
        fill_replica_counts();
        return;
    }
#endif
    const layout::EcoMeta online_meta(meta, ECO_E, ECO_P, ECO_R);
    if (tid == 0)
        trace_point(buffer, 3);
    const auto* routes = reinterpret_cast<const int64_t*>(buffer + OFF_TOPK);
    const uint32_t num_routes = num_tokens * ECO_TOPK;

    const auto send_rows = [&] {
        // The count rows go out FIRST on the two index warps: route_count[] is ready a few us in, and rows sent
        // this early drain ahead of the dispatch role's routing flood instead of queueing behind a ~10us staging
        // phase; their posted remote stores' fabric drain then stalls only these warps' LSU path.
        rows_sync();   // route_count[] complete (helpers arrive once counting is done)
#if ECO_SPLIT_COUNTS
        send_count_rows(buffer, peers, route_count, rank, epoch, tid - kFirstIndexWarp * 32, 64);
        if (tid == kFirstIndexWarp * 32) {
            milestone(17);
        }
#else
#ifdef OFF_ECO_PUB_ISSUED
        // The rows follow this rank's dispatch count publication (its critical pre-barrier remote writes): the two
        // traffic streams go out back to back instead of interleaving in the fabric
        if (lane == 0) {
            uint32_t seen;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(signals + OFF_ECO_PUB_ISSUED / 4) : "memory");
            } while (seen != epoch + 1);
        }
        __syncwarp();
#endif
        if (tid == kFirstIndexWarp * 32)
            trace_point(buffer, 37);
        if (stats != nullptr and lane == 0)
            stats[kP / kR + 44 + warp] = 21;
        constexpr uint32_t kVectors = kE / 12;
        const uint32_t tag = (epoch / 2) % 3 + 1, taghi = tag << 30;
        const auto pack3 = [&](const uint32_t& e) {
            return taghi | min(route_count[e], 1023u) | (min(route_count[e + 1], 1023u) << 10) |
                   (min(route_count[e + 2], 1023u) << 20);
        };
        const uint32_t iw = tid - kFirstIndexWarp * 32;
        for (uint32_t v = iw; v < kR * kVectors; v += 64) {
            const uint32_t peer = v / kVectors, first = (v % kVectors) * 12;
            *reinterpret_cast<uint4*>(peers[peer] + OFF_ECO_COUNTS + (gen * kR + rank) * kP * 2 +
                                    (v % kVectors) * 16) =
                make_uint4(pack3(first), pack3(first + 3), pack3(first + 6), pack3(first + 9));
        }
        if (tid == kFirstIndexWarp * 32) {
            milestone(17);
            trace_point(buffer, 38);
        }
#endif
    };
#ifndef OFF_ECO_PUB_ISSUED
    if (warp >= kFirstIndexWarp)
        send_rows();
#endif

    // ---- roff, the warm start and scalar headers (warp 0 alone: the index warps return after their row send
    // instead of waiting at stage_sync, and the slot/expert tables are staged by the helpers) -------------------
    if (warp == 0) {
        const uint32_t sid = tid;
        // Host-precomputed static tables: the meta's unit section (offset in header[15], `unit_ref.unit_section`)
        const int* us = meta + online_meta.get_header(15);
        constexpr uint32_t kOffSmall = 4, kOffBig = kOffSmall + kMaxG, kOffSlot = kOffBig + kMaxG;
        constexpr uint32_t kOffExp = kOffSlot + kP, kOffCslot = kOffExp + kE, kOffRoff = kOffCslot + kP;
        static_assert(kP % 8 == 0 and kE % 8 == 0 and kMaxG % 4 == 0, "Vector staging");
        const auto* us4 = reinterpret_cast<const int4*>(us);
        constexpr uint32_t kRoffV = (kR + 1 + 3) / 4;
        constexpr uint32_t kQV = kP / 8, kNV = kE / 8, kQI = (kQV + 31) / 32, kNI = (kNV + 31) / 32;
        uint4 qv[kQI], nv[kNI];
        if (sid < kRoffV)
            reinterpret_cast<int4*>(roff)[sid] = __ldg(us4 + kOffRoff / 4 + sid);
        if (state != nullptr) {
            const auto* st4 = reinterpret_cast<const uint4*>(state);
            #pragma unroll
            for (uint32_t i = 0; i < kQI; ++ i)
                if (sid + 32 * i < kQV)
                    qv[i] = st4[sid + 32 * i];
            #pragma unroll
            for (uint32_t i = 0; i < kNI; ++ i)
                if (sid + 32 * i < kNV)
                    nv[i] = st4[kQV + sid + 32 * i];
        }
        const int4 hdr = __ldg(us4);
        for (uint32_t r = sid; r < kR; r += 32) {
            T0[r] = 0; T[r] = 0; Bk[r] = ~0u;
        }
        if (sid == 0) {
            s_ns = hdr.x; s_nb = hdr.y; s_prev = 0; s_state_ready = 0;
        }
        if (state != nullptr) {   // warm start: the previous call's quotas into q, its counts into pcnt
            int part = 0;
            #pragma unroll
            for (uint32_t i = 0; i < kQI; ++ i) {
                const uint32_t v = sid + 32 * i;
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
                const uint32_t v = sid + 32 * i;
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
        __syncwarp();
        if (sid == 0)
            milestone(25);
        const bool warm = state != nullptr and s_prev > 0;   // warm start iff the previous call left counts
        if (sid == 0)
            s_warm = warm;
        __syncwarp();
        if (sid == 0) {
            // Publish q/pcnt/s_warm to the helpers, which assign the replicated slots' warm
            // units while the count rows are still in flight.
            __threadfence_block();
            s_state_ready = 1;
        }
        if (sid == 0) {
            milestone(22);
            trace_point(buffer, 39, &s_warm);
        }
    }

#ifdef OFF_ECO_PUB_ISSUED
    // Local preparation needs neither the dispatch publication nor global counts. Finish it during the gate
    // window, so warp 0 cannot hold up planning on a post-send static-table / warm-state load.
    if (warp >= kFirstIndexWarp)
        send_rows();
#endif
    if (warp >= kFirstIndexWarp) {
        fill_replica_counts();
        return;   // replicated experts' indices are read from the canonical rows directly
    }

    if (warp >= 1) {
        // ---- counts (warps 1..15): every lane packs (tag << 30) | three 10-bit counts; a thread owns one
        // 12-expert column over a strided subset of the sources and re-polls stale ones in parallel --------------
        const uint32_t helper = tid - 32;
        for (uint32_t e = helper; e < kE; e += kHelpers)
#if ECO_SPLIT_COUNTS
            route_count[e] = counts[e] = 0;
#else
            route_count[e] = sums[e] = 0;
#endif
        helper_sync();
#if !ECO_SPLIT_COUNTS
        for (uint32_t i = helper; i < num_routes; i += kHelpers) {
            const auto e = static_cast<uint64_t>(__ldg(routes + i));
            if (e < kE)
                atomicAdd(route_count + e, 1u);
        }
        helper_sync();
        if (stats != nullptr and lane == 0)
            stats[kP / kR + 44 + warp] = 10;
        rows_sync();   // hand route_count[] to the index warps; they issue this rank's remote row stores
        if (stats != nullptr and lane == 0)
            stats[kP / kR + 44 + warp] = 11;
        if (helper == 0)
            milestone(16);
#endif
        // The slot/expert tables while the count rows are in flight (the remote row send is owned by the index
        // warps, so the helpers' loads overlap only the peer-arrival window): cslot/ecnt/eoff enumerate each
        // expert's active copies, clen/cst give the per-slot view, gids/plist drive the auction, and phy
        // maps each physical slot back to its logical expert.
        {
            const int* us = meta + online_meta.get_header(15);
            const auto* us4 = reinterpret_cast<const int4*>(us);
            constexpr uint32_t kOffSmall = 4, kOffBig = kOffSmall + kMaxG, kOffSlot = kOffBig + kMaxG;
            constexpr uint32_t kOffExp = kOffSlot + kP, kOffCslot = kOffExp + kE;
            constexpr uint32_t kOffPairs = kOffCslot + kP + (kR + 1 + 3) / 4 * 4;
            constexpr uint32_t kSlotV = kP / 4, kExpV = kE / 4, kGV = 2 * kMaxG / 4;
            const uint32_t npairs = static_cast<uint32_t>(__ldg(us + 2));
            for (uint32_t v = helper; v < kSlotV; v += kHelpers) {
                const int4 w = __ldg(us4 + kOffSlot / 4 + v);
                const int ws[4] = {w.x, w.y, w.z, w.w};
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++ k) {
                    clen[4 * v + k] = ws[k] & 0xff; cst[4 * v + k] = ws[k] >> 8;
                }
                reinterpret_cast<int4*>(cslot)[v] = __ldg(us4 + kOffCslot / 4 + v);
            }
            for (uint32_t v = helper; v < kExpV; v += kHelpers) {
                const int4 w = __ldg(us4 + kOffExp / 4 + v);
                const int ws[4] = {w.x, w.y, w.z, w.w};
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++ k) {
                    ecnt[4 * v + k] = ws[k] & 0xff; eoff[4 * v + k] = ws[k] >> 8;
                }
            }
            for (uint32_t v = helper; v < kGV; v += kHelpers)
                reinterpret_cast<int4*>(gids)[v] = __ldg(us4 + kOffSmall / 4 + v);
            for (uint32_t v = helper; v < (npairs + 3) / 4; v += kHelpers)
                reinterpret_cast<int4*>(plist)[v] = __ldg(us4 + kOffPairs / 4 + v);
            const auto* p2l4 = reinterpret_cast<const int4*>(meta + online_meta.get_phy2log_offset());
            for (uint32_t v = helper; v < kP / 4; v += kHelpers)
                reinterpret_cast<int4*>(phy)[v] = __ldg(p2l4 + v);
        }
        // ---- warm assignment while the count rows are in flight --------------------------------------------------
        // The replicated slots' unit split depends only on the previous call's state (q: quotas, pcnt:
        // counts) and the staged tables — not on this call's totals — so it moves ahead of the row
        // polling. Slots whose expert has no tokens this call are cleared after the counts join.
        helper_sync();   // staged tables are read across helpers below
        while (s_state_ready == 0)
            __nanosleep(32);
        __threadfence_block();
        if (s_warm != 0 and not UNIT_HYDRO) {
            const int nsm = s_ns, nbg = s_nb;
            for (int g = static_cast<int>(helper); g < nsm; g += kHelpers) {
                const int se = small_ids[g], sk = ecnt[se], so = eoff[se];
                int s4[4];
                #pragma unroll
                for (int j = 0; j < 4; ++ j)
                    s4[j] = (j < sk) ? cslot[so + j] : -1;
                const int npv = pcnt[se], kp = (npv + kBM - 1) / kBM;
                const int remp = npv - (kp - 1) * kBM;
                int sown = -1;
                #pragma unroll
                for (int j = 0; j < 4; ++ j) {   // the previous partial's owner: congruent quota
                    const int qs = s4[j] >= 0 ? q[s4[j]] : 0;
                    if (sown < 0 and s4[j] >= 0 and kp > 0 and qs >= remp and (qs - remp) % kBM == 0)
                        sown = j;
                }
                #pragma unroll
                for (int j = 0; j < 4; ++ j)
                    if (s4[j] >= 0) {
                        const int qs = q[s4[j]];
                        const int suj = sown >= 0 ? (qs - (j == sown ? remp : 0)) / kBM + (j == sown) : 0;
                        y[s4[j]] = suj - (j == sown); pf[s4[j]] = j == sown;
                    }
            }
            constexpr uint32_t kHW = kHelpers / 32;
            constexpr uint32_t kBPH = (kMaxBig + kHW - 1) / kHW;
            #pragma unroll
            for (uint32_t i = 0; i < kBPH; ++ i) {
                const int g = static_cast<int>(warp - 1) + static_cast<int>(kHW) * static_cast<int>(i);
                if (g >= nbg)
                    continue;
                const int e = big_ids[g], bk_ = ecnt[e], bo = eoff[e];
                const bool on = static_cast<int>(lane) < bk_;
                const int bp_ = on ? cslot[bo + lane] : 0;
                const int be = __shfl_sync(~0u, on ? phy[bp_] : 0, 0);
                const int npv = pcnt[be], kp = (npv + kBM - 1) / kBM;
                int bown_ = -1, bu_ = 0;
                if (kp > 0) {
                    const int remp = npv - (kp - 1) * kBM, qs = on ? q[bp_] : 0;
                    const uint32_t bal = __ballot_sync(~0u, on and qs >= remp and (qs - remp) % kBM == 0);
                    bown_ = bal ? static_cast<int>(__ffs(bal)) - 1 : -1;
                    if (bown_ >= 0 and on)
                        bu_ = (qs - (static_cast<int>(lane) == bown_ ? remp : 0)) / kBM + (static_cast<int>(lane) == bown_);
                }
                if (on) {
                    y[bp_] = bu_ - (static_cast<int>(lane) == bown_);
                    pf[bp_] = static_cast<int>(lane) == bown_;
                }
            }
        }
        const uint32_t tag = (epoch / 2) % 3 + 1, taghi = tag << 30;
        const auto* table = reinterpret_cast<const uint4*>(buffer + OFF_ECO_COUNTS + gen * kR * kP * 2);
        constexpr uint32_t kVectors = kE / 12;
        constexpr uint32_t kSplit = kHelpers / kVectors > 0 ? kHelpers / kVectors : 1;
        constexpr uint32_t kMaxSrc = (kR + kSplit - 1) / kSplit;
        for (uint32_t item = helper; item < kVectors * kSplit; item += kHelpers) {
            const uint32_t v = item % kVectors, part = item / kVectors;
            uint32_t srcs[kMaxSrc], nsrc = 0;
            for (uint32_t src = part; src < kR; src += kSplit)
                srcs[nsrc ++] = src;
            uint4 w[kMaxSrc];
            #pragma unroll
            for (uint32_t b = 0; b < kMaxSrc; ++ b)
                w[b] = b < nsrc ? __ldcg(table + srcs[b] * (kP / 8) + v) : make_uint4(taghi, taghi, taghi, taghi);
            // Check each word before combining: OR-ing raw tags can accept mixed old/new words when tag == 3.
            const auto stale = [&](const uint4& x) {
                return (((x.x ^ taghi) | (x.y ^ taghi) | (x.z ^ taghi) | (x.w ^ taghi)) &
                        (3u << 30)) != 0;
            };
            uint32_t pending = 0;
            #pragma unroll
            for (uint32_t b = 0; b < kMaxSrc; ++ b)
                pending |= stale(w[b]) ? (1u << b) : 0u;
            while (pending != 0) {
                // Back off between re-polls: hundreds of helper lanes spinning on ld.volatile
                // hammer the L2 slices the dispatch role's remote count/index writes land on.
                __nanosleep(192);
                #pragma unroll
                for (uint32_t b = 0; b < kMaxSrc; ++ b)
                    if (pending & (1u << b))
                        w[b] = ld_volatile_v4(table + srcs[b] * (kP / 8) + v);
                pending = 0;
                #pragma unroll
                for (uint32_t b = 0; b < kMaxSrc; ++ b)
                    pending |= stale(w[b]) ? (1u << b) : 0u;
            }
            if (stats != nullptr and lane == 0)
                stats[kP / kR + 80 + warp] = item * 64;
            uint32_t sum[12] = {};
            #pragma unroll
            for (uint32_t b = 0; b < kMaxSrc; ++ b) {
                const uint32_t ws[4] = {w[b].x, w[b].y, w[b].z, w[b].w};
                #pragma unroll
                for (uint32_t k = 0; k < 4; ++ k) {
                    sum[3 * k] += ws[k] & 0x3ffu;
                    sum[3 * k + 1] += (ws[k] >> 10) & 0x3ffu;
                    sum[3 * k + 2] += (ws[k] >> 20) & 0x3ffu;
                }
            }
            #pragma unroll
            for (uint32_t k = 0; k < 12; ++ k)
#if ECO_SPLIT_COUNTS
                atomicAdd(counts + v * 12 + k, sum[k]);
#else
                atomicAdd(sums + v * 12 + k, sum[k]);
#endif
        }
        if (stats != nullptr and lane == 0)
            stats[kP / kR + 80 + warp] = 0xffff;
        if (helper == 0)
            milestone(18);
#if !ECO_SPLIT_COUNTS
        helper_sync();
        for (uint32_t e = helper; e < kE; e += kHelpers)
            counts[e] = static_cast<uint16_t>(sums[e]);
        // plan_sync below publishes counts[] within the CTA. Fixed-slot counts and num_fixed are no longer
        // handed to other roles here: they use native receive counters and immutable launch metadata.
        if (helper == 0)
            milestone(19);
#endif
    }

    // ---- plan (warps 0..15) ------------------------------------------------------------------------------------
    plan_sync();
    if (tid == 0) {
#if ECO_SPLIT_COUNTS
        milestone(19);   // all totals, static tables and previous state have joined
#endif
        const auto input_time = trace_point(buffer, 4, &s_warm);
#if defined(ECO_TRACE) and ECO_TRACE
        const auto* slots = reinterpret_cast<const volatile unsigned long long*>(buffer + OFF_ECO_TRACE);
        const auto entry = static_cast<uint32_t>(slots[33]) % 64;
        reinterpret_cast<uint32_t*>(buffer + OFF_ECO_TRACE_RING)[entry * ECO_TRACE_RING_FIELDS + 7] =
            static_cast<uint32_t>(input_time - slots[24]);
#endif
    }
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
#if !UNIT_HYDRO
            if (warm and single and n > 0)   // warm start: the rank-time estimate T0, single-copy part
                atomicAdd(&T0[s / kS], pcost(y[s], pf[s], rm[s]));
#endif
        }
#if UNIT_HYDRO
        else {   // replicated slots start empty; the waterfill below fills them
            y[s] = 0; pf[s] = 0; q[s] = 0;
        }
#endif
    }
    if (tid == 0)
        milestone(31);
#if UNIT_HYDRO
    // ---- hydro plan: block-peak waterfill + relocate + bounded augmenting repair ---------------------------------
    // Bit-exact mirror of HydroPlanner.plan(n, refine=8, aug=64, dmax=8) (work/mpo/ep64/hydro.py):
    // fixed slots already hold their tokens; each replicated expert's units (full bm + one remainder)
    // go one-by-one to its least-loaded copy; cheap single moves off the peak; then multi-source BFS
    // alternating relocation chains push the residual peak to the per-call optimum.
    for (uint32_t r = tid; r < kR; r += kPlanThreads) {
        hload[r] = 0; ucnt[r] = 0;
    }
    if (tid == 0) {
        s_nm = 0; s_nu = 0;
        for (int d = 4; d < 19; ++ d)
            hsc[d] = 0;
    }
    plan_sync();
    for (uint32_t s = tid; s < kP; s += kPlanThreads)   // the immovable floor joins the rank loads
        if (clen[s] <= 1 and pf[s])
            atomicAdd(&hload[s / kS], y[s] + 1);
    // Replicated experts, hottest first (ties: first copy slot = phy appearance order). Warp0
    // compacts the movable list into hplaced (free until greedy) with a ballot prefix — expert
    // index order; then a single-key rank pass (count << 10 | 1023 - first copy slot) ranks each
    // only against the movable set — O(nm^2) cheap comparisons instead of O(E^2) loads
    if (warp == 0) {
        int nmv0 = 0;
        for (int e0 = 0; e0 < static_cast<int>(kE); e0 += 32) {
            const int e = e0 + static_cast<int>(lane);
            const bool mv = e < static_cast<int>(kE) and ecnt[e] >= 2 and counts[e] > 0;
            const uint32_t bal = __ballot_sync(~0u, mv);
            if (mv)
                hplaced[nmv0 + __popc(bal & ((1u << lane) - 1))] = static_cast<uint16_t>(e);
            nmv0 += __popc(bal);
        }
        if (lane == 0)
            s_nm = nmv0;
    }
    plan_sync();
    const int nmv = s_nm;
    if (warp == 1) {   // floor-only rank loads here: max and total feed the block-peak lower bound
        int fm = 0, fs = 0;
        for (int r = static_cast<int>(lane); r < static_cast<int>(kR); r += 32) {
            fm = max(fm, hload[r]); fs += hload[r];
        }
        for (int off = 16; off > 0; off >>= 1) {
            fm = max(fm, __shfl_xor_sync(~0u, fm, off)); fs += __shfl_xor_sync(~0u, fs, off);
        }
        if (lane == 0) { s_fmax = fm; s_fsum = fs; }
    }
    for (int f = static_cast<int>(tid); f < nmv; f += static_cast<int>(kPlanThreads)) {
        const int g = hplaced[f];
        oadd[f] = (UNIT_HORD ? (63 - ecnt[g]) << 23 : 0) | (counts[g] << 10) | (1023 - cslot[eoff[g]]);
    }
    plan_sync();
    for (int i = static_cast<int>(tid); i < nmv; i += static_cast<int>(kPlanThreads)) {
        const int e = hplaced[i], me = oadd[i];
        int rk = 0;
        for (int f = 0; f < nmv; ++ f)
            rk += oadd[f] > me;
        mord[rk] = e;
    }
    plan_sync();
    if (tid == 0)
        milestone(35);
    // ---- greedy waterfill, closed-form serial ---------------------------------------------------
    // Experts deal their units hottest-first to the least-loaded copy. The per-unit argmin
    // (load, slot) picks are exactly the nb smallest (level, slot) fill events — a copy's u-th win
    // lands at level L + u, and event (copy v, level lv) sits at output position
    // C(lv - 1) + #{copies with L < lv and smaller slot}, C(x) = sum(max(0, x - L_v)). So lane j
    // enumerates its own copy's wins with no warp reduce at all: positions strictly increase with
    // u and events past the water level land at position >= nb, which also bounds the loop.
    {
        const uint64_t a_t = (stats != nullptr and tid == 0) ? clock64() : 0;
        for (int i = static_cast<int>(tid); i < nmv; i += static_cast<int>(kPlanThreads)) {
            const int e = mord[i], oo = eoff[e], kk = ecnt[e];
            const int nbe = (counts[e] + kBM - 1) / kBM;
            uint64_t rmk = 0;
            for (int j = 0; j < kk; ++ j)
                rmk |= 1ull << (cslot[oo + j] / kS);
            erm[e] = __popcll(rmk) == kk ? rmk : 0ull;
            epk[i] = static_cast<int>(static_cast<uint32_t>(oo | (kk << 14) | (nbe << 20)) |
                                      (UNIT_HSPLIT ? static_cast<uint32_t>(counts[e] % nbe) << 26 : 0u));
        }
        plan_sync();
        // flatten the per-expert copy gathers once — the serial loop then reaches for the
        // next expert's copies with a single non-dependent load instead of an epk->slot
        // chain. The buffer holds one lane row per expert; experts past its capacity fall
        // back to the direct cslot gather (R=32 can see nmv > capacity)
        for (int x = static_cast<int>(tid); x < min(nmv, 2 * kP / 32) * 32;
             x += static_cast<int>(kPlanThreads)) {
            const int i = x >> 5, l = x & 31, p = epk[i];
            clane[x] = l < ((p >> 14) & 0x3f) ? cslot[(p & 0x3fff) + l] : -1;
        }
        if (stats != nullptr and tid == 0)
            hsc[15] = static_cast<int>(clock64() - a_t);
    }
    {
        const uint64_t b_t = (stats != nullptr and tid == 0) ? clock64() : 0;
        plan_sync();
        if (stats != nullptr and tid == 0)
            hsc[16] = static_cast<int>(clock64() - b_t);
    }
    if (warp == 0) {
        const uint64_t g_t = (stats != nullptr and lane == 0) ? clock64() : 0;
        volatile int* const hl = hload;
        int dst = 0;
        int p = epk[0];
        int cs = clane[lane];                        // prefetch distance 1
        const bool flat_all = nmv <= 2 * kP / 32;
#if UNIT_HPROF
        long long pa = 0, pb = 0, pc = 0, pd = 0, t0 = clock64();
#endif
        for (int i = 0; i < nmv; ++ i) {
            const int kk0 = (p >> 14) & 0x3f, nb0 = (p >> 20) & 0x3f;
            const int ex0 = UNIT_HSPLIT ? static_cast<int>(static_cast<uint32_t>(p) >> 26) : 0;
            int cs_n = -1, pn = p;
            if (flat_all) {                             // next expert's metadata + copies early, no branch
                const int i1 = min(i + 1, nmv - 1);
                pn = epk[i1];
                cs_n = clane[i1 * 32 + lane];
            } else if (i + 1 < nmv) {
                pn = epk[i + 1];
                cs_n = i + 1 < 2 * kP / 32 ? clane[(i + 1) * 32 + lane]
                       : (lane < ((pn >> 14) & 0x3f) ? cslot[(pn & 0x3fff) + lane] : -1);
            }
#if UNIT_HPROF
            const long long t1 = clock64(); pa += t1 - t0;
#endif
            const int L = cs >= 0 ? hl[cs / kS] : (1 << 20);
#if UNIT_HPROF
            const int Lw = __shfl_sync(~0u, L, 0);
            const long long t2 = clock64(); pb += t2 - t1 + (Lw & 0);
#endif
            const auto uflag = [&](const int& u) { return (UNIT_HSPLIT ? u < ex0 : u == nb0 - 1) ? 0x8000 : 0; };
            // closed form of the same picks: they are the nb smallest (level, slot) events; level t above
            // the lowest copy holds one event per copy at or below it, in slot = lane order. Eight
            // independent ballots give every level's events at once (registers only); the event (copy, t)
            // lands at output position cum(t) + #{lower lanes with an event at t} — the exact pick order
            bool dealt = false;
            if (nb0 >= UNIT_HCF) {   // warp-uniform: the ballots below run unconditionally (no per-vote divergence)
                const int d = L - __reduce_min_sync(~0u, L);
                uint32_t lm[kCfLv];
                int lc[kCfLv + 1], lam = kCfLv;
                lc[0] = 0;
#pragma unroll
                for (int t = 0; t < kCfLv; ++ t) {
                    lm[t] = __ballot_sync(~0u, d <= t);
                    lc[t + 1] = lc[t] + __popc(lm[t]);
                }
#pragma unroll
                for (int t = kCfLv - 1; t >= 0; -- t)
                    lam = lc[t + 1] >= nb0 ? t : lam;
                if (lam < kCfLv) {
                    const uint32_t lt = (1u << lane) - 1;
                    int w = 0;
                    uint16_t* const sink = reinterpret_cast<uint16_t*>(plist) + lane;   // free until build
#pragma unroll
                    for (int t = 0; t < kCfLv; ++ t) {   // branch-free: inactive events store to a private sink
                        const int pos = lc[t] + __popc(lm[t] & lt);
                        const bool ok = d <= t and t <= lam and pos < nb0;
                        *(ok ? hplaced + dst + pos : sink) = static_cast<uint16_t>(cs | uflag(pos));
                        w += ok;
                    }
                    if (cs >= 0)
                        hl[cs / kS] = L + w;
                    dealt = true;
                }
            }
            if (not dealt) {
                int ce = cs >= 0 ? (cs | (L << 10)) : 0x7fffffff;
                const uint32_t am = kk0 >= 32 ? ~0u : (1u << kk0) - 1;
                for (int u = 0; u < nb0; ++ u) {
                    const int w = __reduce_min_sync(am, ce);
                    if (ce == w) {
                        ce += 1 << 10;
                        // the winner emits its unit straight away — a fire-and-forget store that
                        // hides under the next reduction's latency, so no separate fill pass
                        hplaced[dst + u] = (w & 1023) | uflag(u);
                    }
                }
                if (cs >= 0)
                    hl[cs / kS] = ce >> 10;
            }
#if UNIT_HPROF
            __syncwarp();
            const long long t3 = clock64(); pc += t3 - t2;
            if (nb0 >= UNIT_HCF) pd += t3 - t2;
#endif
            cs = cs_n;
            p = pn;
            dst += nb0;
#if UNIT_HPROF
            t0 = clock64();
#endif
        }
#if UNIT_HPROF
        if (stats != nullptr and lane == 0) {
            hsc[14] = static_cast<int>(pa); hsc[15] = static_cast<int>(pb); hsc[16] = static_cast<int>(pc);
            hsc[18] = static_cast<int>(pd); hsc[19] = dst;
        }
#endif
        if (lane == 0)
            s_nu = dst;
        if (stats != nullptr and lane == 0)
            atomicAdd(&hsc[5], static_cast<int>(clock64() - g_t));
    }
    {
        const uint64_t b_t = (stats != nullptr and tid == 0) ? clock64() : 0;
        plan_sync();
        if (stats != nullptr and tid == 0) {
            hsc[17] = static_cast<int>(clock64() - b_t);
            milestone(36);
        }
    }
    plan_sync();
    // deferred quota/state flush: count each slot's wins from the unit table, then apply —
    // all plan threads instead of the serial loop's per-expert read-modify-writes
    {
        const int nu0 = min(static_cast<int>(s_nu), kMaxU);
        auto* const gct = oadd;                                  // count | partial flag << 30 (even: +1 units << 16)
        for (uint32_t s = tid; s < kP; s += kPlanThreads)
            gct[s] = 0;
        plan_sync();
        for (int i = static_cast<int>(tid); i < nu0; i += static_cast<int>(kPlanThreads)) {
            const int hp = hplaced[i];
            atomicAdd(&gct[hp & 0x7fff], 1 + ((hp & 0x8000) != 0 ? (UNIT_HSPLIT ? (1 << 16) : (1 << 30)) : 0));
        }
        plan_sync();
        for (uint32_t s = tid; s < kP; s += kPlanThreads) {
            const int gc = gct[s], got = UNIT_HSPLIT ? gc & 0xffff : gc & 0x3fffffff;
            if (UNIT_HSPLIT and got > 0) {
                const int ne = counts[phy[s]], nbs = (ne + kBM - 1) / kBM;
                q[s] += got * (ne / nbs) + (gc >> 16);
            } else if (got > 0) {
                const int ne = counts[phy[s]], nbs = (ne + kBM - 1) / kBM;
                if (gc & (1 << 30)) {   // this slot holds its expert's partial unit
                    y[s] += got - 1; pf[s] = 1;
                    q[s] += (got - 1) * kBM + ne - (nbs - 1) * kBM;
                } else {
                    y[s] += got; q[s] += got * kBM;
                }
            }
        }
    }
    plan_sync();
    if (warp == 0) {
        if (lane == 0)
            milestone(32);
        const int nu = min(static_cast<int>(s_nu), kMaxU);
        // nothing lowers the peak below LB = max(ceil(total / R), max floor): stop there
        const int lb = UNIT_HLB ? max((s_fsum + static_cast<int>(s_nu) + static_cast<int>(kR) - 1) / static_cast<int>(kR),
                                      static_cast<int>(s_fmax)) : -1;
        bool atlb = false;
        // ---- relocate: serial scan in placement order; every unit that can leave a peak rank does ---
        for (int pass = 0; pass < kHydRefine; ++ pass) {
            int top = 0;   // warp-max over the two ranks each lane covers
            for (int r = static_cast<int>(lane); r < static_cast<int>(kR); r += 32)
                top = max(top, hload[r]);
            for (int off = 16; off > 0; off >>= 1)
                top = max(top, __shfl_xor_sync(~0u, top, off));
            if (top <= lb) {
                atlb = true;
                break;
            }
            // ranks a unit may move to (load + 1 < top) as a bit mask: a unit's first such copy (ascending
            // slots = ascending ranks) is its copy-rank mask & free -> lowest bit, no per-copy loop
            const auto free_ranks = [&]() {
                const uint32_t f0 = __ballot_sync(~0u, static_cast<int>(lane) < static_cast<int>(kR) and hload[lane] + 1 < top);
                const uint32_t f1 = kR > 32 ? __ballot_sync(~0u, static_cast<int>(lane) + 32 < static_cast<int>(kR) and
                                                           hload[min(static_cast<int>(lane) + 32, static_cast<int>(kR) - 1)] + 1 < top) : 0u;
                return static_cast<uint64_t>(f0) | (static_cast<uint64_t>(f1) << 32);
            };
            uint64_t fr = free_ranks();
            int pos = 0, moved = 0;
            while (pos < nu) {
                const int i = pos + static_cast<int>(lane);
                int s = -1, s2 = -1;
                if (i < nu) {
                    s = hplaced[i] & 0x7fff;
                    if (hload[s / kS] == top) {
                        const int e = phy[s], oo = eoff[e];
                        const uint64_t em = erm[e];
                        if (em != 0) {
                            const uint64_t m = em & fr;
                            if (m != 0)
                                s2 = cslot[oo + __popcll(em & ((m & (~m + 1)) - 1))];
                        } else {
                            const int kk = ecnt[e];
                            for (int j = 0; j < kk; ++ j) {
                                const int c = cslot[oo + j];
                                if (hload[c / kS] + 1 < top) { s2 = c; break; }
                            }
                        }
                    }
                }
                const uint32_t bal = __ballot_sync(~0u, s2 >= 0);
                if (bal == 0) {
                    pos += 32;
                    continue;
                }
                const int fi = __ffs(bal) - 1, ai = pos + fi;
                const int as = __shfl_sync(~0u, s, fi), as2 = __shfl_sync(~0u, s2, fi);
                if (lane == 0) {
                    const bool part = (hplaced[ai] & 0x8000) != 0;
                    if (not UNIT_HSPLIT) {
                        if (part) { pf[as] = 0; pf[as2] = 1; }
                        else { y[as] -= 1; y[as2] += 1; }
                    }
                    const int amt = UNIT_HSPLIT ? ueven(as) + part : part ? rm[as] : kBM;
                    q[as] -= amt; q[as2] += amt;
                    hload[as / kS] -= 1; hload[as2 / kS] += 1;
                    hplaced[ai] = (hplaced[ai] & 0x8000) | as2;
                }
                __syncwarp();
                fr = free_ranks();
                moved = 1;
                pos = ai + 1;
            }
            if (__ballot_sync(~0u, moved) == 0)
                break;
        }
        if (lane == 0) {
            milestone(33);
            s_lb = lb;
        }
        // ---- per-rank unit positions as monotone append stamps (placement order at build; each
        //      relocate takes the next stamp so the order always matches the reference's
        //      remove + append). Parallel rank-grouped count, one pass. At LB the BFS never runs ---
        for (int r = static_cast<int>(lane); r < static_cast<int>(kR); r += 32)
            ucnt[r] = 0;
        __syncwarp();
        for (int i0 = 0; i0 < (atlb ? 0 : nu); i0 += 32) {
            const int i = i0 + static_cast<int>(lane);
            const bool act = i < nu;
            const int r = act ? (hplaced[i] & 0x7fff) / kS : -1;
            const uint32_t am = __ballot_sync(~0u, act);
            const uint32_t match = act ? __match_any_sync(am, r) : 0;
            const uint32_t lt = match & ((1u << lane) - 1);
            int cr = 0;
            if (act)
                cr = ucnt[r];                                    // pre-window count for my rank
            __syncwarp();
            if (act) {
                upos[i] = cr + __popc(lt);
                if (__ffs(match) - 1 == static_cast<int>(lane))
                    ucnt[r] = cr + __popc(match);                // leader commits the window's count
            }
            __syncwarp();
        }
        for (int r = static_cast<int>(lane); r < static_cast<int>(kR); r += 32)
            useq[r] = ucnt[r];
        __syncwarp();
        if (lane == 0)
            milestone(34);
    }
    plan_sync();
#if UNIT_HYDRO
    // ---- multi-source BFS, level-batched on ALL plan warps: edge keys (queue position, append
    //      stamp, copy, unit) order the atomicMins exactly like the reference's sequential pops;
    //      the least keyed terminal edge wins and the unit index inside the key is the payload.
    //      Warp 0 alone does the serial reduce/enqueue/apply -----------------------------------
    {
        const int nua = min(static_cast<int>(s_nu), kMaxU);
        volatile int& s_fr = hsc[2], & s_stop = hsc[3];    // terminal rank / empty-level flag
        volatile int& s_top = hsc[10], & s_fb = hsc[11], & s_qn = hsc[12];
        for (int it = 0; it < kHydAug; ++ it) {
            if (warp == 0) {
                int top = 0;
                for (int r = static_cast<int>(lane); r < static_cast<int>(kR); r += 32)
                    top = max(top, hload[r]);
                for (int off = 16; off > 0; off >>= 1)
                    top = max(top, __shfl_xor_sync(~0u, top, off));
                const int r0 = static_cast<int>(lane), r1 = r0 + 32;
                const bool src0 = r0 < kR and hload[r0] == top, src1 = r1 < kR and hload[r1] == top;
                const uint32_t sm0 = __ballot_sync(~0u, src0), sm1 = __ballot_sync(~0u, src1);
                if (r0 < kR) {   // sources = peak ranks, fseq = rank order among them
                    const int f0 = src0 ? __popc(sm0 & ((1u << lane) - 1)) : -1;
                    bpar[r0] = src0 ? -1 : -2; bdep[r0] = src0 ? 0 : -1;
                    fseq[r0] = f0; if (f0 >= 0) frk[f0] = r0;
                }
                if (r1 < kR) {
                    const int f1 = src1 ? __popc(sm0) + __popc(sm1 & ((1u << lane) - 1)) : -1;
                    bpar[r1] = src1 ? -1 : -2; bdep[r1] = src1 ? 0 : -1;
                    fseq[r1] = f1; if (f1 >= 0) frk[f1] = r1;
                }
                if (lane == 0) {
                    s_top = top; s_fr = -1; s_stop = 0;
                    s_fb = 0; s_qn = __popc(sm0) + __popc(sm1);
                    s_lbx = top <= s_lb;
                }
                __syncwarp();
                    {   // next level's prep (bedge reset + unvisited-rank mask), done here to save a block sync
                        const int q0 = static_cast<int>(lane), q1 = q0 + 32;
                        if (q0 < kR) bedge[q0] = ~0u;
                        if (q1 < kR) bedge[q1] = ~0u;
                        const uint32_t u0 = __ballot_sync(~0u, q0 < static_cast<int>(kR) and bpar[min(q0, static_cast<int>(kR) - 1)] == -2);
                        const uint32_t u1 = __ballot_sync(~0u, q1 < static_cast<int>(kR) and bpar[min(q1, static_cast<int>(kR) - 1)] == -2);
                        if (lane == 0) { hsc[28] = static_cast<int>(u0); hsc[29] = static_cast<int>(u1); }
                    }
            }
            plan_sync();
            if (s_lbx != 0)
                break;
            if (stats != nullptr and tid == 0)
                atomicAdd(&hsc[8], 1);
            for (int lvl = 0; lvl < kHydDepth and s_fr < 0; ++ lvl) {
                if (stats != nullptr and tid == 0)
                    atomicAdd(&hsc[9], 1);
                const int fb = s_fb, qn = s_qn;        // frontier = queue positions [fb, qn)
                // bedge reset + unvisited mask were prepared by warp 0 before the last block sync
                const uint64_t unv = static_cast<uint64_t>(static_cast<uint32_t>(hsc[28])) |
                                     (static_cast<uint64_t>(static_cast<uint32_t>(hsc[29])) << 32);
                uint64_t db_t = (stats != nullptr and tid == 0) ? clock64() : 0;
                // scatter: every warp strides all units and emits the frontier's edges. The key
                // packs the (queue position, append stamp, copy) order on top and the unit index
                // below — one u32 atomicMin orders the race AND carries the payload.
                for (int i = static_cast<int>(warp) * 32 + static_cast<int>(lane); i < nua; i += 512) {
                    const int s = hplaced[i] & 0x7fff, ra = s / kS;
                    const int fs = fseq[ra];
                    if (fs < fb or fs >= qn)
                        continue;
                    const int e = phy[s], oo = eoff[e], kk = ecnt[e];
                    const uint32_t pre = (static_cast<uint32_t>(fs) << 26) |
                                         (static_cast<uint32_t>(upos[i]) << 16) |
                                         static_cast<uint32_t>(i);
                    const uint64_t em = erm[e];
                    if (em != 0) {   // distinct copy ranks: copy index = # copy ranks below b
                        for (uint64_t m = em & unv & ~(1ull << ra); m; m &= m - 1) {
                            const int b = __ffsll(static_cast<long long>(m)) - 1;
                            atomicMin(&bedge[b], pre | (static_cast<uint32_t>(__popcll(em & ((1ull << b) - 1))) << 11));
                        }
                    } else {
                        for (int cj = 0; cj < kk; ++ cj) {
                            const int s2 = cslot[oo + cj], b = s2 / kS;
                            if (b != ra and bpar[b] == -2)
                                atomicMin(&bedge[b], pre | (static_cast<uint32_t>(cj) << 11));
                        }
                    }
                }
                plan_sync();
                if (stats != nullptr and tid == 0)
                    atomicAdd(&hsc[4], static_cast<int>(clock64() - db_t));
                if (warp == 0) {
                    const int top = s_top;
                    const int r0 = static_cast<int>(lane), r1 = r0 + 32;
                    const uint32_t e0 = r0 < kR ? bedge[r0] : ~0u, e1 = r1 < kR ? bedge[r1] : ~0u;
                    // parallel payload decode + BFS state commit: each lane resolves its own two
                    // ranks' winning keys, so the serial section below only orders the queue
                    const auto dec = [&](const uint32_t& k) {   // key -> (unit << 11) | slot
                        const int i = k & 0x7ff, cj = (k >> 11) & 0x1f;
                        return (i << 11) | cslot[eoff[phy[hplaced[i] & 0x7fff]] + cj];
                    };
                    if (e0 != ~0u) { bpar[r0] = frk[e0 >> 26]; bdep[r0] = lvl + 1; bparu[r0] = dec(e0); }
                    if (e1 != ~0u) { bpar[r1] = frk[e1 >> 26]; bdep[r1] = lvl + 1; bparu[r1] = dec(e1); }
                    uint32_t fmin = ~0u;
                    if (e0 != ~0u and hload[r0] <= top - 2) fmin = e0;
                    if (e1 != ~0u and hload[r1] <= top - 2) fmin = min(fmin, e1);
                    for (int off = 16; off > 0; off >>= 1)
                        fmin = min(fmin, __shfl_xor_sync(~0u, fmin, off));
                    if (fmin != ~0u) {   // terminal: the least keyed terminal edge wins
                        const bool ow0 = e0 == fmin, ow1 = e1 == fmin;
                        const uint32_t ow = __ballot_sync(~0u, ow0 or ow1);
                        s_fr = __shfl_sync(~0u, ow0 ? r0 : r1, __ffs(ow) - 1);
                    } else {
                        // enqueue the discovered ranks in flat edge order: position = # smaller keys
                        const bool d0 = e0 != ~0u, d1 = e1 != ~0u;
                        const uint32_t act = __ballot_sync(~0u, d0 or d1);
                        const uint32_t bth = __ballot_sync(~0u, d0 and d1);
                        if (act == 0) {
                            s_stop = 1;
                        } else {
                            const int qn0 = qn;
                            // position = # smaller keys over all 64 candidates (keys are unique; ~0u = none):
                            // a fixed 32-step shuffle sweep instead of a dependent shfl+ballot chain per rank
                            int p0 = 0, p1 = 0;
#pragma unroll 8
                            for (int j = 0; j < 32; ++ j) {
                                const uint32_t ka = __shfl_sync(~0u, e0, j), kb = __shfl_sync(~0u, e1, j);
                                p0 += static_cast<int>(ka < e0) + static_cast<int>(kb < e0);
                                p1 += static_cast<int>(ka < e1) + static_cast<int>(kb < e1);
                            }
                            if (d0) { fseq[r0] = qn0 + p0; frk[qn0 + p0] = r0; }
                            if (d1) { fseq[r1] = qn0 + p1; frk[qn0 + p1] = r1; }
                            if (lane == 0) {
                                s_fb = qn0;
                                s_qn = qn0 + __popc(act) + __popc(bth);
                            }
                        }
                    }
                    __syncwarp();
                    {   // next level's prep (bedge reset + unvisited-rank mask), done here to save a block sync
                        const int q0 = static_cast<int>(lane), q1 = q0 + 32;
                        if (q0 < kR) bedge[q0] = ~0u;
                        if (q1 < kR) bedge[q1] = ~0u;
                        const uint32_t u0 = __ballot_sync(~0u, q0 < static_cast<int>(kR) and bpar[min(q0, static_cast<int>(kR) - 1)] == -2);
                        const uint32_t u1 = __ballot_sync(~0u, q1 < static_cast<int>(kR) and bpar[min(q1, static_cast<int>(kR) - 1)] == -2);
                        if (lane == 0) { hsc[28] = static_cast<int>(u0); hsc[29] = static_cast<int>(u1); }
                    }
                }
                plan_sync();
                if (stats != nullptr and tid == 0)
                    atomicAdd(&hsc[6], static_cast<int>(clock64() - db_t));
                if (s_fr >= 0 or s_stop != 0)
                    break;
            }
            if (s_fr < 0)
                break;
            const uint64_t db3 = (stats != nullptr and tid == 0) ? clock64() : 0;
            if (warp == 0 and lane == 0) {   // apply the chain back from the terminal
                int b = s_fr;
                while (bpar[b] != -1) {
                    const int pk = bparu[b];
                    const int i = pk >> 11, s2 = pk & 0x7ff;
                    const int s = hplaced[i] & 0x7fff, a2 = s / kS;
                    const bool part = (hplaced[i] & 0x8000) != 0;
                    if (not UNIT_HSPLIT) {
                        if (part) { pf[s] = 0; pf[s2] = 1; }
                        else { y[s] -= 1; y[s2] += 1; }
                    }
                    const int amt = UNIT_HSPLIT ? ueven(s) + part : part ? rm[s] : kBM;
                    q[s] -= amt; q[s2] += amt;
                    hload[a2] -= 1; hload[b] += 1;
                    hplaced[i] = (hplaced[i] & 0x8000) | s2;
                    // the append stamp keeps the within-rank edge order identical to the
                    // reference's remove + append without maintaining any list structure
                    upos[i] = useq[b]; ++ useq[b];
                    b = bpar[b];
                }
            }
            plan_sync();
            if (stats != nullptr and tid == 0)
                atomicAdd(&hsc[7], static_cast<int>(clock64() - db3));
        }
        if (tid == 0) {
            milestone(30);
            trace_point(buffer, 17);
            if (stats != nullptr) {
                for (int d = 0; d < 6; ++ d)
                    stats[4030 + d] = hsc[4 + d];
                stats[4036] = s_nu;
                stats[4037] = s_nm;
                for (int d = 0; d < 6; ++ d)
                    stats[4038 + d] = hsc[14 + d];
            }
        }
    }
#endif
    plan_sync();
    if (tid == 0) {
        milestone(21);
        trace_point(buffer, 18);
    }
#else
    // Small replicated experts: one lane each, striding when there are more than kPlanThreads; big ones: a warp
    // each (lane = copy). A lane replays its whole expert locally — the planning phase uses no collectives.
    int bp[kBigPerWarp], bq[kBigPerWarp], bk[kBigPerWarp], btot[kBigPerWarp];
    bool bvalid[kBigPerWarp], bon[kBigPerWarp];
    // Hand big experts to the warps the small experts leave idle (small experts sit one per lane, on the
    // first ceil(NS/32) warps), so the big-expert work overlaps the small-expert passes instead of
    // stacking behind them.
    const int bw = (NS + 31) / 32;
    #pragma unroll
    for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
        const int g = (static_cast<int>(warp) - bw + static_cast<int>(kPlanWarps)) % static_cast<int>(kPlanWarps) +
                      static_cast<int>(kPlanWarps) * i;
        bvalid[i] = g < NB;
        const int e = bvalid[i] ? big_ids[g] : 0;
        bk[i] = bvalid[i] ? ecnt[e] : 0;
        bon[i] = static_cast<int>(lane) < bk[i];
        bp[i] = bon[i] ? cslot[eoff[e] + lane] : 0;
        btot[i] = bvalid[i] ? counts[e] : 0;
    }
    if (warm) {
        // Unit-space warm start (`UnitPlanner._warm`): (1) the replicated slots' warm unit split was
        // assigned by the helpers during the count-row window (previous call's quotas and counts only);
        // here the experts with no tokens this call are cleared and the per-copy rank-time estimates
        // fold into T0; (2) the experts whose unit count changed add units to the copies with the lowest
        // estimate (+ own additions) or remove them from the highest (+ own removals); the partial stays
        // with its owner while it holds a unit, else goes to the holding copy with the highest estimate
        constexpr int kUnit = kWB + kBM;
        for (uint32_t s = tid; s < kP; s += kPlanThreads) {
            if (clen[s] > 1) {
                if (counts[phy[s]] == 0) {
                    y[s] = 0; pf[s] = 0;
                } else {
                    atomicAdd(&T0[s / kS], pcost(y[s], pf[s], rm[s]));
                }
            }
        }
        plan_sync();
        if (tid == 0)
            milestone(32);
        // The picks are the delta smallest elements of the copies' merged arithmetic sequences
        // (tr + i*kUnit, lowest index on ties): closed form via the threshold T*. Each lane replays its whole
        // expert locally — loads all copies from shared memory, then straight-line math with no loop.
        for (int g = tid; g < NS; g += kPlanThreads) {
            const int se = small_ids[g], sk = ecnt[se], so = eoff[se], stot = counts[se];
            int s4[4], trj[4], suj[4], sown = -1;
            #pragma unroll
            for (int j = 0; j < 4; ++ j) {
                s4[j] = (j < sk) ? cslot[so + j] : -1;
                const bool on = s4[j] >= 0;
                trj[j] = on ? T0[s4[j] / kS] : 0;
                suj[j] = on ? y[s4[j]] + (pf[s4[j]] != 0 ? 1 : 0) : 0;
                if (on and pf[s4[j]] and sown < 0)
                    sown = j;
            }
            const int delta = ((stot > 0) ? (stot + kBM - 1) / kBM : 0) - (suj[0] + suj[1] + suj[2] + suj[3]);
            const bool pos = delta > 0, neg = delta < 0;
            // Copy j's elements live on levels m = trj/kUnit + i (residue rj = trj - n*kUnit orders
            // same-level picks). Both deltas are closed forms over the level domain — no search loop.
            int nn[4], rr[4];
            #pragma unroll
            for (int j = 0; j < 4; ++ j) {
                nn[j] = static_cast<int>(static_cast<uint32_t>(trj[j]) / kUnit);
                rr[j] = trj[j] - nn[j] * kUnit;
            }
            int ap[4] = {0, 0, 0, 0}, an[4] = {0, 0, 0, 0};
            if (pos) {
                // smallest m with count_below(m+1) = sum_j max(0, m+1-n_j) >= delta. C(X) (X = m + 1) is
                // piecewise linear: with exactly the t smallest levels a < X it reads t X - P_t, so the
                // crossing X* = ceil((delta + P_t) / t) is a closed form per segment — no search loop.
                int a0 = 0 < sk ? nn[0] : 0x3fffffff, a1 = 1 < sk ? nn[1] : 0x3fffffff,
                    a2 = 2 < sk ? nn[2] : 0x3fffffff, a3 = 3 < sk ? nn[3] : 0x3fffffff;
                if (a0 > a1) { const int t = a0; a0 = a1; a1 = t; }
                if (a2 > a3) { const int t = a2; a2 = a3; a3 = t; }
                if (a0 > a2) { const int t = a0; a0 = a2; a2 = t; }
                if (a1 > a3) { const int t = a1; a1 = a3; a3 = t; }
                if (a1 > a2) { const int t = a1; a1 = a2; a2 = t; }
                const int p2 = a0 + a1, p3 = p2 + a2, p4 = p3 + a3;
                int X = (delta + p4 + 3) / 4;   // the last segment (t = sk) always holds the crossing
                if (sk == 3)
                    X = (delta + p3 + 2) / 3;
                else if (sk == 2)
                    X = (delta + p2 + 1) / 2;
                else if (sk == 1)
                    X = delta + a0;
                // a smaller segment wins the crossing iff its own candidate lands inside it
                if (sk > 3) {
                    const int x3 = (delta + p3 + 2) / 3;
                    if (x3 >= a2 and x3 < a3)
                        X = x3;
                }
                if (sk > 2) {
                    const int x2 = (delta + p2 + 1) / 2;
                    if (x2 >= a1 and x2 < a2)
                        X = x2;
                }
                if (delta + a0 < a1)
                    X = delta + a0;
                const int L = X - 1;
                int cbe = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++ j)
                    cbe += (j < sk) ? max(0, L - nn[j]) : 0;
                const int rho = delta - cbe - 1;
                #pragma unroll
                for (int j = 0; j < 4; ++ j) {
                    if (j < sk and nn[j] <= L) {
                        int rk = 0;
                        #pragma unroll
                        for (int i = 0; i < 4; ++ i)
                            rk += (i != j and i < sk and nn[i] <= L and (rr[i] < rr[j] or (rr[i] == rr[j] and i < j)));
                        ap[j] = L - nn[j] + (rk <= rho);
                    }
                }
            } else if (neg) {
                const int D = -delta;
                // count_ge(m) = sum_j min(su_j, n_j-m+1)^+ is decreasing piecewise-linear in m with
                // breaks only at n_j and d_j = n_j-su_j+1. lo = largest probed n_j with count_ge >= D
                // brackets M; inside the bracket a linear solve plus at-most-3 straight-line
                // saturation corrections pin M exactly — again no loop.
                int g0 = 0, g1 = 0, g2 = 0, g3 = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++ j) {
                    g0 += max(0, min(suj[j], nn[j] - nn[0] + 1));
                    g1 += max(0, min(suj[j], nn[j] - nn[1] + 1));
                    g2 += max(0, min(suj[j], nn[j] - nn[2] + 1));
                    g3 += max(0, min(suj[j], nn[j] - nn[3] + 1));
                }
                int lo = -0x40000000, sat = 0, M;
                lo = max(lo, g0 >= D ? nn[0] : -0x40000000);
                lo = max(lo, g1 >= D ? nn[1] : -0x40000000);
                lo = max(lo, g2 >= D ? nn[2] : -0x40000000);
                lo = max(lo, g3 >= D ? nn[3] : -0x40000000);
                M = lo;
                #pragma unroll
                for (int rep = 0; rep < 4; ++ rep) {   // copies above lo+1 contribute linearly;
                    int S = 0, Pn = 0, u = 0;          // ones whose d_j exceeds the solve saturate
                    #pragma unroll
                    for (int j = 0; j < 4; ++ j) {
                        const bool lin = j < sk and nn[j] >= lo + 1 and !((sat >> j) & 1);
                        S += ((sat >> j) & 1) ? suj[j] : 0;
                        Pn += lin ? nn[j] + 1 : 0;
                        u += lin;
                    }
                    if (u == 0)
                        break;
                    const int v = Pn + S - D;   // floor division (v may be negative)
                    const int m0 = v >= 0 ? v / u : -((-v + u - 1) / u);
                    if (m0 <= lo)
                        break;
                    int vio = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++ j)
                        vio |= (j < sk and nn[j] >= lo + 1 and !((sat >> j) & 1) and nn[j] - suj[j] + 1 > m0) ? (1 << j) : 0;
                    if (vio == 0) {
                        M = m0;
                        break;
                    }
                    sat |= vio;
                }
                int cab = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++ j)
                    cab += (j < sk) ? min(suj[j], max(0, nn[j] - M)) : 0;
                const int rho = D - cab;
                #pragma unroll
                for (int j = 0; j < 4; ++ j) {
                    if (j < sk) {
                        an[j] = min(suj[j], max(0, nn[j] - M));
                        if (M <= nn[j] and nn[j] - suj[j] + 1 <= M) {
                            int rk = 0;
                            #pragma unroll
                            for (int i = 0; i < 4; ++ i)
                                rk += (i != j and i < sk and M <= nn[i] and nn[i] - suj[i] + 1 <= M and
                                       (rr[i] > rr[j] or (rr[i] == rr[j] and i < j)));
                            an[j] += rk < rho;
                        }
                    }
                }
            }
            int sun[4];
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                sun[j] = suj[j] + ap[j] - an[j];
            const bool lost = stot > 0 and
                              not ((sown == 0 and sun[0] > 0) or (sown == 1 and sun[1] > 0) or
                                   (sown == 2 and sun[2] > 0) or (sown == 3 and sun[3] > 0));
            uint32_t pk = 0;
            pk = max(pk, (0 < sk and sun[0] > 0) ? (static_cast<uint32_t>(trj[0]) << 2 | 3u) : 0u);
            pk = max(pk, (1 < sk and sun[1] > 0) ? (static_cast<uint32_t>(trj[1]) << 2 | 2u) : 0u);
            pk = max(pk, (2 < sk and sun[2] > 0) ? (static_cast<uint32_t>(trj[2]) << 2 | 1u) : 0u);
            pk = max(pk, (3 < sk and sun[3] > 0) ? (static_cast<uint32_t>(trj[3]) << 2 | 0u) : 0u);
            if (lost)   // partial's owner lost all its units: it goes to the holding copy with the highest estimate
                sown = 3 - static_cast<int>(pk & 3u);
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                if (s4[j] >= 0) {
                    y[s4[j]] = sun[j] - (j == sown); pf[s4[j]] = j == sown;
                }
        }
        if (tid == 0)
            milestone(37);
        if (stats != nullptr and lane == 0)
            atomicMax(stats + kP / kR + 52 + warp, static_cast<int>((clock64() - t_entry) / 16));
        // No barrier here: big-expert slots are disjoint from the small experts' and live on the warps
        // the small pass left idle, so the two passes run concurrently (same split as the cold path).
        if (tid == 0)
            milestone(34);
        #pragma unroll
        for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
            if (not bvalid[i] or btot[i] == 0)
                continue;
            const int tr = bon[i] ? T0[bp[i] / kS] : 0;
            // Units per copy incl. the partial, and the partial's owner, re-read from the
            // helpers' warm assignment (y/pf).
            int add = 0, u = bon[i] ? y[bp[i]] + pf[bp[i]] : 0;
            const uint32_t obal = __ballot_sync(~0u, bon[i] and pf[bp[i]] != 0);
            int own = obal ? static_cast<int>(__ffs(obal)) - 1 : -1;
            int delta = (btot[i] + kBM - 1) / kBM - __reduce_add_sync(~0u, bon[i] ? u : 0);
            // Same closed form as the small experts: the picks are the delta smallest (largest, when removing)
            // elements of the lanes' merged arithmetic sequences, the threshold found by binary search
            if (delta > 0) {
                int lo = __reduce_min_sync(~0u, bon[i] ? tr : 0x7fffffff), hi = lo + delta * kUnit;
                while (lo < hi) {
                    const int mid = lo + (hi - lo) / 2;
                    if (__reduce_add_sync(~0u, (bon[i] and tr <= mid) ? static_cast<uint32_t>(mid - tr) / kUnit + 1 : 0) >= delta)
                        hi = mid;
                    else
                        lo = mid + 1;
                }
                const int tt = lo;
                add = (bon[i] and tr <= tt - 1) ? static_cast<uint32_t>(tt - 1 - tr) / kUnit + 1 : 0;
                int rem = delta - __reduce_add_sync(~0u, add);
                const uint32_t bal = __ballot_sync(~0u, bon[i] and tr + add * kUnit == tt);
                if (bon[i] and tr + add * kUnit == tt and static_cast<int>(__popc(bal & ((1u << lane) - 1))) < rem)
                    ++ add;
                u += add;
            } else if (delta < 0) {
                delta = -delta;
                int lo = __reduce_min_sync(~0u, (bon[i] and u > 0) ? tr - (u - 1) * kUnit : 0x7fffffff);
                int hi = __reduce_max_sync(~0u, (bon[i] and u > 0) ? tr : 0);
                while (lo < hi) {
                    const int mid = lo + (hi - lo + 1) / 2;
                    if (__reduce_add_sync(~0u, (bon[i] and u > 0 and tr >= mid) ? min(u, static_cast<int>(static_cast<uint32_t>(tr - mid) / kUnit + 1)) : 0) >=
                        delta)
                        lo = mid;
                    else
                        hi = mid - 1;
                }
                const int tt = lo;
                add = (bon[i] and u > 0 and tr >= tt + 1) ? min(u, static_cast<int>(static_cast<uint32_t>(tr - tt - 1) / kUnit + 1)) : 0;
                int rem = delta - __reduce_add_sync(~0u, add);
                const uint32_t bal = __ballot_sync(~0u, bon[i] and add < u and tr - add * kUnit == tt);
                if (bon[i] and add < u and tr - add * kUnit == tt and
                    static_cast<int>(__popc(bal & ((1u << lane) - 1))) < rem)
                    ++ add;
                u -= add;
            }
            if (not __any_sync(~0u, static_cast<int>(lane) == own and u > 0)) {
                const uint32_t mk = __reduce_max_sync(~0u, (bon[i] and u > 0) ? ((static_cast<uint32_t>(tr) << 5) | (31 - lane)) : 0u);
                own = 31 - static_cast<int>(mk & 31);
            }
            if (bon[i]) {
                y[bp[i]] = u - (static_cast<int>(lane) == own); pf[bp[i]] = static_cast<int>(lane) == own;
            }
        }
        plan_sync();
        if (tid == 0)
            milestone(33);
    } else {   // cold: an even split rounded to units
        for (int g = tid; g < NS; g += kPlanThreads) {
            const int se = small_ids[g], sk = ecnt[se], so = eoff[se], stot = counts[se];
            const int b0 = sk ? stot / sk : 0;
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                if (j < sk)
                    q[cslot[so + j]] = b0 + (j == 0 ? stot - b0 * sk : 0);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kBigPerWarp; ++ i) {
            bq[i] = 0;
            if (bon[i]) {
                const int bb = btot[i] / bk[i];
                bq[i] = bb + (lane == 0 ? btot[i] - bb * bk[i] : 0);
            }
        }
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
        if (stats != nullptr) {
            if (tid < kR)
                stats[kP / kR + 160 + tid] = T0[tid];
            for (uint32_t s = tid; s < kP; s += kPlanThreads)
                stats[kP / kR + 3000 + s] = q[s];
        }
        // Load-aware rounding to units: round up the copies with the smallest (rank time * bm - leftover * (wb + bm));
        // the partial unit goes to the rounded-up copy with the smallest leftover
        for (int g = tid; g < NS; g += kPlanThreads) {
            const int se = small_ids[g], sk = ecnt[se], so = eoff[se], stot = counts[se];
            int s4[4], sqv[4], key[4], g2[4];
            const int kk = (stot + kBM - 1) / kBM;
            int gsum = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++ j) {
                s4[j] = (j < sk) ? cslot[so + j] : -1;
                sqv[j] = s4[j] >= 0 ? q[s4[j]] : 0;
                key[j] = s4[j] >= 0 ? T0[s4[j] / kS] * kBM - (sqv[j] % kBM) * (kWB + kBM) : 0x7fffffff;
                g2[j] = s4[j] >= 0 ? sqv[j] / kBM : 0;
                gsum += g2[j];
            }
            const int left = kk - gsum;
            uint32_t pk = 0;
            int gp[4];
            #pragma unroll
            for (int j = 0; j < 4; ++ j) {
                int rk = 0;
                #pragma unroll
                for (int j2 = 0; j2 < 4; ++ j2)
                    rk += j2 < sk and (key[j2] < key[j] or (key[j2] == key[j] and j2 < j));
                gp[j] = g2[j] + (s4[j] >= 0 and rk < left);
                if (s4[j] >= 0 and gp[j] > 0)
                    pk = max(pk, (static_cast<uint32_t>(gp[j] * kBM - sqv[j] + (1 << 22)) << 2) | (3u - static_cast<uint32_t>(j)));
            }
            const int jp = pk ? 3 - static_cast<int>(pk & 3u) : -1;
            #pragma unroll
            for (int j = 0; j < 4; ++ j)
                if (s4[j] >= 0) {
                    y[s4[j]] = gp[j] - (j == jp); pf[s4[j]] = j == jp;
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
    if (tid == 0) {
        milestone(30);
        trace_point(buffer, 17);
    }

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
#endif

    // ---- publish: quotas, canonical starts within each expert's copies (ascending slots) -------------------------
    // Quotas (full units * bm + the partial unit) computed where they are read: no rewrite pass and barrier
#if UNIT_HYDRO
    const auto quota = [&](const int& c) { return q[c]; };   // hydro keeps exact per-slot token quotas
#else
    const auto quota = [&](const int& c) { return y[c] * kBM + pf[c] * rm[c]; };
#endif
    auto* out_quota = signals + OFF_ECO_QUOTA / 4;
    auto* out_start = signals + OFF_ECO_QUOTA_START / 4;
    // Banked by call parity: bank[epoch&1] holds this call's plan while
    // bank[epoch&1^1] keeps the previous one readable for speculation.
    const uint32_t kQBank = ECO_QBANK ? uint32_t(kP) : 0u;
    out_quota += (epoch & 1u) * kQBank;
    out_start += (epoch & 1u) * kQBank;
    for (uint32_t p = tid; p < kP; p += kPlanThreads) {
        const int o = cst[p], k = ecnt[phy[p]];
        int start = 0;
        // active copies ascend: sum the copies below p (passive copies hold nothing). No early exit, so
        // the independent loads pipeline instead of one dependent slot+quota round trip per copy
#pragma unroll 4
        for (int j = 0; j < k; ++ j) {
            const int c = cslot[o + j];
            start += c < static_cast<int>(p) ? quota(c) : 0;
        }
        out_quota[p] = static_cast<uint32_t>(quota(static_cast<int>(p)));
        out_start[p] = static_cast<uint32_t>(start);
    }
    plan_sync();
    if (tid == 0) {
        milestone(40);
        __threadfence();
        milestone(41);
        st_release_gpu(signals + OFF_ECO_PLAN_READY / 4, epoch + 1);
        const auto publish_time = trace_point(buffer, 5);
        milestone(20);
#if defined(ECO_TRACE) and ECO_TRACE
        auto* slots = reinterpret_cast<unsigned long long*>(buffer + OFF_ECO_TRACE);
        const auto since = publish_time - *reinterpret_cast<volatile unsigned long long*>(slots + 24);
        atomicAdd(slots + 25, since);
        const auto entry = static_cast<uint32_t>(*reinterpret_cast<volatile unsigned long long*>(slots + 33)) % 64;
        reinterpret_cast<uint32_t*>(buffer + OFF_ECO_TRACE_RING)[entry * ECO_TRACE_RING_FIELDS + 3] = static_cast<uint32_t>(since);
#endif
    }
#if ECO_COMBINE_EXACT
    // ---- exact combine readiness (off the critical path): every rank's plan is identical, so this rank derives
    // which rank computes each of its own routes. Route (token, k) to logical expert e sits at slot j of this
    // rank's row in e's canonical order (recorded by role1); its canonical position is
    // p = sum_s min(c_s, j) + #{s < rank : c_s > j} (round-robin over sources, ascending rank), computed by the
    // active copy whose [start, start + quota) holds p. Result overwrites j; role3 waits for that rank only.
    if (tid < kPlanThreads) {
        const uint32_t lane = tid & 31u, warp = tid >> 5;
        const auto* done = reinterpret_cast<const volatile uint32_t*>(signals + OFF_ECO_ROUTES_DONE / 4);
        for (uint32_t b = lane; b < ECO_DISPATCH_SMS; b += 32)
            while (done[b] != epoch + 1u) {}
        __syncwarp();
        __threadfence();
        auto* route_peer = reinterpret_cast<volatile uint32_t*>(signals + OFF_ECO_ROUTE_PEER / 4);
        const auto* ctab = reinterpret_cast<const volatile uint32_t*>(buffer + OFF_ECO_COUNTS + gen * kR * kP * 2u);
        const auto* xroutes = reinterpret_cast<const int64_t*>(buffer + OFF_TOPK);
        for (uint32_t r = warp; r < num_tokens * ECO_TOPK; r += kPlanWarps) {
            const int64_t e = xroutes[r];
            if (e < 0 or e >= static_cast<int64_t>(kE))
                continue;
            const int k = ecnt[e], o = eoff[e];
            int dst = cslot[o];
            if (k > 1) {
                const uint32_t j = route_peer[r];
                uint32_t part = 0;
                for (uint32_t s = lane; s < kR; s += 32) {
                    const uint32_t c = (ctab[s * (kP / 2) + static_cast<uint32_t>(e) / 3u] >>
                                        ((static_cast<uint32_t>(e) % 3u) * 10u)) & 0x3ffu;
                    part += min(c, j) + ((s < rank and c > j) ? 1u : 0u);
                }
                const uint32_t pos = __reduce_add_sync(0xffffffffu, part);
                int hit = -1;
                for (int jj = static_cast<int>(lane); jj < k; jj += 32) {
                    const int c = cslot[o + jj];
                    const uint32_t st = out_start[c], q = out_quota[c];
                    if (pos >= st and pos < st + q)
                        hit = c;
                }
                dst = __reduce_max_sync(0xffffffffu, hit);
            }
            if (lane == 0)
                route_peer[r] = static_cast<uint32_t>(dst) / kS;
        }
    }
#endif
    // ---- next call's warm state (after the release, off the critical path): this plan's quotas of the replicated
    // slots, then the counts
    if (state != nullptr) {
        for (uint32_t p = tid; p < kP; p += kPlanThreads)
            state[p] = static_cast<uint16_t>(clen[p] >= 2 ? quota(static_cast<int>(p)) : 0);
        for (uint32_t e = tid; e < kE; e += kPlanThreads)
            state[kP + e] = counts[e];
    }
#if defined(ECO_TRACE) and ECO_TRACE
    plan_sync();
    if (tid == 0) {   // ring field 10: plan kernel exit (ns since this call's role1 start)
        const auto exit_time = trace_point(buffer, 43);
        const auto* slots = reinterpret_cast<const volatile unsigned long long*>(buffer + OFF_ECO_TRACE);
        const auto entry = static_cast<uint32_t>(slots[33]) % 64;
        atomicMax(reinterpret_cast<uint32_t*>(buffer + OFF_ECO_TRACE_RING) + entry * ECO_TRACE_RING_FIELDS + 10,
                  static_cast<uint32_t>(exit_time - slots[24]));
    }
#endif
}

// Auction rounds of the plan role (included inside eco_plan_kernel); mirrors `UnitPlanner._rounds` /
// `_auction_round` / `_accept` (gc_moe/eco/unit_ref.py). Per round,
// against the round's rank times T (mean mu = tot / R):
//   (A)  two-hop rounds only: every rank's best one-hop outflow (receiver time << 15 | pair index), decoded;
//   (C1) a thread per pair out of a replicated slot of a source rank (T > mu): its best one- or two-hop unit path
//        whose touched ranks all end below the source's time; per slot the min key (max(source after, value),
//        value, hops, s1, s0) and its shed (source time decrease);
//   (C2) offered while the shed of the source's smaller keys is below its excess: offer key (receiver, source time
//        desc, value, s0, s1), the receiver's time increase and the moved unit's type; two-hop offers: per
//        intermediate the min (source time desc, s0);
//   (F1) offers grouped by receiver; two-hop offers that are not their intermediate's min, and of a transit slot and
//        its own offer the lower (source time desc, s0) one, dropped -> the kept offers' slots are distinct;
//   (F2) accepted while the receiver stays below the source's time and mu + slack with all its smaller kept offers
//        (prefix); accepted offers move their units at once (typed moves; exact rank time deltas, the slot unit
//        counts' atomics telling the 0 <-> 1 activation changes, order independent).
// The first UNIT_HOP2 rounds are two-hop rounds, the rest one-hop rounds (no outflows, no drops).
{
    static constexpr int kSlack = kWB + kBM + kWA;
    static constexpr uint32_t kMaxV = (1u << 18) - 1;
    static_assert(kP <= 2048 and kR <= 64, "Auction keys: 11-bit slots, 6-bit ranks");
    uint64_t* const okey = ogids;
    uint32_t* const sclaim = reinterpret_cast<uint32_t*>(q);
    struct Out { int k2, e1, r2, s1b, s2b, d1, in2, ip; };
    const auto outflow = [&](const int& r) {   // rank r's best one-hop outflow, decoded in (A) (r2 < 0: none)
        Out o;
        o.r2 = dr2[r]; o.k2 = dk2[r]; o.e1 = de1[r]; o.s1b = ds1b[r]; o.s2b = ds2b[r]; o.d1 = dd1[r]; o.in2 = din2[r]; o.ip = dip[r];
        return o;
    };
    const auto mv = [&](const int& a, const int& b, const int& part) {   // typed unit move, exact rank time deltas
        const int du = part ? kWB + rm[a] : kWB + kBM;
        if (part) {
            pf[a] = 0; pf[b] = 1;
        } else {
            atomicSub(&y[a], 1); atomicAdd(&y[b], 1);
        }
        const int oa = atomicSub(&cnt[a], 1), ob = atomicAdd(&cnt[b], 1);
        atomicAdd(&T[a / kS], -du - (oa == 1 ? kWA : 0));
        atomicAdd(&T[b / kS], du + (ob == 0 ? kWA : 0));
    };
    if (tid < kR)   // per rank: its first active replicated slot (pairs ascend by slot)
        rfirst[tid] = roff[tid] < roff[tid + 1] ? static_cast<int>(plist[roff[tid]] >> 16) : static_cast<int>((tid + 1) * kS);
    const int nr_cap = warm ? UNIT_NR : UNIT_NRC;
    // Round schedule: the first UNIT_HOP2 rounds two-hop (sparse rank graphs need transit paths), the rest one-hop
    bool stop = false;
    int k = 0;
    for (; k < nr_cap and k < UNIT_HOP2; ++ k) {
        // (A)
        if (tid < kR) {
            imin[tid] = ~0u; rcnt[tid] = 0; rfill[tid] = 0; Ts[tid] = T[tid];
        }
        for (uint32_t i = tid; i < kP; i += kPlanThreads)
            ckey[i] = ~0ull;   // also holds the previous round's grouped offers
        if (tid == 0)
            s_noff = 0;
        for (uint32_t i = tid; i < kP; i += kPlanThreads)
            sclaim[i] = 0u;   // per slot: offered this round
        {   // warp w owns ranks [w kRW, (w + 1) kRW): one strided loop over their pairs, per-rank minima; lane i decodes
            static constexpr uint32_t kRW = (kR + kPlanWarps - 1) / kPlanWarps;
            const uint32_t rlo = warp * kRW, rhi = min(rlo + kRW, kR);
            uint32_t mins[kRW];
            #pragma unroll
            for (uint32_t i = 0; i < kRW; ++ i)
                mins[i] = ~0u;
            if (rlo < rhi) {
                const int pb0 = roff[rlo], pb1 = roff[rhi];
                for (int p = pb0 + static_cast<int>(lane); p < pb1; p += 32) {
                    const uint32_t pr = plist[p];
                    const int s1 = static_cast<int>(pr >> 16), s2 = static_cast<int>(pr & 0xffffu);
                    const int ip = pf[s1];
                    if (y[s1] > 0 or ip) {
                        const int r1 = s1 / kS, rr = rm[s1], y2 = y[s2], p2 = pf[s2];
                        const int inc = pcost(y2 + (ip ? 0 : 1), p2 + ip, rr) - pcost(y2, p2, rr);
                        const uint32_t key = (static_cast<uint32_t>(T[s2 / kS] + inc) << 15) | static_cast<uint32_t>(p - roff[r1]);
                        #pragma unroll
                        for (uint32_t i = 0; i < kRW; ++ i)
                            if (static_cast<uint32_t>(r1) == rlo + i)
                                mins[i] = min(mins[i], key);
                    }
                }
            }
            uint32_t mine = ~0u;
            #pragma unroll
            for (uint32_t i = 0; i < kRW; ++ i) {
                const uint32_t m = __reduce_min_sync(~0u, mins[i]);
                if (lane == i)
                    mine = m;
            }
            const uint32_t r = rlo + lane;
            if (lane < kRW and r < rhi) {
                dr2[r] = -1;
                if (mine != ~0u) {
                    const uint32_t pr = plist[roff[r] + static_cast<int>(mine & 0x7fffu)];
                    const int s1b = static_cast<int>(pr >> 16), s2b = static_cast<int>(pr & 0xffffu);
                    const int yb = y[s1b], rb = rm[s1b], ib = pf[s1b];
                    const int k2 = static_cast<int>(mine >> 15);
                    dk2[r] = k2; de1[r] = phy[s1b]; dr2[r] = s2b / kS; ds1b[r] = s1b; ds2b[r] = s2b; dip[r] = ib;
                    dd1[r] = ib ? pcost(yb, 0, rb) - pcost(yb, 1, rb) : pcost(yb - 1, 0, rb) - pcost(yb, 0, rb);
                    din2[r] = k2 - T[s2b / kS];
                }
            }
        }
        plan_sync();
        int tot = (lane < kR ? T[lane] : 0) + (lane + 32 < kR ? T[lane + 32] : 0);
        tot = __reduce_add_sync(~0u, tot);
        // (C1) a thread per pair (s0 -> s1) out of a source slot; per slot s0 the min key (== min (value, hops, s1))
        {
            const uint32_t npairs = static_cast<uint32_t>(roff[kR]);
            for (uint32_t p = tid; p < npairs; p += kPlanThreads) {
                const uint32_t pr = plist[p];
                const int s0 = static_cast<int>(pr >> 16), s1 = static_cast<int>(pr & 0xffffu), r0 = s0 / kS;
                const int tmax = T[r0], ys = y[s0], ip = pf[s0];
                if (tmax * static_cast<int>(kR) <= tot or not (ys > 0 or ip))
                    continue;
                const int rr = rm[s0], r1 = s1 / kS, y1 = y[s1], p1 = pf[s1];
                const int t1v = T[r1] + pcost(y1 + (ip ? 0 : 1), p1 + ip, rr) - pcost(y1, p1, rr);
                int v = 0, hops = 0;
                if (t1v < tmax) {
                    v = t1v; hops = 1;
                }
                const int r2 = dr2[r1];
                if (r2 >= 0 and r2 != r0 and de1[r1] != phy[s0] and t1v + dd1[r1] < tmax and dk2[r1] < tmax) {
                    const int v2 = max(dk2[r1], t1v + dd1[r1]);
                    if (hops == 0 or v2 < v) {
                        v = v2; hops = 2;
                    }
                }
                if (hops == 0)
                    continue;
                const int d0 = ip ? pcost(ys, 0, rr) - pcost(ys, 1, rr) : pcost(ys - 1, 0, rr) - pcost(ys, 0, rr);
                const uint64_t key = (static_cast<uint64_t>(max(tmax + d0, v)) << 41) | (static_cast<uint64_t>(v) << 23) |
                                     (static_cast<uint64_t>(hops - 1) << 22) | (static_cast<uint64_t>(s1) << 11) | static_cast<uint64_t>(s0);
                atomicMin(reinterpret_cast<unsigned long long*>(&ckey[s0]), static_cast<unsigned long long>(key));
                csh[s0] = -d0;
            }
        }
        plan_sync();
        // (C2)
        for (uint32_t sb0 = warp * 32; sb0 < kP; sb0 += kPlanThreads) {   // warp-uniform: one offer index atomic per warp
            const uint32_t s0 = sb0 + lane;
            const uint64_t key = s0 < kP ? ckey[s0] : ~0ull;
            bool emit = false;
            uint64_t ok = 0;
            int add = 0, r1 = 0, hops = 0;
            if (key != ~0ull) {
                const int r0 = static_cast<int>(s0 / kS), tmax = T[r0];
                int excl = 0;
                for (int j = rfirst[r0]; j < static_cast<int>((r0 + 1) * kS); ++ j)
                    excl += ckey[j] < key ? csh[j] : 0;
                if (excl * static_cast<int>(kR) < tmax * static_cast<int>(kR) - tot) {
                    const int v = static_cast<int>((key >> 23) & kMaxV);
                    hops = static_cast<int>((key >> 22) & 1u) + 1;
                    const int s1 = static_cast<int>((key >> 11) & 2047u);
                    r1 = s1 / kS;
                    int rcv;
                    if (hops == 1) {
                        const int ip = pf[s0], rr = rm[s0], y1 = y[s1], p1 = pf[s1];
                        rcv = r1; add = pcost(y1 + (ip ? 0 : 1), p1 + ip, rr) - pcost(y1, p1, rr);
                    } else {
                        rcv = dr2[r1]; add = din2[r1];
                    }
                    ok = (static_cast<uint64_t>(rcv) << 58) | (static_cast<uint64_t>(kMaxV - static_cast<uint32_t>(tmax)) << 40) |
                         (static_cast<uint64_t>(v) << 22) | (static_cast<uint64_t>(s0) << 11) | static_cast<uint64_t>(s1);
                    emit = true;
                }
            }
            const uint32_t bal = __ballot_sync(~0u, emit);
            if (bal == 0)
                continue;
            int base = 0;
            if (lane == static_cast<uint32_t>(__ffs(bal) - 1))
                base = atomicAdd(&s_noff, __popc(bal));
            base = __shfl_sync(~0u, base, __ffs(bal) - 1);
            if (emit) {
                const int idx = base + __popc(bal & ((1u << lane) - 1));
                okey[idx] = ok; oadd[idx] = add | (pf[s0] << 30);   // + the source unit's type
                sclaim[s0] = 1u;
                atomicAdd(&rcnt[ok >> 58], 1);
                if (hops == 2)
                    atomicMin(&imin[r1], (static_cast<uint32_t>((ok >> 40) & kMaxV) << 11) | static_cast<uint32_t>((ok >> 11) & 2047u));
            }
        }
        plan_sync();
        // (F1) offers grouped by receiver (segments in key-free order; ckey / csh space: keys / adds)
        const int noff = s_noff;
        uint64_t* const gkey = ckey;
        int* const gadd = csh;
        int rb_lo = 0, rb_hi = 0;   // exclusive prefix of the receiver counts, ranks lane and lane + 32
        {
            const int c0 = lane < kR ? rcnt[lane] : 0, c1 = lane + 32 < kR ? rcnt[lane + 32] : 0;
            int x0 = c0, x1 = c1;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int u0 = __shfl_up_sync(~0u, x0, o), u1 = __shfl_up_sync(~0u, x1, o);
                if (static_cast<int>(lane) >= o) {
                    x0 += u0; x1 += u1;
                }
            }
            const int tot0 = __shfl_sync(~0u, x0, 31);
            rb_lo = x0 - c0; rb_hi = tot0 + x1 - c1;
        }
        const auto rbase = [&](const int& r) {   // uniform shuffles (r differs per lane)
            const int lo = __shfl_sync(~0u, rb_lo, r & 31), hi = __shfl_sync(~0u, rb_hi, r & 31);
            return r < 32 ? lo : hi;
        };
        if (warp == 0) {
            if (lane < kR)
                rbs[lane] = rb_lo;
            if (lane + 32 < kR)
                rbs[lane + 32] = rb_hi;
        }
        for (int i0 = static_cast<int>(warp * 32); i0 < noff; i0 += static_cast<int>(kPlanThreads)) {   // warp-uniform
            const int i = i0 + static_cast<int>(lane);
            const uint64_t ki = i < noff ? okey[i] : 0;
            const int rv = static_cast<int>(ki >> 58);
            const int b = rbase(rv);
            if (i < noff) {
                const int s0 = static_cast<int>((ki >> 11) & 2047u), r0 = s0 / kS, s1 = static_cast<int>(ki & 2047u), r1 = s1 / kS;
                const uint32_t ck = (static_cast<uint32_t>((ki >> 40) & kMaxV) << 11) | static_cast<uint32_t>(s0);
                bool drop = false;
                if (r1 != rv) {   // two-hop: not its intermediate's min, or its transit slot's own offer outranks it
                    const int x = ds1b[r1];
                    drop = imin[r1] != ck or (sclaim[x] and ((kMaxV - static_cast<uint32_t>(T[r1])) << 11 | static_cast<uint32_t>(x)) < ck);
                }
                if (imin[r0] != ~0u and ds1b[r0] == s0 and imin[r0] < ck)   // its slot is a transit of a higher offer
                    drop = true;
                const int pos = b + atomicAdd(&rfill[rv], 1);
                gkey[pos] = ki; gadd[pos] = drop ? -1 : oadd[i];
            }
        }
        plan_sync();
        // (F2) per offer: its receiver segment's smaller kept adds -> prefix acceptance; accepted offers move (typed;
        // the kept offers' source and transit slots are distinct)
        bool won = false;
        for (int i = static_cast<int>(tid); i < noff; i += static_cast<int>(kPlanThreads)) {
            const uint64_t ki = gkey[i];
            const int ai = gadd[i];
            if (ai < 0)
                continue;
            const int rv = static_cast<int>(ki >> 58), s0 = static_cast<int>((ki >> 11) & 2047u), s1 = static_cast<int>(ki & 2047u);
            const int r1 = s1 / kS;
            const int sb = rbs[rv], se = sb + rcnt[rv];
            int S = 0;
            #pragma unroll 4
            for (int j = sb; j < se; ++ j) {
                const int aj = gadd[j];
                S += (gkey[j] <= ki and aj > 0) ? (aj & 0x3fffffff) : 0;
            }
            const int lim = static_cast<int>(kMaxV - static_cast<uint32_t>((ki >> 40) & kMaxV)), cur = Ts[rv] + S;
            if (cur >= lim or (cur - kSlack) * static_cast<int>(kR) > tot)
                continue;
            won = true;
            mv(s0, s1, (ai >> 30) & 1);
            if (r1 != rv)
                mv(ds1b[r1], ds2b[r1], dip[r1]);
        }
        if (plan_sync_count(won) == 0) {
            stop = true;
            break;
        }
    }
    // One-hop rounds: no outflows, no duplicates, no claims (the source slots of the accepted offers are distinct,
    // there are no transit slots) -> the acceptance pass applies its moves
    if (not stop and k < nr_cap) {
    uint64_t* kc = ckey;    // this round's candidate keys (clean: all ~0), then its grouped offers
#if ECO_R <= 32
    uint64_t* kn = ckey2;   // the other buffer: reset during the round for the next one
#endif
    for (uint32_t i = tid; i < kP; i += kPlanThreads) {
        ckey[i] = ~0ull;
#if ECO_R <= 32
        ckey2[i] = ~0ull;
#endif
    }
    plan_sync();
    for (; k < nr_cap; ++ k) {
        // (C1) a thread per pair (s0 -> s1) out of a source slot; per slot the min key (== min (value, s1))
        if (tid < kR) {
            rcnt[tid] = 0; rfill[tid] = 0; Ts[tid] = T[tid];
        }
        if (tid == 0)
            s_noff = 0;
        int tot = (lane < kR ? T[lane] : 0) + (lane + 32 < kR ? T[lane + 32] : 0);
        tot = __reduce_add_sync(~0u, tot);
        {
            const uint32_t npairs = static_cast<uint32_t>(roff[kR]);
            for (uint32_t p = tid; p < npairs; p += kPlanThreads) {
                const uint32_t pr = plist[p];
                const int s0 = static_cast<int>(pr >> 16), s1 = static_cast<int>(pr & 0xffffu), r0 = s0 / kS;
                const int tmax = T[r0], ys = y[s0], ip = pf[s0];
                if (tmax * static_cast<int>(kR) <= tot or not (ys > 0 or ip))
                    continue;
                const int rr = rm[s0], r1 = s1 / kS, y1 = y[s1], p1 = pf[s1];
                const int v = T[r1] + pcost(y1 + (ip ? 0 : 1), p1 + ip, rr) - pcost(y1, p1, rr);
                if (v >= tmax)
                    continue;
                const int d0 = ip ? pcost(ys, 0, rr) - pcost(ys, 1, rr) : pcost(ys - 1, 0, rr) - pcost(ys, 0, rr);
                const uint64_t key = (static_cast<uint64_t>(max(tmax + d0, v)) << 41) | (static_cast<uint64_t>(v) << 23) |
                                     (static_cast<uint64_t>(s1) << 11) | static_cast<uint64_t>(s0);
                atomicMin(reinterpret_cast<unsigned long long*>(&kc[s0]), static_cast<unsigned long long>(key));
                csh[s0] = -d0;
            }
        }
        plan_sync();
        // (C2) offered while the shed of the source's smaller keys is below its excess
        for (uint32_t sb0 = warp * 32; sb0 < kP; sb0 += kPlanThreads) {   // warp-uniform: one offer index atomic per warp
            const uint32_t s0 = sb0 + lane;
            const uint64_t key = s0 < kP ? kc[s0] : ~0ull;
            bool emit = false;
            uint64_t ok = 0;
            int add = 0;
            if (key != ~0ull) {
                const int r0 = static_cast<int>(s0 / kS), tmax = T[r0];
                int excl = 0;
                for (int j = rfirst[r0]; j < static_cast<int>((r0 + 1) * kS); ++ j)
                    excl += kc[j] < key ? csh[j] : 0;
                if (excl * static_cast<int>(kR) < tmax * static_cast<int>(kR) - tot) {
                    const int v = static_cast<int>((key >> 23) & kMaxV), s1 = static_cast<int>((key >> 11) & 2047u);
                    const int ip = pf[s0], rr = rm[s0], y1 = y[s1], p1 = pf[s1];
                    add = (pcost(y1 + (ip ? 0 : 1), p1 + ip, rr) - pcost(y1, p1, rr)) | (ip << 30);   // + the unit type
                    ok = (static_cast<uint64_t>(s1 / kS) << 58) | (static_cast<uint64_t>(kMaxV - static_cast<uint32_t>(tmax)) << 40) |
                         (static_cast<uint64_t>(v) << 22) | (static_cast<uint64_t>(s0) << 11) | static_cast<uint64_t>(s1);
                    emit = true;
                }
            }
            const uint32_t bal = __ballot_sync(~0u, emit);
            if (bal == 0)
                continue;
            int base = 0;
            if (lane == static_cast<uint32_t>(__ffs(bal) - 1))
                base = atomicAdd(&s_noff, __popc(bal));
            base = __shfl_sync(~0u, base, __ffs(bal) - 1);
            if (emit) {
                const int idx = base + __popc(bal & ((1u << lane) - 1));
                okey[idx] = ok; oadd[idx] = add;
                atomicAdd(&rcnt[ok >> 58], 1);
            }
        }
        plan_sync();
        // (F1) offers grouped by receiver (ckey / csh space: keys / adds); the next round's candidates reset after
        const int noff = s_noff;
        uint64_t* const gkey = kc;
        int* const gadd = csh;
        int rb_lo = 0, rb_hi = 0;
        {
            const int c0 = lane < kR ? rcnt[lane] : 0, c1 = lane + 32 < kR ? rcnt[lane + 32] : 0;
            int x0 = c0, x1 = c1;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int u0 = __shfl_up_sync(~0u, x0, o), u1 = __shfl_up_sync(~0u, x1, o);
                if (static_cast<int>(lane) >= o) {
                    x0 += u0; x1 += u1;
                }
            }
            rb_lo = x0 - c0; rb_hi = __shfl_sync(~0u, x0, 31) + x1 - c1;
        }
        if (warp == 0) {
            if (lane < kR)
                rbs[lane] = rb_lo;
            if (lane + 32 < kR)
                rbs[lane + 32] = rb_hi;
        }
#if ECO_R <= 32
        for (uint32_t i = tid; i < kP; i += kPlanThreads)   // the next round's candidate buffer
            kn[i] = ~0ull;
#endif
        for (int i0 = static_cast<int>(warp * 32); i0 < noff; i0 += static_cast<int>(kPlanThreads)) {   // warp-uniform
            const int i = i0 + static_cast<int>(lane);
            const uint64_t ki = i < noff ? okey[i] : 0;
            const int rv = static_cast<int>(ki >> 58);
            const int lo = __shfl_sync(~0u, rb_lo, rv & 31), hi = __shfl_sync(~0u, rb_hi, rv & 31);
            if (i < noff) {
                const int pos = (rv < 32 ? lo : hi) + atomicAdd(&rfill[rv], 1);
                gkey[pos] = ki; gadd[pos] = oadd[i];
            }
        }
        plan_sync();
        // (F2) per offer: its receiver segment's smaller adds -> prefix acceptance; accepted offers move (typed)
        bool won = false;
        for (int i = static_cast<int>(tid); i < noff; i += static_cast<int>(kPlanThreads)) {
            const uint64_t ki = gkey[i];
            const int rv = static_cast<int>(ki >> 58), s0 = static_cast<int>((ki >> 11) & 2047u), s1 = static_cast<int>(ki & 2047u);
            const int sb = rbs[rv], se = sb + rcnt[rv];
            int S = 0;
            #pragma unroll 4
            for (int j = sb; j < se; ++ j)
                S += gkey[j] <= ki ? (gadd[j] & 0x3fffffff) : 0;
            const int lim = static_cast<int>(kMaxV - static_cast<uint32_t>((ki >> 40) & kMaxV)), cur = Ts[rv] + S;
            if (cur >= lim or (cur - kSlack) * static_cast<int>(kR) > tot)
                continue;
            won = true;
            mv(s0, s1, (gadd[i] >> 30) & 1);
        }
        if (plan_sync_count(won) == 0)
            break;
#if ECO_R <= 32
        uint64_t* const tb = kc;
        kc = kn; kn = tb;
#else
        if (k + 1 < nr_cap) {   // one buffer (shared memory): clean it for the next round
            for (uint32_t i = tid; i < kP; i += kPlanThreads)
                ckey[i] = ~0ull;
            plan_sync();
        }
#endif
    }
    }
}

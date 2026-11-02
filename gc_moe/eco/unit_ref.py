"""Bit-exact numpy reference of the ECO plan role (`eco_unit_kernel.cu`).

Units: an expert's n tokens are k = ceil(n / bm) units, k - 1 full ones of bm tokens and a last partial one.
Rank time t_r = wb * blocks + wa * active slots + tokens over the rank's slots (wb = wa = 0: tokens alone); the plan
minimizes the slowest rank's time. One plan:
  1. start: warm, the previous plan's units per copy (partial owner: the first copy whose quota is congruent to the
     previous partial size); only experts whose unit count changed add units (copy with the lowest estimated rank time
     + this expert's additions) or remove units (highest first among copies holding one); the partial stays with its
     owner while it holds a unit, else goes to the holding copy on the highest-time rank. All experts decide against
     one rank-time estimate (kept units, new partial sizes). Cold: an even split, load-aware rounding to units.
  2. auction rounds: every rank above the mean offers, per replicated slot, its best one-hop unit move (or two-hop
     path through another rank's best outgoing unit, first `hop2_rounds` rounds) below its own time, up to its excess
     over the mean; receivers accept a prefix (higher source time first) while staying below the source's time and
     the mean + slack. Stops early when no move is accepted.
A copy is passive (never planned) when a lower copy of the same expert sits on the same rank."""

from __future__ import annotations

import numpy as np


class UnitPlanner:
    def __init__(self, phy, num_ranks: int, num_logical: int, block_m: int = 64, wb: int = 320, wa: int = 54,
                 rounds: int = 1, cold_rounds: int = 6, hop2_rounds: int = 0):
        phy = np.asarray(phy, np.int64)
        self.phy, self.P, self.R, self.E = phy, len(phy), num_ranks, num_logical
        self.S = self.P // num_ranks
        self.rank = np.arange(self.P) // self.S
        self.bm, self.wb, self.wa = block_m, wb, wa
        self.nr, self.nr_cold, self.hop2_rounds = rounds, cold_rounds, hop2_rounds
        self.slack = wb + block_m + wa
        self.active = np.ones(self.P, bool)
        self.slot_of = np.zeros(self.E, np.int64)
        copies = {}
        for e in range(self.E):
            seen, act = set(), []
            for s in np.nonzero(phy == e)[0]:
                r = int(self.rank[s])
                if r in seen:
                    self.active[s] = False
                    continue
                seen.add(r)
                act.append(int(s))
            copies[e] = act
            self.slot_of[e] = act[0] if act else -1
        self.copies = {e: cs for e, cs in copies.items() if len(cs) > 1}

    # ---- costs -------------------------------------------------------------------------------------------------
    def rank_time(self, q):
        q = np.asarray(q, np.int64)
        return np.bincount(self.rank, self.wb * (-(-q // self.bm)) + self.wa * (q > 0) + q, self.R).astype(np.int64)

    def _cost(self, s, y, hp):
        blk = y + hp
        return self.wb * blk + (self.wa if blk > 0 else 0) + y * self.bm + hp * self.rem[self.phy[s]]

    def _has_unit(self, s):
        return self.y[s] > 0 or self.part[self.phy[s]] == s

    def _out(self, s):  # time change of rank(s) when one unit (the partial if s owns it) leaves s
        y, hp = int(self.y[s]), int(self.part[self.phy[s]] == s)
        if hp:
            return self._cost(s, y, 0) - self._cost(s, y, 1), 1
        return self._cost(s, y - 1, 0) - self._cost(s, y, 0), 0

    def _in(self, s, ispart):  # time change of rank(s) when one unit of type `ispart` enters s
        y, hp = int(self.y[s]), int(self.part[self.phy[s]] == s)
        return self._cost(s, y + (0 if ispart else 1), hp + ispart) - self._cost(s, y, hp)

    def _move(self, a, b, ispart):  # the partial unit (a must own it) or one full unit from slot a to slot b
        e = self.phy[a]
        if ispart:
            assert self.part[e] == a
            self.part[e] = b
        else:
            assert self.y[a] > 0
            self.y[a] -= 1
            self.y[b] += 1

    def quota(self):
        q = self.y * self.bm
        m = self.part >= 0
        q[self.part[m]] += self.rem[m]
        return q

    # ---- plan --------------------------------------------------------------------------------------------------
    def plan(self, n, prev=None):
        """Quota per physical slot for logical counts `n`; `prev` = (previous quotas, previous counts) or None."""
        n = np.asarray(n, np.int64)
        bm = self.bm
        self.n = n
        self.k = -(-n // bm)
        self.rem = n - (self.k - 1) * bm
        self.y = np.zeros(self.P, np.int64)
        self.part = np.full(self.E, -1, np.int64)
        for e in np.nonzero(n > 0)[0]:
            if int(e) not in self.copies:
                s0 = int(self.slot_of[e])
                self.y[s0] = self.k[e] - 1
                self.part[e] = s0
        if prev is None or int(np.asarray(prev[1]).sum()) == 0:
            self._cold(n)
            return self._rounds(self.nr_cold)
        self._warm(n, np.asarray(prev[0], np.int64), np.asarray(prev[1], np.int64))
        return self._rounds(self.nr)

    def _cold(self, n):   # even split, then round up the copies with the smallest (rank time * bm - leftover * (wb + bm))
        bm = self.bm
        x = np.zeros(self.P, np.int64)
        for e in range(self.E):
            if e not in self.copies and self.slot_of[e] >= 0:
                x[self.slot_of[e]] = n[e]
        for e, cs in self.copies.items():
            tot, k = int(n[e]), len(cs)
            x[cs] = tot // k
            x[cs[0]] += tot - (tot // k) * k
        t0 = self.rank_time(x)
        for e, cs in self.copies.items():
            if n[e] == 0:
                continue
            g = [int(x[s]) // bm for s in cs]
            left = int(self.k[e]) - sum(g)
            order = sorted(range(len(cs)), key=lambda j: (int(t0[self.rank[cs[j]]]) * bm - (int(x[cs[j]]) % bm) * (self.wb + bm), j))
            for i in range(left):
                g[order[i]] += 1
            jp = max((j for j in range(len(cs)) if g[j] > 0), key=lambda j: (g[j] * bm - int(x[cs[j]]), -j))
            for j, s in enumerate(cs):
                self.y[s] = g[j] - (j == jp)
            self.part[e] = cs[jp]

    def _warm(self, n, qp, npv):
        bm, unit = self.bm, self.wb + self.bm
        keep = {}
        for e, cs in self.copies.items():
            if n[e] == 0:
                continue
            kp = -(-int(npv[e]) // bm)
            u, own = [0] * len(cs), -1
            if kp > 0:
                remp = int(npv[e]) - (kp - 1) * bm
                qs = [int(qp[s]) for s in cs]
                own = next((j for j in range(len(cs)) if qs[j] >= remp and (qs[j] - remp) % bm == 0), -1)
                if own >= 0:   # else an inconsistent previous plan: dropped for this expert
                    u = [(qs[j] - (remp if j == own else 0)) // bm + (j == own) for j in range(len(cs))]
            keep[e] = (u, own)
        # estimate: kept units, new partial sizes (unit count changes not yet applied)
        y0, part0 = self.y.copy(), self.part.copy()
        for e, (u, own) in keep.items():
            cs = self.copies[e]
            if own >= 0:
                for j, s in enumerate(cs):
                    y0[s] = u[j] - (j == own)
                part0[e] = cs[own]
        q0 = y0 * bm
        m = part0 >= 0
        q0[part0[m]] += self.rem[m]
        t0 = self.rank_time(q0)
        for e, (u, own) in keep.items():
            cs = self.copies[e]
            u = list(u)
            delta = int(self.k[e]) - sum(u)
            add = [0] * len(cs)
            while delta > 0:
                j = min(range(len(cs)), key=lambda j: (int(t0[self.rank[cs[j]]]) + add[j] * unit, j))
                u[j] += 1; add[j] += 1; delta -= 1
            while delta < 0:
                j = max((j for j in range(len(cs)) if u[j] > 0), key=lambda j: (int(t0[self.rank[cs[j]]]) + add[j] * unit, -j))
                u[j] -= 1; add[j] -= 1; delta += 1
            if own < 0 or u[own] == 0:
                own = max((j for j in range(len(cs)) if u[j] > 0), key=lambda j: (int(t0[self.rank[cs[j]]]), -j))
            for j, s in enumerate(cs):
                self.y[s] = u[j] - (j == own)
            self.part[e] = cs[own]

    # ---- auction rounds ----------------------------------------------------------------------------------------
    def _rounds(self, nr):
        for k in range(nr):
            if self._auction_round(k < self.hop2_rounds) == 0:
                break
        return self.quota()

    def _auction_round(self, hop2):
        t = self.rank_time(self.quota())
        R, rank, phy = self.R, self.rank, self.phy
        tot = int(t.sum())
        best_out = {}   # per rank: its best outgoing unit (key (receiver time after, s1, s2), d1, in2, ispart)
        if hop2:
            for e, cs in self.copies.items():
                for s1 in cs:
                    if not self._has_unit(s1):
                        continue
                    r1 = int(rank[s1])
                    d1, ip = self._out(s1)
                    for s2 in cs:
                        if s2 != s1:
                            key = (int(t[rank[s2]]) + self._in(s2, ip), s1, s2)
                            if r1 not in best_out or key < best_out[r1][0]:
                                best_out[r1] = (key, d1, self._in(s2, ip), ip)
        # offers: (receiver, (-source time, path value, s0), s0, source time, path, s0 owns the partial)
        # path: (value, hops, s1, receiver, receiver's added time, (s1b, s2) or None, ispart of the second hop)
        offers = []
        for r0 in range(R):
            if int(t[r0]) * R <= tot:
                continue
            excess, lim = int(t[r0]) * R - tot, int(t[r0])
            best = {}
            for s0 in range(r0 * self.S, (r0 + 1) * self.S):
                e = int(phy[s0])
                if e not in self.copies or not self.active[s0] or not self._has_unit(s0):
                    continue
                d0, ip = self._out(s0)
                for s1 in self.copies[e]:
                    if s1 == s0:
                        continue
                    r1 = int(rank[s1]); in1 = self._in(s1, ip)
                    c = (int(t[r1]) + in1, 1, s1, r1, in1, None, 0) if int(t[r1]) + in1 < lim else None
                    if r1 in best_out:
                        (k2, s1b, s2), d1, in2, ip2 = best_out[r1]
                        if rank[s2] != r0 and phy[s1b] != e and int(t[r1]) + in1 + d1 < lim and k2 < lim:
                            c2 = (max(k2, int(t[r1]) + in1 + d1), 2, s1, int(rank[s2]), in2, (s1b, s2), ip2)
                            if c is None or c2[0] < c[0]:
                                c = c2
                    if c is not None and (s0 not in best or c[:3] < best[s0][2][:3]):
                        best[s0] = (int(t[r0]) + d0, d0, c)
            shed = 0
            for s0, (src_new, d0, c) in sorted(best.items(), key=lambda kv: (max(kv[1][0], kv[1][2][0]), kv[1][2][:3], kv[0])):
                if shed * R >= excess:
                    break
                offers.append((c[3], (-int(t[r0]), c[0], s0), s0, lim, c, int(self.part[phy[s0]] == s0)))
                shed -= d0
        return self._accept(offers, t, tot)

    def _accept(self, offers, t, tot):
        R = self.R
        ck = lambda o: (o[1][0], o[2])   # (-source time, s0)
        win2 = {}   # per transit slot: the two-hop offer with the min (-source time, s0)
        for o in offers:
            if o[4][5] is not None and (o[4][5][0] not in win2 or ck(o) < win2[o[4][5][0]]):
                win2[o[4][5][0]] = ck(o)
        drop = set()   # a transit slot that also makes its own offer: the lower (-source time, s0) one is dropped
        own = {o[2]: o for o in offers}
        for x, kw in win2.items():
            if x in own:
                drop.add(('own', x) if kw < ck(own[x]) else ('transit', x))
        groups = {}
        for o in sorted(offers, key=lambda o: (o[0], o[1])):
            groups.setdefault(o[0], []).append(o)
        acc = []
        for rcv, grp in groups.items():
            cur = int(t[rcv])
            for o in grp:
                c = o[4]
                if ('own', o[2]) in drop:
                    continue
                if c[5] is not None and (('transit', c[5][0]) in drop or win2[c[5][0]] != ck(o)):
                    continue
                if cur + c[4] >= o[3] or (cur + c[4] - self.slack) * R > tot:
                    break
                cur += c[4]
                acc.append(o)
        for o in acc:
            self._move(o[2], o[4][2], o[5])
            if o[4][5] is not None:
                self._move(o[4][5][0], o[4][5][1], o[4][6])
        return len(acc)


def unit_section(phy, num_ranks: int, num_logical: int):
    """Static tables of the unit plan role, appended to the layer's EcoMeta (offset in header[15]), int32 words:
    [ns, nb, npairs, 0] [small experts (2..4 active copies), kP - kE] [big experts (>= 5), kP - kE]
    [per slot: list start << 8 | active copies of its expert (0: passive), kP] [per expert: list start << 8 | active
    copies, kE] [active copies expert-major ascending, kP] [per-rank first pair, kR + 1, padded to 4]
    [pairs (slot << 16 | copy) ascending slots, then copies]."""
    phy = np.asarray(phy, np.int64)
    P, R, E = len(phy), num_ranks, num_logical
    S = P // R
    act, ecnt = {}, np.zeros(E, np.int64)
    for e in range(E):
        seen, a = set(), []
        for s in np.nonzero(phy == e)[0]:
            if s // S not in seen:
                seen.add(s // S); a.append(int(s))
        act[e] = a; ecnt[e] = len(a)
    eoff = np.concatenate([[0], np.cumsum(ecnt)])[:-1]
    cslot = np.zeros(P, np.int64)
    clen = np.zeros(P, np.int64); cst = np.zeros(P, np.int64)
    for e in range(E):
        for j, s in enumerate(act[e]):
            cslot[eoff[e] + j] = s
        for s in np.nonzero(phy == e)[0]:
            cst[s] = eoff[e]; clen[s] = ecnt[e] if s in act[e] else 0
    small = [e for e in range(E) if 2 <= ecnt[e] <= 4]
    big = [e for e in range(E) if ecnt[e] > 4]
    plist, first = [], np.zeros(P, np.int64)
    for s in range(P):
        first[s] = len(plist)
        if clen[s] >= 2:
            e = phy[s]
            plist += [(s << 16) | s2 for s2 in act[e] if s2 != s]
    roff = np.array([first[r * S] for r in range(R)] + [len(plist)], np.int64)
    assert len(plist) <= min((P - E) * min(R, 32), 2048), "unit planner: too many copy pairs for the static table"
    G = P - E
    pad = lambda a, n: np.concatenate([np.asarray(a, np.int64), np.zeros(n - len(a), np.int64)])
    words = np.concatenate([[len(small), len(big), len(plist), 0], pad(small, G), pad(big, G), (cst << 8) | clen,
                            (eoff << 8) | ecnt, cslot, pad(roff, (R + 1 + 3) // 4 * 4), np.asarray(plist, np.int64)])
    return (words & 0xFFFFFFFF).astype(np.uint32).view(np.int32)

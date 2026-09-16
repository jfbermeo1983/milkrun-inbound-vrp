# -*- coding: utf-8 -*-
"""
Milk-run inbound VRP with two alternative objectives: total distance travelled
and total CO2e emitted.

Modelling conventions
---------------------
Emissions.  The per-kilometre emission factor is AFFINE in the payload carried
    on each arc and is fitted by least squares to Table 1 of the manuscript
    (Ubeda et al., 2011):
        e(load) = E0 + G * load_kg        [kg CO2 / km]
    The intercept E0 is the emission rate of the empty vehicle and therefore
    dominates: a purely load-proportional factor would attribute no emissions
    at all to an empty leg and would misrepresent the trade-off between route
    length and payload sequencing.

Route convention.  By default the approach leg from the plant to the first
    supplier is charged to the route (travelled empty). Setting
    include_outbound=False reproduces the "open route" convention, in which
    that leg is ignored; both conventions are run in the experiments so that
    the bias introduced by the open-route accounting can be quantified.

Search.  A tabu search over shift, swap and Or-opt moves is run from several
    random seeds; the best solution is reported together with dispersion
    statistics across seeds, so that the stability of the metaheuristic can be
    assessed. DELTA_MAX (maximum detour per intermediate stop) and the
    consolidation thresholds are explicit parameters of the model. The stop
    limit S is not fixed a priori but selected with the marginal-improvement
    rule studied in sweep.py.

Usage:  from solver import Instance, Params, solve, exact_solve
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Emission model derived from Table 1 of the manuscript
# --------------------------------------------------------------------------
# Table 1: fuel consumption and emission factor by load percentage (24 t truck)
TABLA1 = [(0, 0.7319), (25, 0.8148), (50, 0.9079), (75, 1.0089), (100, 1.0838)]


def fit_emission_factor(cap_kg: float, tabla=TABLA1) -> Tuple[float, float]:
    """Least-squares fit of e(load) = E0 + G*load to the Table 1 data points.

    The tabulated load percentages are first converted to absolute payloads
    using the vehicle weight capacity, so that the slope G is expressed in
    kg CO2 per kg of payload and per kilometre.
    """
    xs = [cap_kg * p / 100.0 for p, _ in tabla]
    ys = [f for _, f in tabla]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    g = sxy / sxx
    e0 = my - g * mx
    return e0, g


@dataclass
class Params:
    cap_weight: float = 24000.0      # weight capacity per vehicle (kg)
    cap_volume: float = 100.0        # volume capacity per vehicle (m3)
    max_stops: int = 4               # stop limit S
    delta_max: Optional[float] = None    # maximum detour per intermediate stop (km); None = unbounded
    min_weight_visit: float = 0.0    # consolidation threshold on weight (kg)
    min_volume_visit: float = 0.0    # consolidation threshold on volume (m3)
    consol_policy: str = "and"       # a supplier is consolidated if it is below both ("and") or either ("or") threshold
    include_outbound: bool = True    # charge the empty plant -> first supplier leg to the route
    e0: float = 0.7299               # emission rate of the empty vehicle (kg CO2/km)
    gamma: float = 1.497e-5          # marginal emission rate per unit payload (kg CO2/(kg*km))
    # tabu search control parameters
    tenure: int = 15
    iters: int = 1200
    sample_moves: int = 250
    stall_limit: int = 250


@dataclass
class Instance:
    name: str
    D: List[List[float]]
    q: List[float]
    v: List[float]
    depot: int = 0

    @property
    def n(self) -> int:
        return len(self.D)


# --------------------------------------------------------------------------
# Route evaluation
# --------------------------------------------------------------------------
class Model:
    """Bind an instance to a parameter set and evaluate routes.

    A route is represented as the ordered sequence of suppliers visited; the
    depot is implicit and is added at the start (when the outbound leg is
    charged) and always at the end. Suppliers with zero demand, and those
    falling below the consolidation thresholds, are excluded from routing and
    are assumed to be served by a separate consolidation scheme.
    """

    def __init__(self, inst: Instance, p: Params):
        self.inst, self.p = inst, p
        self.D, self.q, self.v = inst.D, inst.q, inst.v
        self.depot = inst.depot
        sup_all = [i for i in range(inst.n) if i != self.depot]
        self.sup_zero = [i for i in sup_all if self.q[i] == 0 and self.v[i] == 0]
        rest = [i for i in sup_all if i not in self.sup_zero]
        if p.min_weight_visit <= 0 and p.min_volume_visit <= 0:
            self.sup_conso = []
        elif p.consol_policy == "and":
            self.sup_conso = [i for i in rest
                              if self.q[i] < p.min_weight_visit and self.v[i] < p.min_volume_visit]
        else:
            self.sup_conso = [i for i in rest
                              if self.q[i] < p.min_weight_visit or self.v[i] < p.min_volume_visit]
        self.sup_visit = [i for i in rest if i not in self.sup_conso]

    # ---- performance measures ----
    def distance(self, seq: Sequence[int]) -> float:
        if not seq:
            return 0.0
        D, d0 = self.D, self.depot
        km = self.D[d0][seq[0]] if self.p.include_outbound else 0.0
        for a, b in zip(seq, seq[1:]):
            km += D[a][b]
        km += D[seq[-1]][d0]
        return km

    def emissions(self, seq: Sequence[int]) -> float:
        """Affine emissions: sum over arcs of d_ij * (E0 + G * load on the arc).

        In an inbound milk run the payload grows monotonically along the route,
        so the order in which suppliers are visited changes the emissions even
        when the total distance is unchanged: collecting heavy suppliers late
        keeps the vehicle lighter over more of the route.
        """
        if not seq:
            return 0.0
        D, q, d0 = self.D, self.q, self.depot
        e0, g = self.p.e0, self.p.gamma
        em = 0.0
        if self.p.include_outbound:
            em += D[d0][seq[0]] * e0          # approach leg, travelled empty
        load = 0.0
        for a, b in zip(seq, seq[1:]):
            load += q[a]
            em += D[a][b] * (e0 + g * load)
        load += q[seq[-1]]
        em += D[seq[-1]][d0] * (e0 + g * load)
        return em

    def weight(self, seq): return sum(self.q[i] for i in seq)
    def volume(self, seq): return sum(self.v[i] for i in seq)

    def cost(self, seq: Sequence[int], objective: str) -> float:
        return self.distance(seq) if objective == "dist" else self.emissions(seq)

    # ---- feasibility ----
    def delta_ok(self, seq: Sequence[int]) -> bool:
        if self.p.delta_max is None or len(seq) <= 1:
            return True
        D, dm = self.D, self.p.delta_max
        ext = ([self.depot] if self.p.include_outbound else []) + list(seq) + [self.depot]
        for t in range(1, len(ext) - 1):
            i, j, k = ext[t - 1], ext[t], ext[t + 1]
            if D[i][j] + D[j][k] - D[i][k] > dm + 1e-9:
                return False
        return True

    def feasible(self, seq: Sequence[int]) -> bool:
        return (0 < len(seq) <= self.p.max_stops
                and self.weight(seq) <= self.p.cap_weight + 1e-9
                and self.volume(seq) <= self.p.cap_volume + 1e-9
                and self.delta_ok(seq))

    def total(self, routes, objective) -> float:
        return sum(self.cost(r, objective) for r in routes)


# --------------------------------------------------------------------------
# Intra-route 2-opt, applied with respect to the active objective
# --------------------------------------------------------------------------
# Under the emission objective the improvement test is not symmetric in the
# direction of travel, so every reversed segment must be re-evaluated in full
# rather than compared on arc lengths alone.
def two_opt(m: Model, seq: List[int], objective: str) -> List[int]:
    if len(seq) < 3:
        return seq
    best, bc = seq, m.cost(seq, objective)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not m.feasible(cand):
                    continue
                c = m.cost(cand, objective)
                if c < bc - 1e-9:
                    best, bc, improved = cand, c, True
    return best


# --------------------------------------------------------------------------
# Initial solution: Clarke and Wright savings adapted to inbound milk runs
# --------------------------------------------------------------------------
# Routes begin at a supplier rather than at the depot, and savings are DIRECTED:
# the pair (a, b) merges the tail of one route with the head of another, so
# s(a, b) and s(b, a) are distinct candidates. This preserves the visiting order
# that the load-dependent emission term depends on. A small multiplicative noise
# term diversifies the starting solution across seeds.
def clarke_wright(m: Model, objective: str, rnd: random.Random, noise: float = 0.0) -> List[List[int]]:
    routes = [[i] for i in m.sup_visit]
    if not routes:
        return []
    D, d0 = m.D, m.depot
    sav = []
    for a in m.sup_visit:
        for b in m.sup_visit:
            if a == b:
                continue
            s = D[a][d0] + D[d0][b] - D[a][b]
            if noise:
                s *= 1.0 + rnd.uniform(-noise, noise)
            sav.append((s, a, b))
    sav.sort(reverse=True)
    tail = {r[-1]: idx for idx, r in enumerate(routes)}
    head = {r[0]: idx for idx, r in enumerate(routes)}
    alive = [True] * len(routes)
    for _, a, b in sav:
        ia, ib = tail.get(a), head.get(b)
        if ia is None or ib is None or ia == ib or not alive[ia] or not alive[ib]:
            continue
        merged = routes[ia] + routes[ib]
        if not m.feasible(merged):
            continue
        head.pop(routes[ia][0], None); tail.pop(routes[ia][-1], None)
        head.pop(routes[ib][0], None); tail.pop(routes[ib][-1], None)
        alive[ib] = False
        routes[ia] = merged
        head[merged[0]] = ia
        tail[merged[-1]] = ia
    out = [routes[i] for i in range(len(routes)) if alive[i]]
    return [two_opt(m, r, objective) for r in out]


# --------------------------------------------------------------------------
# Tabu search with incremental evaluation: only the routes touched by a move
# are re-costed, which keeps the neighbourhood sampling affordable
# --------------------------------------------------------------------------
def tabu_search(m: Model, objective: str, seed: int) -> Tuple[List[List[int]], dict]:
    rnd = random.Random(seed)
    p = m.p
    cur = clarke_wright(m, objective, rnd, noise=0.15 if seed else 0.0)
    if not cur:
        return [], {"iters": 0}
    cc = [m.cost(r, objective) for r in cur]
    cur_cost = sum(cc)
    best, best_cost = [r[:] for r in cur], cur_cost
    tabu = {}
    stall = 0
    it = 0
    for it in range(1, p.iters + 1):
        for k in list(tabu):
            if tabu[k] <= it:
                del tabu[k]
        nR = len(cur)
        best_mv = None
        best_delta = math.inf
        best_key = None
        for _ in range(p.sample_moves):
            kind = rnd.random()
            ri = rnd.randrange(nR)
            rj = rnd.randrange(nR)
            if not cur[ri]:
                continue
            if kind < 0.45:                       # SHIFT (1,0): relocate one supplier
                if ri == rj:
                    continue
                pi = rnd.randrange(len(cur[ri]))
                u = cur[ri][pi]
                pj = rnd.randint(0, len(cur[rj]))
                ni = cur[ri][:pi] + cur[ri][pi + 1:]
                nj = cur[rj][:pj] + [u] + cur[rj][pj:]
                if not m.feasible(nj) or (ni and not m.feasible(ni)):
                    continue
                nj = two_opt(m, nj, objective)
                delta = (m.cost(ni, objective) if ni else 0.0) + m.cost(nj, objective) - cc[ri] - cc[rj]
                key = ("shift", u, rj)
                mv = (ri, ni, rj, nj)
            elif kind < 0.80:                     # SWAP (1,1): exchange two suppliers
                if ri == rj or not cur[rj]:
                    continue
                pi = rnd.randrange(len(cur[ri])); pj = rnd.randrange(len(cur[rj]))
                ni = cur[ri][:]; nj = cur[rj][:]
                ni[pi], nj[pj] = nj[pj], ni[pi]
                if not m.feasible(ni) or not m.feasible(nj):
                    continue
                ni = two_opt(m, ni, objective); nj = two_opt(m, nj, objective)
                delta = m.cost(ni, objective) + m.cost(nj, objective) - cc[ri] - cc[rj]
                a, b = cur[ri][pi], cur[rj][pj]
                key = ("swap", min(a, b), max(a, b))
                mv = (ri, ni, rj, nj)
            else:                                  # OR-OPT: relocate a block of two consecutive suppliers
                if ri == rj or len(cur[ri]) < 2:
                    continue
                pi = rnd.randrange(len(cur[ri]) - 1)
                blk = cur[ri][pi:pi + 2]
                ni = cur[ri][:pi] + cur[ri][pi + 2:]
                pj = rnd.randint(0, len(cur[rj]))
                nj = cur[rj][:pj] + blk + cur[rj][pj:]
                if not m.feasible(nj) or (ni and not m.feasible(ni)):
                    continue
                nj = two_opt(m, nj, objective)
                delta = (m.cost(ni, objective) if ni else 0.0) + m.cost(nj, objective) - cc[ri] - cc[rj]
                key = ("or2", blk[0], rj)
                mv = (ri, ni, rj, nj)

            tabu_hit = key in tabu
            aspires = cur_cost + delta < best_cost - 1e-9
            if tabu_hit and not aspires:
                continue
            if delta < best_delta - 1e-9:
                best_delta, best_mv, best_key = delta, mv, key

        if best_mv is None:
            break
        ri, ni, rj, nj = best_mv
        cur[ri], cur[rj] = ni, nj
        cc[ri] = m.cost(ni, objective) if ni else 0.0
        cc[rj] = m.cost(nj, objective)
        if not cur[ri]:
            cur.pop(ri); cc.pop(ri)
        cur_cost = sum(cc)
        tabu[best_key] = it + p.tenure
        if cur_cost < best_cost - 1e-9:
            best, best_cost, stall = [r[:] for r in cur], cur_cost, 0
        else:
            stall += 1
            if stall >= p.stall_limit:
                break
    return best, {"iters": it}


def solve(inst: Instance, p: Params, objective: str, seeds=range(10)):
    """Run the tabu search from several seeds.

    Returns the best solution found together with summary statistics (best,
    mean, standard deviation and worst objective value, and running times),
    which are used to report the dispersion of the metaheuristic.
    """
    import time
    res = []
    t0 = time.perf_counter()
    for s in seeds:
        sol, _ = tabu_search(Model(inst, p), objective, s)
        m = Model(inst, p)
        res.append((m.total(sol, objective), sol))
    t = time.perf_counter() - t0
    res.sort(key=lambda x: x[0])
    vals = [r[0] for r in res]
    mean = sum(vals) / len(vals)
    sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
    return res[0][1], {
        "best": vals[0], "mean": mean, "sd": sd, "worst": vals[-1],
        "time_s": t, "time_per_seed": t / len(vals), "n_seeds": len(vals),
    }


# --------------------------------------------------------------------------
# Exact optimum by set partitioning, for small instances used in validation
# --------------------------------------------------------------------------
def exact_solve(inst: Instance, p: Params, objective: str, max_nodes: int = 14):
    """Exact set partitioning.

    All feasible routes of at most S stops are enumerated, each is costed at
    its best permutation under the active objective, and the optimal partition
    of the supplier set is obtained by dynamic programming over bitmasks. The
    result provides the reference optimum against which the tabu search gap is
    measured.
    """
    from itertools import combinations, permutations
    m = Model(inst, p)
    nodes = m.sup_visit
    if len(nodes) > max_nodes:
        raise ValueError("instancia demasiado grande para enumeracion exacta")
    idx = {c: k for k, c in enumerate(nodes)}
    N = len(nodes)
    best_route = {}
    for size in range(1, p.max_stops + 1):
        for comb in combinations(nodes, size):
            if sum(inst.q[i] for i in comb) > p.cap_weight + 1e-9:
                continue
            if sum(inst.v[i] for i in comb) > p.cap_volume + 1e-9:
                continue
            mask = 0
            for c in comb:
                mask |= 1 << idx[c]
            bc, bs = math.inf, None
            for perm in permutations(comb):
                if not m.feasible(list(perm)):
                    continue
                c = m.cost(list(perm), objective)
                if c < bc:
                    bc, bs = c, list(perm)
            if bs is not None:
                best_route[mask] = (bc, bs)
    FULL = (1 << N) - 1
    dp = [math.inf] * (1 << N)
    par = [None] * (1 << N)
    dp[0] = 0.0
    for state in range(1 << N):
        if dp[state] == math.inf:
            continue
        low = (~state) & FULL
        if not low:
            continue
        first = (low & -low)
        sub = low
        while sub:
            if sub & first:
                r = best_route.get(sub)
                if r and dp[state] + r[0] < dp[state | sub] - 1e-12:
                    dp[state | sub] = dp[state] + r[0]
                    par[state | sub] = (state, sub)
            sub = (sub - 1) & low
    routes, st = [], FULL
    while st:
        prev, sub = par[st]
        routes.append(best_route[sub][1])
        st = prev
    return routes, dp[FULL]

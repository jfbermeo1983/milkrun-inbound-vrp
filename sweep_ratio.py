# -*- coding: utf-8 -*-
"""
Designed experiment: the stop threshold against the load-to-capacity ratio.

Hypothesis: the threshold S* at which the marginal emission improvement is
exhausted coincides with Q / q_mean, that is, with the number of suppliers
whose combined load saturates the weight capacity of the vehicle. If this
holds, S* need not be tuned instance by instance but can be read off the
demand profile.

Design: the demands of each base instance are rescaled so that the mean load
per supplier yields a controlled predicted S*, and the stop limit is then swept
from 2 to 10. Rescaling the demands rather than changing the instances keeps
the geometry fixed, so any shift in the observed threshold is attributable to
the load-to-capacity ratio alone.
"""
import csv, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instances import catalogue                       # noqa: E402
from solver import Instance, Params, Model, solve     # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

BASE = ['A-n32-k5', 'VW-n40', 'A-n53-k7', 'A-n80-k10']
TARGETS = [2.5, 3.5, 4.5, 6.0, 8.0]      # predicted S* = Q / q_mean
S_VALUES = list(range(2, 11))
SEEDS = range(5)
Q = 24000.0


def rescale(inst, target):
    """Rescale the demands so that Q / q_mean equals the target ratio.

    Volumes are rescaled by the same factor when the instance carries them, so
    that the two capacity dimensions stay in proportion.
    """
    sup = [i for i in range(1, inst.n) if inst.q[i] > 0 or inst.v[i] > 0]
    qbar = sum(inst.q[i] for i in sup) / len(sup)
    f = (Q / target) / qbar
    q = [x * f for x in inst.q]
    v = [x * f for x in inst.v] if max(inst.v) > 0 else list(inst.v)
    return Instance(name=f"{inst.name}|S*={target}", D=inst.D, q=q, v=v)


def job(args):
    inst, target, S = args
    p = Params(cap_weight=Q,
               cap_volume=1e18,          # volume is made non-binding so that weight alone drives the result
               max_stops=S, include_outbound=False)
    sol, st = solve(inst, p, 'emis', seeds=SEEDS)
    m = Model(inst, p)
    sup = [i for i in range(1, inst.n) if inst.q[i] > 0]
    qbar = sum(inst.q[i] for i in sup) / len(sup)
    return {"base": inst.name.split('|')[0], "S_pred": target, "S": S,
            "q_medio": round(qbar), "rutas": len(sol),
            "co2": round(m.total(sol, 'emis'), 3),
            "km": round(m.total(sol, 'dist'), 2),
            "cv_pct": round(100 * st['sd'] / st['mean'], 3) if st['mean'] else 0}


if __name__ == "__main__":
    base = {i.name: i for i in catalogue()}
    jobs = [(rescale(base[b], t), t, S) for b in BASE for t in TARGETS for S in S_VALUES]
    print(f"{len(jobs)} configuraciones x {len(SEEDS)} semillas", flush=True)
    t0 = time.perf_counter()
    rows = []
    with Pool(os.cpu_count()) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r['base'], r['S_pred'], r['S']))
    with open(os.path.join(RESULTS, 'results_ratio.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"done ({time.perf_counter()-t0:.0f}s)")

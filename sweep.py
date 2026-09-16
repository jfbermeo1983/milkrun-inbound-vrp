# -*- coding: utf-8 -*-
"""
Full experimental battery.

Design:  8 instances x 5 values of the stop limit S x 2 objectives x 10 seeds,
         each run under both route conventions, that is, with and without the
         empty approach leg from the plant to the first supplier. Crossing the
         stop limit with the objective isolates how much of the emission gain
         comes from re-sequencing and how much from the routing structure
         itself, while the two conventions bound the accounting bias.

Output:  results_raw.csv   one row per (instance, S, objective, convention)
         validation.csv    tabu search against the exact optimum on small
                           sub-instances, which calibrates the solution gap
"""
import csv
import itertools
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instances import catalogue                 # noqa: E402
from solver import Instance, Params, Model, solve, exact_solve   # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

S_VALUES = [2, 3, 4, 5, None]          # None leaves the number of stops unrestricted
OBJECTIVES = ["dist", "emis"]
SEEDS = range(10)
OUT = os.path.dirname(os.path.abspath(__file__))


def make_params(inst, S, outbound):
    return Params(
        cap_weight=24000.0,
        cap_volume=100.0 if inst.name.startswith("VW") else 1e18,
        max_stops=inst.n if S is None else S,
        include_outbound=outbound,
    )


def job(args):
    inst, S, obj, outbound = args
    p = make_params(inst, S, outbound)
    sol, st = solve(inst, p, obj, seeds=SEEDS)
    m = Model(inst, p)
    return {
        "instance": inst.name,
        "n_nodes": inst.n,
        "S": "sin limite" if S is None else S,
        "objective": obj,
        "outbound": int(outbound),
        "routes": len(sol),
        "distance": round(m.total(sol, "dist"), 2),
        "co2": round(m.total(sol, "emis"), 3),
        "obj_best": round(st["best"], 3),
        "obj_mean": round(st["mean"], 3),
        "obj_sd": round(st["sd"], 3),
        "obj_worst": round(st["worst"], 3),
        "cv_pct": round(100 * st["sd"] / st["mean"], 3) if st["mean"] else 0,
        "time_per_seed_s": round(st["time_per_seed"], 3),
    }


def run_main():
    insts = catalogue()
    jobs = [(i, S, o, ob)
            for i in insts
            for S in S_VALUES
            for o in OBJECTIVES
            for ob in (True, False)]
    print(f"{len(jobs)} configuraciones x {len(SEEDS)} semillas = "
          f"{len(jobs)*len(SEEDS)} ejecuciones")
    t0 = time.perf_counter()
    with Pool(os.cpu_count()) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["n_nodes"], str(r["S"]), r["objective"], -r["outbound"]))
    path = os.path.join(RESULTS, "results_raw.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}  ({time.perf_counter()-t0:.0f}s)")
    return rows


def sub_instance(inst, k, name):
    """Extract the sub-instance formed by the first k suppliers plus the depot.

    These reduced instances are small enough for exact set partitioning and
    are used to measure the gap of the tabu search.
    """
    idx = list(range(k + 1))
    D = [[inst.D[a][b] for b in idx] for a in idx]
    return Instance(name=name, D=D, q=[inst.q[i] for i in idx], v=[inst.v[i] for i in idx])


def run_validation():
    insts = {i.name: i for i in catalogue()}
    base = insts["VW-n40"]
    rows = []
    for k in (6, 8, 10, 12):
        sub = sub_instance(base, k, f"VW-n{k+1}")
        for obj in OBJECTIVES:
            p = make_params(sub, 4, True)
            t0 = time.perf_counter()
            _, opt = exact_solve(sub, p, obj)
            t_ex = time.perf_counter() - t0
            sol, st = solve(sub, p, obj, seeds=SEEDS)
            gap = 100 * (st["best"] - opt) / opt if opt else 0.0
            rows.append({
                "instance": sub.name, "suppliers": k, "objective": obj,
                "exact_optimum": round(opt, 3), "exact_time_s": round(t_ex, 3),
                "ts_best": round(st["best"], 3), "ts_mean": round(st["mean"], 3),
                "ts_time_per_seed_s": round(st["time_per_seed"], 3),
                "gap_best_pct": round(gap, 4),
                "gap_mean_pct": round(100 * (st["mean"] - opt) / opt, 4) if opt else 0,
            })
            print(rows[-1], flush=True)
    path = os.path.join(RESULTS, "validation.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")
    return rows


if __name__ == "__main__":
    run_validation()
    run_main()

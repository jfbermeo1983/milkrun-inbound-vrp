# -*- coding: utf-8 -*-
"""
Extension of the main sweep to larger stop limits (S = 6, 7, 8).

The main battery stops at S = 5, which is enough to locate the threshold but
not to show that the marginal emission improvement stays negligible beyond it.
These runs use the emission objective and the closed-route convention, in which
the empty approach leg from the plant to the first supplier is charged to the
route, and they append to the main results with the same column layout.
"""
import csv, os, sys, time
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from instances import catalogue
from solver import Params, Model, solve

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
SEEDS=range(10)
def job(a):
    inst,S=a
    p=Params(cap_weight=24000.,cap_volume=100. if inst.name.startswith("VW") else 1e18,
             max_stops=S, include_outbound=True)
    sol,st=solve(inst,p,"emis",seeds=SEEDS); m=Model(inst,p)
    return {"instance":inst.name,"n_nodes":inst.n,"S":S,"objective":"emis","outbound":1,
            "routes":len(sol),"distance":round(m.total(sol,"dist"),2),"co2":round(m.total(sol,"emis"),3),
            "obj_best":round(st["best"],3),"obj_mean":round(st["mean"],3),"obj_sd":round(st["sd"],3),
            "obj_worst":round(st["worst"],3),
            "cv_pct":round(100*st["sd"]/st["mean"],3),"time_per_seed_s":round(st["time_per_seed"],3)}
if __name__=="__main__":
    jobs=[(i,S) for i in catalogue() for S in (6,7,8)]
    t0=time.perf_counter(); rows=[]
    with Pool(os.cpu_count()) as p:
        for k,r in enumerate(p.imap_unordered(job,jobs),1):
            rows.append(r); print(k,len(jobs),int(time.perf_counter()-t0),flush=True)
    with open(os.path.join(RESULTS, "results_extra_S.csv"), "w", newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("done")

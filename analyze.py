# -*- coding: utf-8 -*-
"""Build the tables reported in the article from the raw sweep output."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(HERE, "results", "results_raw.csv"))
val = pd.read_csv(os.path.join(HERE, "results", "validation.csv"))

S_ORDER = ["2", "3", "4", "5", "sin limite"]
raw["S"] = raw["S"].astype(str)
main = raw[raw.outbound == 1].copy()          # closed-route convention, the one recommended here


def pivot_dm_em(df):
    """Main table: distance and emissions under each objective.

    For every instance and stop limit S, the solution obtained under the
    distance-minimising objective (DM) is placed side by side with the one
    obtained under the emission-minimising objective (EM), so that the
    emission gain of EM and the distance penalty it incurs can be read off
    directly.
    """
    d = df[df.objective == "dist"].set_index(["instance", "S"])
    e = df[df.objective == "emis"].set_index(["instance", "S"])
    t = pd.DataFrame({
        "n": d["n_nodes"],
        "rutas_DM": d["routes"], "km_DM": d["distance"], "co2_DM": d["co2"],
        "rutas_EM": e["routes"], "km_EM": e["distance"], "co2_EM": e["co2"],
    })
    t["gap_co2_pct"] = 100 * (t.co2_DM - t.co2_EM) / t.co2_DM     # emission gain of EM over DM
    t["penal_km_pct"] = 100 * (t.km_EM - t.km_DM) / t.km_DM       # extra distance incurred by EM
    return t.reset_index()


tbl_main = pivot_dm_em(main)
tbl_main["_o"] = tbl_main.S.map({s: i for i, s in enumerate(S_ORDER)})
tbl_main = tbl_main.sort_values(["n", "_o"]).drop(columns="_o")

# --- effect of the stop limit: tightest setting (S=2) against no limit -----
rows = []
for inst, g in main.groupby("instance"):
    for obj, lab in (("dist", "DM"), ("emis", "EM")):
        gg = g[g.objective == obj].set_index("S")
        if "2" not in gg.index or "sin limite" not in gg.index:
            continue
        a, b = gg.loc["2"], gg.loc["sin limite"]
        rows.append({
            "instance": inst, "n": int(a.n_nodes), "objetivo": lab,
            "km_S2": a.distance, "km_libre": b.distance,
            "red_km_pct": round(100 * (a.distance - b.distance) / a.distance, 2),
            "co2_S2": a.co2, "co2_libre": b.co2,
            "red_co2_pct": round(100 * (a.co2 - b.co2) / a.co2, 2),
        })
tbl_stops = pd.DataFrame(rows).sort_values(["n", "objetivo"])

# --- stop-limit selection rule --------------------------------------------
# The threshold S* is the last stop limit whose marginal emission improvement
# still exceeds one per cent; beyond it, additional stops buy little and the
# operational cost of longer routes is no longer justified.
rows = []
for inst, g in main[main.objective == "emis"].groupby("instance"):
    gg = g.set_index("S")
    seq = [s for s in ["2", "3", "4", "5"] if s in gg.index]
    thr = None
    for a, b in zip(seq, seq[1:]):
        marg = 100 * (gg.loc[a].co2 - gg.loc[b].co2) / gg.loc[a].co2
        rows.append({"instance": inst, "n": int(gg.loc[a].n_nodes),
                     "de_S": a, "a_S": b, "mejora_marginal_co2_pct": round(marg, 3)})
        if thr is None and marg < 1.0:
            thr = int(a)
    rows[-1]["umbral"] = thr
tbl_marg = pd.DataFrame(rows)
umbrales = (tbl_marg.dropna(subset=["umbral"])[["instance", "umbral"]]
            if "umbral" in tbl_marg else pd.DataFrame())

# --- bias introduced by the open-route convention -------------------------
# Ignoring the empty approach leg understates the emissions actually incurred;
# the comparison below quantifies that understatement.
rows = []
for (inst, S, obj), g in raw.groupby(["instance", "S", "objective"]):
    if len(g) != 2:
        continue
    closed = g[g.outbound == 1].iloc[0]
    openr = g[g.outbound == 0].iloc[0]
    rows.append({
        "instance": inst, "n": int(closed.n_nodes), "S": S, "objetivo": obj,
        "co2_cerrada": closed.co2, "co2_abierta_reportada": openr.co2,
        "subestimacion_pct": round(100 * (closed.co2 - openr.co2) / closed.co2, 2),
        "rutas_cerrada": closed.routes, "rutas_abierta": openr.routes,
    })
tbl_open = pd.DataFrame(rows)
tbl_open["_o"] = tbl_open.S.map({s: i for i, s in enumerate(S_ORDER)})
tbl_open = tbl_open.sort_values(["n", "_o", "objetivo"]).drop(columns="_o")

# --- stability of the metaheuristic across seeds ---------------------------
tbl_stab = (main.groupby(["instance", "objective"])
            .agg(cv_medio_pct=("cv_pct", "mean"), cv_max_pct=("cv_pct", "max"),
                 t_medio_s=("time_per_seed_s", "mean"))
            .round(3).reset_index())

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\n=== TABU SEARCH VS CERTIFIED OPTIMUM ===");  print(val.to_string(index=False))
    print("\n=== MAIN TABLE (extract, S = 4) ===")
    print(tbl_main[tbl_main.S == "4"].to_string(index=False))
    print("\n=== EFFECT OF THE STOP LIMIT ===");    print(tbl_stops.to_string(index=False))
    print("\n=== MARGINAL CO2e REDUCTION PER ADDITIONAL STOP ===")
    print(tbl_marg.to_string(index=False))
    print("\n=== BIAS OF THE OPEN-ROUTE CONVENTION (S = 4) ===")
    print(tbl_open[tbl_open.S == "4"].to_string(index=False))
    print("\n=== STABILITY ACROSS SEEDS ===");                     print(tbl_stab.to_string(index=False))
    print("\n=== OVERALL SUMMARY ===")
    print(f"CO2e gain of EM over DM   : mean {tbl_main.gap_co2_pct.mean():.2f}% "
          f"| max {tbl_main.gap_co2_pct.max():.2f}% | min {tbl_main.gap_co2_pct.min():.2f}%")
    print(f"distance penalty of EM    : mean {tbl_main.penal_km_pct.mean():.2f}% "
          f"| max {tbl_main.penal_km_pct.max():.2f}%")

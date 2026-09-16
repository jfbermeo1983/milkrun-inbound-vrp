# -*- coding: utf-8 -*-
"""Instance loading: the real-world VW Navarra dataset and the Augerat (1995)
benchmark set used for external comparability."""
import math
import os
import re
from solver import Instance

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_vw(path, name):
    """Read a case-study file defining the blocks D = [...], q = [...], v = [...].

    D is the road-distance matrix between the plant and the suppliers, q the
    weight collected at each supplier and v the corresponding volume.
    """
    ns = {}
    with open(path) as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    return Instance(name=name, D=ns["D"], q=[float(x) for x in ns["q"]],
                    v=[float(x) for x in ns["v"]])


def load_augerat(path, cap_weight=24000.0):
    """Read an Augerat A-set file (EUC_2D distances).

    Benchmark demands are expressed in the file's own capacity units; they are
    rescaled to kilograms so that the stated vehicle capacity corresponds to
    cap_weight. This keeps the load-to-capacity ratio of each benchmark
    instance unchanged while making its payloads comparable with the
    case-study fleet.
    """
    txt = open(path).read()
    name = re.search(r"NAME\s*:\s*(\S+)", txt).group(1)
    cap = float(re.search(r"CAPACITY\s*:\s*(\d+)", txt).group(1))
    coords, dem = {}, {}
    sec = None
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("NODE_COORD_SECTION"):
            sec = "c"; continue
        if s.startswith("DEMAND_SECTION"):
            sec = "d"; continue
        if s.startswith("DEPOT_SECTION") or s.startswith("EOF"):
            sec = None; continue
        parts = s.split()
        if sec == "c" and len(parts) >= 3:
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif sec == "d" and len(parts) >= 2:
            dem[int(parts[0])] = float(parts[1])
    ids = sorted(coords)
    n = len(ids)
    D = [[0.0] * n for _ in range(n)]
    for a in range(n):
        xa, ya = coords[ids[a]]
        for b in range(n):
            xb, yb = coords[ids[b]]
            D[a][b] = math.hypot(xa - xb, ya - yb)
    scale = cap_weight / cap
    q = [dem[i] * scale for i in ids]
    v = [0.0] * n                      # the Augerat instances carry no volume data
    return Instance(name=name, D=D, q=q, v=v)


def catalogue():
    """Return every instance used in the study, ordered by number of nodes."""
    out = []
    for f in sorted(os.listdir(os.path.join(DATA, "augerat"))):
        if f.endswith(".vrp"):
            out.append(load_augerat(os.path.join(DATA, "augerat", f)))
    vw = load_vw(os.path.join(DATA, "vw40.py"), "VW-n40")
    out.append(vw)
    out.sort(key=lambda i: i.n)
    return out

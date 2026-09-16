# -*- coding: utf-8 -*-
"""
Milk-run inbound vehicle routing with a distance and a CO2e objective.

The problem is an open vehicle routing problem on a many-to-one collection
network. Each supplier has a truck stationed in its vicinity, so a route
starts at a supplier, visits the remaining suppliers assigned to it, and ends
at the plant. No vehicle travels from the plant to a supplier empty, which is
why the initial solution is built from unit routes.

Four constraints apply to every route: a weight capacity, a volume capacity, a
contractual cap on the number of stops, and a maximum admissible detour per
intermediate stop. Suppliers whose consignment falls below the consolidation
threshold are removed from the milk-run and sent through the consolidation
centre; they are not part of the routing problem.

Because the load accumulates monotonically along the route, the emission
factor of an arc depends on the routing decision itself. The objective is kept
linear by writing the factor as an affine function of the payload.

The script reports the initial solution, the tabu-search solution over several
independent seeds, and, when the number of routed suppliers allows it, the
certified optimum obtained by set partitioning, so that the gap of the
metaheuristic is measured rather than assumed.

Run:  python milkrun_vrp.py
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================================
#  EMISSION MODEL
# ============================================================================
# Fuel consumption of the articulated vehicle (24,000 kg maximum payload) as a
# function of its load factor. This is the only physical input of the model.
CONSUMO_L_100KM = [(0, 27.83), (25, 30.98), (50, 34.52), (75, 38.36), (100, 41.21)]

# Conversion factors for automotive diesel of average biofuel blend.
# TTW is direct combustion; WTT covers extraction, refining and distribution of
# the fuel; WTW is their sum and is the basis required by ISO 14083:2023.
FACTOR_TTW = 2.58354   # kg CO2e per litre
FACTOR_WTT = 0.61101   # kg CO2e per litre
FACTOR_WTW = FACTOR_TTW + FACTOR_WTT

# National cross-check for the direct component, used to show that the choice
# of source is not material for the conclusions.
FACTOR_TTW_ES = 2.52

BASE_REPORTE = "WTW"   # basis entering the objective function: "TTW" or "WTW"


def ajustar_factor_afin(cap_kg: float, consumo=CONSUMO_L_100KM, factor=FACTOR_WTW):
    """Fit e(w) = E0 + GAMMA * w by ordinary least squares on the consumption table.

    Returns (E0, GAMMA, R2, maximum absolute residual), where E0 is the
    emission intensity of the unladen vehicle in kg CO2e per km and GAMMA the
    marginal emission per kilogram carried per kilometre.

    The decomposition is what makes the emission objective linear. It also has
    a consequence worth stating: both coefficients derive from the same
    consumption profile through a single multiplicative conversion factor, so
    changing that factor rescales E0 and GAMMA by the same amount and
    multiplies the objective by a constant. The optimal plan is therefore
    invariant to the conversion factor and to the choice between TTW and WTW;
    only the absolute emission figures change. What does affect the optimal
    plan is the shape of the consumption profile, that is, the ratio between
    unladen and fully laden consumption.
    """
    xs = [cap_kg * p / 100.0 for p, _ in consumo]
    ys = [c / 100.0 * factor for _, c in consumo]      # l/100 km -> kg CO2e/km
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    g = sxy / sxx
    e0 = my - g * mx
    pred = [e0 + g * x for x in xs]
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - p) ** 2 for y, p in zip(ys, pred))
    r2 = 1.0 - sse / sst
    resid = max(abs(y - p) for y, p in zip(ys, pred))
    return e0, g, r2, resid


# ============================================================================
#  PARAMETERS
# ============================================================================
CV_WEIGHT = 24000      # vehicle weight capacity (kg)
CV_VOLUME = 100        # vehicle volume capacity (m3)
MAX_NODOS = 4          # S, contractual cap on stops per route
DELTA_MAX = 300        # maximum admissible detour per intermediate stop (km); None to disable

# Consolidation rule. A consignment below the threshold is diverted to the
# consolidation centre and leaves the routing problem. "weight" applies the
# weight condition alone, which is the operating directive of the plant;
# "and" and "or" combine it with the volume condition for sensitivity runs.
MIN_WEIGHT_VISIT = 6000    # kg
MIN_VOLUME_VISIT = 3       # m3
CONSOL_POLICY = "weight"   # "weight" | "and" | "or"

# System boundary. False places the vehicle at the supplier where its route
# starts, which is the operating model of this network. True adds the empty
# leg from the plant to the first supplier, appropriate to a plant-owned fleet.
INCLUDE_OUTBOUND = False

# Objective: 'distance' | 'emissions' | 'hybrid'
OBJECTIVE = 'emissions'
ALPHA_DIST = 1.0
BETA_EMIS = 1.0

# Tabu search. Several independent seeds are run and their dispersion reported,
# so that the result does not depend on a single trajectory.
TENURE = 15
ITERS = 1200
SAMPLE_MOVES = 250
STALL_LIMIT = 250
SEEDS = [1001, 1002, 1003, 1004, 1005]
START_MODE = 'cw'      # 'trivial' | 'cw'
CW_NOISE = 0.15

DEPOT = 0


# ============================================================================
#  DATA
# ============================================================================
# Case-study network: the plant (node 0) and its 39 suppliers.
# D holds road distances in km, q the consignment weights in kg, v the volumes in m3.
D = [
  [0, 2149, 2152, 2202, 2346, 2556, 2399, 2341, 2296, 2221, 2939, 2040, 1889, 2033, 2425, 1778, 2529, 2949, 2144, 2462, 1802, 2425, 2347, 2018, 2215, 2161, 1777, 2416, 2416, 2425, 2425, 2370, 2110, 2342, 2156, 2040, 2909, 1822, 2264, 2999],
  [2149, 0, 736, 97, 232, 452, 964, 236, 435, 782, 1504, 743, 836, 164, 990, 447, 830, 1818, 603, 692, 968, 1076, 302, 249, 776, 607, 989, 311, 659, 990, 990, 324, 654, 296, 734, 743, 1473, 400, 160, 1564],
  [2152, 736, 0, 664, 625, 568, 259, 510, 212, 81, 799, 168, 269, 601, 285, 637, 389, 1113, 145, 322, 459, 371, 460, 913, 75, 198, 433, 611, 276, 285, 285, 439, 181, 472, 6, 168, 769, 590, 591, 859],
  [2202, 97, 664, 0, 168, 379, 891, 164, 362, 709, 1431, 670, 763, 193, 917, 501, 757, 1745, 531, 619, 895, 1003, 229, 344, 704, 535, 916, 238, 586, 917, 917, 251, 581, 223, 661, 670, 1401, 453, 87, 1491],
  [2346, 232, 625, 168, 0, 272, 847, 120, 307, 665, 1387, 627, 719, 340, 873, 679, 651, 1702, 487, 513, 851, 959, 134, 515, 660, 491, 872, 69, 479, 873, 873, 156, 537, 128, 618, 627, 1357, 631, 127, 1447],
  [2556, 452, 568, 379, 272, 0, 664, 248, 369, 477, 996, 754, 847, 548, 690, 807, 387, 1519, 615, 335, 979, 777, 257, 640, 483, 464, 1000, 148, 305, 690, 690, 212, 665, 251, 561, 754, 965, 759, 300, 1008],
  [2399, 964, 259, 891, 847, 664, 0, 733, 397, 194, 545, 423, 521, 824, 34, 860, 254, 936, 367, 253, 711, 117, 617, 1135, 196, 420, 685, 833, 372, 34, 34, 597, 403, 630, 252, 423, 515, 812, 813, 605],
  [2341, 236, 510, 164, 120, 248, 733, 0, 203, 551, 1272, 512, 604, 335, 758, 564, 625, 1587, 372, 487, 737, 844, 58, 486, 545, 376, 757, 106, 454, 758, 758, 80, 422, 52, 503, 512, 1242, 516, 86, 1332],
  [2296, 435, 212, 362, 307, 369, 397, 203, 0, 206, 938, 341, 498, 488, 424, 524, 413, 1252, 202, 345, 631, 510, 167, 722, 202, 151, 651, 280, 229, 424, 424, 146, 252, 179, 205, 341, 908, 476, 321, 998],
  [2221, 782, 81, 709, 665, 477, 194, 551, 206, 0, 735, 239, 341, 640, 221, 676, 299, 1050, 184, 231, 531, 307, 434, 952, 13, 237, 505, 650, 186, 221, 221, 414, 220, 446, 72, 239, 705, 629, 630, 795],
  [2939, 1504, 799, 1431, 1387, 996, 545, 1272, 938, 735, 0, 962, 1060, 1363, 514, 1399, 538, 484, 907, 621, 1250, 450, 1157, 1675, 735, 960, 1224, 1373, 704, 514, 514, 1136, 943, 1169, 791, 962, 64, 1351, 1352, 109],
  [2040, 743, 168, 670, 627, 754, 423, 512, 341, 239, 962, 0, 148, 602, 447, 647, 552, 1276, 127, 484, 320, 533, 460, 914, 234, 199, 341, 612, 439, 447, 447, 440, 66, 473, 165, 0, 931, 590, 591, 1021],
  [1889, 836, 269, 763, 719, 847, 521, 604, 498, 341, 1060, 148, 0, 638, 536, 590, 640, 1201, 339, 572, 201, 622, 615, 1055, 326, 356, 175, 704, 527, 536, 536, 597, 249, 609, 267, 138, 1019, 523, 684, 1110],
  [2033, 164, 601, 193, 340, 548, 824, 335, 488, 640, 1363, 602, 638, 0, 849, 261, 954, 1678, 463, 886, 705, 935, 397, 288, 636, 467, 731, 407, 841, 849, 849, 420, 513, 391, 594, 603, 1333, 213, 256, 1423],
  [2425, 990, 285, 917, 873, 690, 34, 758, 424, 221, 514, 447, 536, 849, 0, 886, 275, 853, 393, 443, 737, 60, 644, 1161, 222, 446, 711, 859, 398, 20, 20, 623, 430, 656, 278, 449, 484, 838, 839, 574],
  [1778, 447, 637, 501, 679, 807, 860, 564, 524, 676, 1399, 647, 590, 261, 886, 0, 987, 1711, 497, 920, 593, 969, 574, 529, 669, 500, 619, 664, 874, 883, 883, 596, 547, 568, 627, 639, 1367, 48, 444, 1457],
  [2529, 830, 389, 757, 651, 387, 254, 625, 413, 299, 538, 552, 640, 954, 275, 987, 0, 1057, 498, 84, 842, 264, 568, 1076, 306, 551, 816, 580, 183, 274, 274, 547, 535, 577, 383, 554, 507, 943, 676, 556],
  [2949, 1818, 1113, 1745, 1702, 1519, 936, 1587, 1252, 1050, 484, 1276, 1201, 1678, 853, 1711, 1057, 0, 1221, 1270, 1269, 791, 1471, 1988, 1049, 1273, 1243, 1686, 1225, 852, 852, 1450, 1257, 1483, 1105, 1276, 542, 1665, 1666, 495],
  [2144, 603, 145, 531, 487, 615, 367, 372, 202, 184, 907, 127, 339, 463, 393, 497, 498, 1221, 0, 430, 472, 479, 321, 774, 180, 59, 492, 472, 385, 393, 393, 300, 93, 333, 137, 128, 877, 450, 451, 967],
  [2462, 692, 322, 619, 513, 335, 253, 487, 345, 231, 621, 484, 572, 886, 443, 920, 84, 1270, 430, 0, 774, 305, 429, 937, 238, 483, 748, 441, 115, 445, 445, 408, 467, 438, 315, 486, 591, 875, 537, 640],
  [1802, 968, 459, 895, 851, 979, 711, 737, 631, 531, 1250, 320, 201, 705, 737, 593, 842, 1269, 472, 774, 0, 823, 745, 1030, 527, 486, 33, 835, 729, 737, 737, 728, 379, 739, 468, 317, 1221, 639, 814, 1311],
  [2425, 1076, 371, 1003, 959, 777, 117, 844, 510, 307, 450, 533, 622, 935, 60, 969, 264, 791, 479, 305, 823, 0, 731, 1249, 310, 534, 799, 947, 486, 61, 61, 711, 517, 743, 366, 536, 421, 925, 926, 511],
  [2347, 302, 460, 229, 134, 257, 617, 58, 167, 434, 1157, 460, 615, 397, 644, 574, 568, 1471, 321, 429, 745, 731, 0, 548, 430, 271, 769, 112, 398, 644, 644, 23, 372, 11, 407, 461, 1127, 528, 147, 1218],
  [2018, 249, 913, 344, 515, 640, 1135, 486, 722, 952, 1675, 914, 1055, 288, 1161, 529, 1076, 1988, 774, 937, 1030, 1249, 548, 0, 945, 776, 1057, 561, 897, 1158, 1158, 563, 822, 534, 903, 912, 1642, 510, 399, 1732],
  [2215, 776, 75, 704, 660, 483, 196, 545, 202, 13, 735, 234, 326, 636, 222, 669, 306, 1049, 180, 238, 527, 310, 430, 945, 0, 237, 502, 650, 192, 221, 221, 414, 220, 446, 69, 240, 705, 629, 630, 795],
  [2161, 607, 198, 535, 491, 464, 420, 376, 151, 237, 960, 199, 356, 467, 446, 500, 551, 1273, 59, 483, 486, 534, 271, 776, 237, 0, 513, 384, 320, 449, 449, 251, 113, 283, 194, 203, 933, 458, 391, 1023],
  [1777, 989, 433, 916, 872, 1000, 685, 757, 651, 505, 1224, 341, 175, 731, 711, 619, 816, 1243, 492, 748, 33, 799, 769, 1057, 502, 513, 0, 859, 703, 711, 711, 752, 403, 763, 442, 341, 1195, 667, 838, 1285],
  [2416, 311, 611, 238, 69, 148, 833, 106, 280, 650, 1373, 612, 704, 407, 859, 664, 580, 1686, 472, 441, 835, 947, 112, 561, 650, 384, 859, 0, 410, 859, 859, 137, 523, 108, 604, 612, 1343, 617, 158, 1433],
  [2416, 659, 276, 586, 479, 305, 372, 454, 229, 186, 704, 439, 527, 841, 398, 874, 183, 1225, 385, 115, 729, 486, 398, 897, 192, 320, 703, 410, 0, 399, 399, 378, 421, 408, 270, 441, 674, 830, 506, 716],
  [2425, 990, 285, 917, 873, 690, 34, 758, 424, 221, 514, 447, 536, 849, 20, 883, 274, 852, 393, 445, 737, 61, 644, 1158, 221, 449, 711, 859, 399, 0, 0, 623, 430, 656, 278, 449, 484, 838, 839, 574],
  [2425, 990, 285, 917, 873, 690, 34, 758, 424, 221, 514, 447, 536, 849, 20, 883, 274, 852, 393, 445, 737, 61, 644, 1158, 221, 449, 711, 859, 399, 0, 0, 623, 430, 656, 278, 449, 484, 838, 839, 574],
  [2370, 324, 439, 251, 156, 212, 597, 80, 146, 414, 1136, 440, 597, 420, 623, 596, 547, 1450, 300, 408, 728, 711, 23, 563, 414, 251, 752, 137, 378, 623, 623, 0, 352, 33, 388, 442, 1108, 550, 171, 1198],
  [2110, 654, 181, 581, 537, 665, 403, 422, 252, 220, 943, 66, 249, 513, 430, 547, 535, 1257, 93, 467, 379, 517, 372, 822, 220, 113, 403, 523, 421, 430, 430, 352, 0, 384, 173, 67, 912, 501, 502, 1002],
  [2342, 296, 472, 223, 128, 251, 630, 52, 179, 446, 1169, 473, 609, 391, 656, 568, 577, 1483, 333, 438, 739, 743, 11, 534, 446, 283, 763, 108, 408, 656, 656, 33, 384, 0, 419, 517, 1139, 522, 143, 1230],
  [2156, 734, 6, 661, 618, 561, 252, 503, 205, 72, 791, 165, 267, 594, 278, 627, 383, 1105, 137, 315, 468, 366, 407, 903, 69, 194, 442, 604, 270, 278, 278, 388, 173, 419, 0, 163, 761, 582, 583, 851],
  [2040, 743, 168, 670, 627, 754, 423, 512, 341, 239, 962, 0, 138, 603, 449, 639, 554, 1276, 128, 486, 317, 536, 461, 912, 240, 203, 341, 612, 441, 449, 449, 442, 67, 517, 163, 0, 931, 590, 591, 1021],
  [2909, 1473, 769, 1401, 1357, 965, 515, 1242, 908, 705, 64, 931, 1019, 1333, 484, 1367, 507, 542, 877, 591, 1221, 421, 1127, 1642, 705, 933, 1195, 1343, 674, 484, 484, 1108, 912, 1139, 761, 931, 0, 1320, 1321, 108],
  [1822, 400, 590, 453, 631, 759, 812, 516, 476, 629, 1351, 590, 523, 213, 838, 48, 943, 1665, 450, 875, 639, 925, 528, 510, 629, 458, 667, 617, 830, 838, 838, 550, 501, 522, 582, 590, 1320, 0, 394, 1407],
  [2264, 160, 591, 87, 127, 300, 813, 86, 321, 630, 1352, 591, 684, 256, 839, 444, 676, 1666, 451, 537, 814, 926, 147, 399, 630, 391, 838, 158, 506, 839, 839, 171, 502, 143, 583, 591, 1321, 394, 0, 1411],
  [2999, 1564, 859, 1491, 1447, 1008, 605, 1332, 998, 795, 109, 1021, 1110, 1423, 574, 1457, 556, 495, 967, 640, 1311, 511, 1218, 1732, 795, 1023, 1285, 1433, 716, 574, 574, 1198, 1002, 1230, 851, 1021, 108, 1407, 1411, 0],
]

# ----- PESO (Q) -----
q = [0, 16818, 140, 4015, 12015, 13997, 7279, 10691, 16384, 5136, 2515, 3092, 5706, 5177, 7854, 2191, 1103, 1648, 3692, 1451, 244, 5136, 4538, 6786, 2032, 3075, 233, 1391, 16390, 4132, 2169, 3975, 3630, 1651, 4029, 8184, 2413, 24000, 11409, 2904]

# ----- VOLUMEN (V) -----
v = [0, 3, 0, 14, 43, 50, 58, 46, 92, 58, 29, 10, 17, 34, 99, 15, 8, 10, 29, 11, 1, 44, 27, 37, 1, 65, 1, 7, 26, 73, 3, 29, 1, 8, 34, 68, 26, 99, 52, 15]

Q = q
V = v
assert len(D) == len(Q) == len(V), "Inconsistent dimensions between D, Q and V"

E0, GAMMA, R2_FIT, RESID_FIT = ajustar_factor_afin(
    CV_WEIGHT, factor=(FACTOR_WTW if BASE_REPORTE == "WTW" else FACTOR_TTW))
E0_TTW, GAMMA_TTW, _, _ = ajustar_factor_afin(CV_WEIGHT, factor=FACTOR_TTW)
E0_WTW, GAMMA_WTW, _, _ = ajustar_factor_afin(CV_WEIGHT, factor=FACTOR_WTW)


# ============================================================================
#  SUPPLIER CLASSIFICATION
# ============================================================================
def split_suppliers(Q, V, min_w, min_v, policy=CONSOL_POLICY):
    """Split the supplier base into routed, consolidated and inactive suppliers.

    Suppliers with neither weight nor volume are inactive in the period and are
    excluded from both sets.
    """
    sup_all = list(range(1, len(Q)))
    sup_zero = [i for i in sup_all if Q[i] == 0 and V[i] == 0]
    rest = [i for i in sup_all if i not in sup_zero]

    p = policy.lower()
    if p == "weight":
        sup_conso = [i for i in rest if Q[i] < min_w]
    elif p == "and":
        sup_conso = [i for i in rest if Q[i] < min_w and V[i] < min_v]
    elif p == "or":
        sup_conso = [i for i in rest if Q[i] < min_w or V[i] < min_v]
    else:
        raise ValueError("CONSOL_POLICY must be 'weight', 'and' or 'or'")

    sup_visit = [i for i in rest if i not in sup_conso]
    return sup_visit, sup_conso, sup_zero


Sup_visit, Sup_conso, Sup_zero = split_suppliers(Q, V, MIN_WEIGHT_VISIT, MIN_VOLUME_VISIT)

# One stationed vehicle per routed supplier is available; which of them is
# activated is itself a decision, and a solution never uses more than this.
T_MAX_VEHICLES = max(1, len(Sup_visit))
MIN_ROUTES = 1


# ============================================================================
#  ROUTE EVALUATION
# ============================================================================
def route_distance(seq: List[int]) -> float:
    """Distance of an open route: from the first supplier through the sequence
    to the plant. The leg from the plant to the first supplier is added only
    when the system boundary includes vehicle repositioning."""
    if not seq:
        return 0.0
    d = D[DEPOT][seq[0]] if INCLUDE_OUTBOUND else 0.0
    for a, b in zip(seq, seq[1:]):
        d += D[a][b]
    d += D[seq[-1]][DEPOT]
    return d


def weight_of(seq):
    return sum(Q[i] for i in seq)


def volume_of(seq):
    return sum(V[i] for i in seq)


def check_delta_constraints(seq: List[int]) -> bool:
    """Maximum detour rule: inserting a stop between two consecutive nodes may
    not lengthen that leg by more than DELTA_MAX kilometres."""
    if DELTA_MAX is None or len(seq) <= 1:
        return True
    ext = seq + [DEPOT]
    for t in range(1, len(ext) - 1):
        i, j, k = ext[t - 1], ext[t], ext[t + 1]
        if D[i][j] + D[j][k] - D[i][k] > DELTA_MAX + 1e-9:
            return False
    return True


def route_feasible(seq: List[int]) -> bool:
    return (len(seq) <= MAX_NODOS and
            weight_of(seq) <= CV_WEIGHT and
            volume_of(seq) <= CV_VOLUME and
            check_delta_constraints(seq))


def route_emissions(seq: List[int], e0: float = None, gamma: float = None) -> float:
    """Emissions of a collection route, in kg CO2e.

    The payload grows monotonically along the route, so the emission factor
    differs from arc to arc. On the arc leaving node a the vehicle already
    carries the consignments of every supplier visited up to and including a;
    the final leg into the plant carries the full load.
    """
    if not seq:
        return 0.0
    e0 = E0 if e0 is None else e0
    gamma = GAMMA if gamma is None else gamma
    em = 0.0
    if INCLUDE_OUTBOUND:
        em += D[DEPOT][seq[0]] * e0
    load = 0.0
    for a, b in zip(seq, seq[1:]):
        load += Q[a]
        em += D[a][b] * (e0 + gamma * load)
    load += Q[seq[-1]]
    em += D[seq[-1]][DEPOT] * (e0 + gamma * load)
    return em


def total_distance(routes) -> float:
    return sum(route_distance(r.seq) for r in routes)


def total_emissions(routes, e0=None, gamma=None) -> float:
    return sum(route_emissions(r.seq, e0, gamma) for r in routes)


def route_cost(seq: List[int]) -> float:
    if OBJECTIVE == 'distance':
        return route_distance(seq)
    if OBJECTIVE == 'emissions':
        return route_emissions(seq)
    return ALPHA_DIST * route_distance(seq) + BETA_EMIS * route_emissions(seq)


def total_cost(routes) -> float:
    return sum(route_cost(r.seq) for r in routes)


@dataclass
class Route:
    seq: List[int] = field(default_factory=list)

    @property
    def load_w(self):
        return weight_of(self.seq)

    @property
    def load_v(self):
        return volume_of(self.seq)

    @property
    def dist(self):
        return route_distance(self.seq)


# ============================================================================
#  INITIAL SOLUTION
# ============================================================================
def trivial_initial() -> List[Route]:
    """One unit route per routed supplier: every stationed truck serves its own
    supplier and drives to the plant. This is the feasible starting point
    implied by the operating model."""
    return [Route([i]) for i in Sup_visit]


def clarke_wright_initial(cw_noise: float = 0.0, seed: Optional[int] = None) -> List[Route]:
    """Clarke and Wright savings heuristic adapted to open routes.

    The saving is directed, from the tail of one route to the head of another:
    appending j after i saves the return leg of i to the plant and costs the
    arc from i to j. Route origin is free and only the destination is fixed,
    which is why the classical symmetric saving does not apply here.

    The initial solution is built on distance even when the objective is
    emissions, so that the starting point does not favour either objective in
    the comparison; the tabu search and the 2-opt then adapt it.
    """
    rnd = random.Random(seed)
    routes = [Route([i]) for i in Sup_visit]
    pos = {i: idx for idx, i in enumerate(Sup_visit)}

    savings = []
    for i in Sup_visit:
        for j in Sup_visit:
            if i == j:
                continue
            s = D[i][DEPOT] - D[i][j]
            if cw_noise > 0:
                s *= (1.0 + rnd.uniform(-cw_noise, cw_noise))
            savings.append((s, i, j))
    savings.sort(key=lambda t: (-t[0], t[1], t[2]))

    for s, i, j in savings:
        ri, rj = pos.get(i), pos.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        if routes[ri].seq[-1] != i or routes[rj].seq[0] != j:
            continue
        new_seq = routes[ri].seq + routes[rj].seq
        if not route_feasible(new_seq):
            continue
        routes[ri].seq = new_seq
        routes[rj].seq = []
        for c in new_seq:
            pos[c] = ri
    return [r for r in routes if r.seq]


# ============================================================================
#  INTRA-ROUTE 2-OPT
# ============================================================================
def two_opt_best_improvement(route: Route) -> bool:
    """Best-improvement 2-opt on an open route.

    Both the first and the last position may move, because the origin of the
    route is free and only the plant is fixed. The move is evaluated with the
    active objective, so the operator intensifies on emissions or on distance
    according to the run.
    """
    seq = route.seq
    n = len(seq)
    if n < 2:
        return False
    base = route_cost(seq)
    best_gain, best_move = 1e-9, None
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            new_seq = seq[:i] + list(reversed(seq[i:j + 1])) + seq[j + 1:]
            if not route_feasible(new_seq):
                continue
            gain = base - route_cost(new_seq)
            if gain > best_gain:
                best_gain, best_move = gain, new_seq
    if best_move:
        route.seq = best_move
        return True
    return False


def two_opt_local_search(routes: List[Route]) -> None:
    improved = True
    while improved:
        improved = False
        for r in routes:
            if two_opt_best_improvement(r):
                improved = True


# ============================================================================
#  NEIGHBOURHOODS
# ============================================================================
# Three lambda-interchange operators act between routes: SHIFT relocates one
# supplier, SWAP exchanges two, and OR-OPT2 relocates a block of two
# consecutive suppliers. Every candidate is checked against all four route
# constraints before it is evaluated.

def all_shift_moves(routes):
    return [(ri, pi, rj, pj)
            for ri, r in enumerate(routes)
            for pi in range(len(r.seq))
            for rj, s in enumerate(routes) if rj != ri
            for pj in range(len(s.seq) + 1)]


def apply_shift(routes, move):
    ri, pi, rj, pj = move
    new = [Route(r.seq.copy()) for r in routes]
    u = new[ri].seq.pop(pi)
    new[rj].seq.insert(pj, u)
    new = [r for r in new if r.seq]
    if not (MIN_ROUTES <= len(new) <= T_MAX_VEHICLES):
        return None
    return new if all(route_feasible(r.seq) for r in new) else None


def all_swap_moves(routes):
    return [(ri, pi, rj, pj)
            for ri, r in enumerate(routes)
            for rj, s in enumerate(routes) if rj > ri
            for pi in range(len(r.seq))
            for pj in range(len(s.seq))]


def apply_swap(routes, move):
    ri, pi, rj, pj = move
    new = [Route(r.seq.copy()) for r in routes]
    new[ri].seq[pi], new[rj].seq[pj] = new[rj].seq[pj], new[ri].seq[pi]
    if not (MIN_ROUTES <= len(new) <= T_MAX_VEHICLES):
        return None
    return new if all(route_feasible(r.seq) for r in new) else None


def all_oropt2_moves(routes):
    return [(ri, pi, rj, pj)
            for ri, r in enumerate(routes) if len(r.seq) >= 2
            for pi in range(len(r.seq) - 1)
            for rj, s in enumerate(routes) if rj != ri
            for pj in range(len(s.seq) + 1)]


def apply_oropt2(routes, move):
    ri, pi, rj, pj = move
    new = [Route(r.seq.copy()) for r in routes]
    if pi >= len(new[ri].seq) - 1:
        return None
    block = [new[ri].seq.pop(pi), new[ri].seq.pop(pi)]
    for k, u in enumerate(block):
        new[rj].seq.insert(pj + k, u)
    new = [r for r in new if r.seq]
    if not (MIN_ROUTES <= len(new) <= T_MAX_VEHICLES):
        return None
    return new if all(route_feasible(r.seq) for r in new) else None


# ============================================================================
#  TABU SEARCH
# ============================================================================
def tabu_search(initial_routes, seed=1001, tenure=TENURE, iters=ITERS,
                sample_moves=SAMPLE_MOVES, stall_limit=STALL_LIMIT, verbose=False):
    """Tabu search over the three neighbourhoods, with 2-opt intensification.

    A move is recorded in the tabu list by the supplier it displaces and the
    route it enters, and stays forbidden for TENURE iterations. The aspiration
    criterion overrides the list whenever a forbidden move would improve the
    best solution found so far. The search stops when no admissible neighbour
    exists, when the iteration budget is exhausted, or after STALL_LIMIT
    iterations without improvement.

    Randomness is drawn from a generator seeded per run, so a run is fully
    reproducible from its seed.
    """
    rnd = random.Random(seed)
    cur = [Route(r.seq.copy()) for r in initial_routes]
    best = [Route(r.seq.copy()) for r in initial_routes]
    best_cost = total_cost(best)
    tabu = {}
    no_improve = 0

    for it in range(1, iters + 1):
        for k in list(tabu):
            tabu[k] -= 1
            if tabu[k] <= 0:
                tabu.pop(k, None)

        # The neighbourhood is sampled rather than enumerated, which keeps the
        # cost of an iteration bounded as the number of routes grows.
        cand_shift = all_shift_moves(cur)
        cand_swap = all_swap_moves(cur)
        cand_or2 = all_oropt2_moves(cur)
        if sample_moves is not None:
            rnd.shuffle(cand_shift); cand_shift = cand_shift[:sample_moves]
            rnd.shuffle(cand_swap);  cand_swap = cand_swap[:sample_moves]
            rnd.shuffle(cand_or2);   cand_or2 = cand_or2[:sample_moves]

        candidates = ([('shift', m) for m in cand_shift] +
                      [('swap', m) for m in cand_swap] +
                      [('or2', m) for m in cand_or2])

        best_neigh, best_neigh_cost, best_key = None, math.inf, None
        for kind, mv in candidates:
            ri, pi, rj, pj = mv
            if kind == 'shift':
                key = ('shift', cur[ri].seq[pi], rj)
                new_routes = apply_shift(cur, mv)
            elif kind == 'swap':
                u, w = cur[ri].seq[pi], cur[rj].seq[pj]
                key = ('swap', min(u, w), max(u, w))
                new_routes = apply_swap(cur, mv)
            else:
                key = ('or2', cur[ri].seq[pi], rj)
                new_routes = apply_oropt2(cur, mv)
            if new_routes is None:
                continue

            two_opt_local_search(new_routes)
            new_cost = total_cost(new_routes)

            if key in tabu and new_cost >= best_cost - 1e-9:
                continue
            if new_cost < best_neigh_cost - 1e-9:
                best_neigh, best_neigh_cost, best_key = new_routes, new_cost, key

        if best_neigh is None:
            break

        cur = best_neigh
        if best_key is not None:
            tabu[best_key] = tenure

        if best_neigh_cost < best_cost - 1e-9:
            best = [Route(r.seq.copy()) for r in cur]
            best_cost = best_neigh_cost
            no_improve = 0
        else:
            no_improve += 1

        if verbose and it % 50 == 0:
            print(f"  [it {it:4d}] best={best_cost:.4f}  routes={len(cur)}"
                  f"  km={total_distance(cur):.0f}  CO2e={total_emissions(cur):.1f}")

        if no_improve >= stall_limit:
            break

    return best, best_cost


def solve(seeds=SEEDS, verbose=False):
    """Multi-start tabu search. Returns the best solution together with the
    mean and standard deviation of the objective across seeds, which is what
    quantifies the stability of the metaheuristic."""
    results = []
    best_sol, best_val = None, math.inf
    for sd in seeds:
        init = (trivial_initial() if START_MODE == 'trivial'
                else clarke_wright_initial(cw_noise=CW_NOISE, seed=sd))
        two_opt_local_search(init)
        sol, val = tabu_search(init, seed=sd, verbose=verbose)
        results.append(val)
        if val < best_val:
            best_sol, best_val = sol, val
    mean = sum(results) / len(results)
    sd_ = (sum((x - mean) ** 2 for x in results) / len(results)) ** 0.5
    return best_sol, best_val, mean, sd_, results


# ============================================================================
#  CERTIFIED OPTIMUM BY SET PARTITIONING
# ============================================================================
def exact_solve(max_nodes: int = 14):
    """Solve the instance to proven optimality.

    Every feasible subset of routed suppliers is enumerated and its optimal
    visiting sequence obtained by complete permutation; the partition of the
    supplier set into routes is then solved by dynamic programming over bit
    masks. Fixing the lowest-indexed free supplier in each state avoids
    generating the same partition more than once.

    The procedure is exponential and is applied only when the number of routed
    suppliers is small enough. Its purpose is to measure the true gap of the
    tabu search rather than to solve large instances.
    """
    from itertools import permutations
    nodes = list(Sup_visit)
    n = len(nodes)
    if n > max_nodes:
        return None, None

    best_sub = {}
    for mask in range(1, 1 << n):
        sub = [nodes[k] for k in range(n) if mask >> k & 1]
        if len(sub) > MAX_NODOS:
            continue
        if weight_of(sub) > CV_WEIGHT or volume_of(sub) > CV_VOLUME:
            continue
        bc, bs = math.inf, None
        for perm in permutations(sub):
            p = list(perm)
            if not check_delta_constraints(p):
                continue
            c = route_cost(p)
            if c < bc:
                bc, bs = c, p
        if bs is not None:
            best_sub[mask] = (bc, bs)

    FULL = (1 << n) - 1
    dp = [math.inf] * (1 << n)
    choice = [None] * (1 << n)
    dp[0] = 0.0
    for mask in range(1 << n):
        if dp[mask] == math.inf:
            continue
        free = FULL & ~mask
        if free == 0:
            continue
        low = free & -free
        sub = free
        while sub:
            if sub & low and sub in best_sub:
                c = dp[mask] + best_sub[sub][0]
                if c < dp[mask | sub] - 1e-12:
                    dp[mask | sub] = c
                    choice[mask | sub] = (mask, sub)
            sub = (sub - 1) & free

    if dp[FULL] == math.inf:
        return None, None
    routes, m = [], FULL
    while m:
        prev, sub = choice[m]
        routes.append(Route(best_sub[sub][1]))
        m = prev
    return routes[::-1], dp[FULL]


# ============================================================================
#  REPORTING
# ============================================================================
def print_header():
    print("=" * 78)
    print("EMISSION MODEL")
    print("=" * 78)
    print(f"  Reporting basis            : {BASE_REPORTE}")
    print(f"  Diesel factor, TTW         : {FACTOR_TTW:.5f} kg CO2e/l")
    print(f"  Diesel factor, WTT         : {FACTOR_WTT:.5f} kg CO2e/l")
    print(f"  Diesel factor, WTW         : {FACTOR_WTW:.5f} kg CO2e/l")
    print(f"  National cross-check       : {FACTOR_TTW_ES:.2f} kg CO2e/l"
          f"   (deviation {100*(FACTOR_TTW-FACTOR_TTW_ES)/FACTOR_TTW:.1f}%)")
    print()
    print("  e(w) = E0 + GAMMA * w      [kg CO2e/km]")
    print(f"    TTW : E0 = {E0_TTW:.4f}   GAMMA = {GAMMA_TTW:.4e}")
    print(f"    WTW : E0 = {E0_WTW:.4f}   GAMMA = {GAMMA_WTW:.4e}")
    print(f"    R2 = {R2_FIT:.4f}   maximum residual = {RESID_FIT:.4f} kg CO2e/km")
    print()
    print(f"  System boundary            : "
          f"{'plant to first supplier included' if INCLUDE_OUTBOUND else 'open route, vehicle stationed at supplier'}")
    print()
    print("=" * 78)
    print("SUPPLIER CLASSIFICATION")
    print("=" * 78)
    print(f"  Consolidation rule         : {CONSOL_POLICY}"
          f"  (weight < {MIN_WEIGHT_VISIT} kg"
          + (f", volume < {MIN_VOLUME_VISIT} m3" if CONSOL_POLICY != 'weight' else "") + ")")
    print(f"  Suppliers                  : {len(Q) - 1}")
    print(f"  Sent to consolidation      : {len(Sup_conso)}  -> {sorted(Sup_conso)}")
    print(f"  Inactive in the period     : {len(Sup_zero)}  -> {sorted(Sup_zero)}")
    print(f"  Routed                     : {len(Sup_visit)}  -> {sorted(Sup_visit)}")
    if Sup_visit:
        qbar = sum(Q[i] for i in Sup_visit) / len(Sup_visit)
        print(f"  Mean consignment           : {qbar:,.0f} kg")
        # Load-to-capacity rule: the stop level beyond which the vehicle
        # saturates and further relaxation of the cap yields nothing.
        print(f"  Saturation rule S* = Q/qbar = {CV_WEIGHT/qbar:.2f}  -> S* = {round(CV_WEIGHT/qbar)}")
    print()


def print_solution(routes, title="SOLUTION"):
    km = total_distance(routes)
    em_ttw = total_emissions(routes, E0_TTW, GAMMA_TTW)
    em_wtw = total_emissions(routes, E0_WTW, GAMMA_WTW)
    peso = sum(Q[i] for r in routes for i in r.seq)

    print("=" * 78)
    print(f"{title}   [objective = {OBJECTIVE}"
          + (f", alpha={ALPHA_DIST}, beta={BETA_EMIS}" if OBJECTIVE == 'hybrid' else "") + f", S = {MAX_NODOS}]")
    print("=" * 78)
    print(f"  Routes                     : {len(routes)} / {T_MAX_VEHICLES}")
    print(f"  Total distance             : {km:,.1f} km")
    print(f"  Emissions, TTW             : {em_ttw:,.1f} kg CO2e")
    print(f"  Emissions, WTW             : {em_wtw:,.1f} kg CO2e")
    print(f"  Weight collected           : {peso:,.0f} kg")
    if peso > 0:
        print(f"  Intensity                  : {1000*em_wtw/peso:,.1f} kg CO2e per tonne (WTW)")
    print(f"  Objective value            : {total_cost(routes):,.4f}")
    print()
    for k, r in enumerate(routes):
        seq = " -> ".join(map(str, r.seq + [DEPOT]))
        print(f"  Veh {k:2d}  {seq:<28s} stops={len(r.seq)}"
              f"  weight={r.load_w:8,.0f}  vol={r.load_v:6.1f}"
              f"  km={r.dist:7.1f}  CO2e={route_emissions(r.seq, E0_WTW, GAMMA_WTW):7.1f}"
              f"  use={100*r.load_w/CV_WEIGHT:5.1f}% weight / {100*r.load_v/CV_VOLUME:5.1f}% vol")
    print()


# ============================================================================
#  MAIN
# ============================================================================
def main():
    print_header()

    if not Sup_visit:
        print("No routed suppliers: the whole base is consolidated or inactive.")
        return

    init = (trivial_initial() if START_MODE == 'trivial'
            else clarke_wright_initial(cw_noise=CW_NOISE, seed=SEEDS[0]))
    two_opt_local_search(init)
    print_solution(init, f"INITIAL SOLUTION ({START_MODE})")

    best, val, mean, sd_, vals = solve(SEEDS, verbose=False)
    print_solution(best, "TABU SEARCH (best of {} seeds)".format(len(SEEDS)))
    print(f"  Dispersion across seeds    : mean={mean:,.4f}  sd={sd_:,.4f}"
          f"  CV={100*sd_/mean:.2f}%")
    print("  Objective by seed          : " + "  ".join(f"{x:,.3f}" for x in vals))
    print()

    exact, exact_val = exact_solve()
    if exact is not None:
        gap = 100.0 * (val - exact_val) / exact_val
        print("=" * 78)
        print("VALIDATION AGAINST THE CERTIFIED OPTIMUM (set partitioning)")
        print("=" * 78)
        print(f"  Certified optimum          : {exact_val:,.4f}")
        print(f"  Tabu search                : {val:,.4f}")
        print(f"  Gap                        : {gap:.4f} %")
        print()
        if gap > 1e-6:
            print_solution(exact, "CERTIFIED OPTIMUM")
    else:
        print(f"  (Certified optimum not computed: {len(Sup_visit)} routed suppliers "
              f"exceed the enumeration limit.)")


if __name__ == "__main__":
    main()

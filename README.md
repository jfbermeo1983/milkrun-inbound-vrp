# Milk-run inbound VRP: contractual parameters as the environmental lever

Code and data accompanying the article on inbound collection routing at an
automotive assembly plant, in which the contractual cap on stops per route and
the consolidation threshold are treated as environmental decision variables.

The repository reproduces every number reported in the paper: the validation
against certified optima, the main distance-versus-emissions comparison, the
effect of the stop limit, the designed experiment behind the load-to-capacity
rule, the consolidation-threshold analysis, and Figures 3 to 5.

## Problem

An open vehicle routing problem on a many-to-one collection network. Each
supplier has a truck stationed in its vicinity, so a route starts at a
supplier, visits the remaining suppliers assigned to it, and ends at the plant;
no vehicle travels out from the plant empty. Four constraints apply to every
route: weight capacity, volume capacity, a contractual cap `S` on the number of
stops, and a maximum admissible detour per intermediate stop. Consignments
below the consolidation threshold leave the milk-run and travel through the
consolidation centre.

Because the payload accumulates along the route, the emission factor of an arc
depends on the routing decision itself. The objective stays linear by writing
that factor as an affine function of the payload,

```
e(w) = E0 + GAMMA * w        [kg CO2e / km]
```

fitted by ordinary least squares to the vehicle's consumption profile. With the
24 t articulated vehicle of this network the fit gives `E0 = 0.7170` and
`GAMMA = 1.470e-5` on a tank-to-wheel basis, `E0 = 0.8866` and
`GAMMA = 1.818e-5` well-to-wheel, with R squared of 0.998 and a maximum
residual of 0.012 kg CO2e per km.

One consequence is worth stating, because it governs how these results should
be read: both coefficients come from the same consumption profile through a
single multiplicative conversion factor, so changing that factor rescales them
equally, multiplies the objective by a constant, and leaves the optimal plan
unchanged. Only absolute emission figures move. What does change the optimal
plan is the shape of the consumption profile, that is, the ratio between
unladen and fully laden consumption.

## Requirements

Python 3.9 or later. The solver and the experiments use the standard library
only. `pandas` and `matplotlib` are needed to rebuild the tables and figures:

```
pip install -r requirements.txt
```

## Layout

```
milkrun_vrp.py     Standalone case-study run: emission model, supplier
                   classification, tabu search over several seeds, and
                   validation against the certified optimum. Start here.
solver.py          Core library: Instance, Params, Model, Clarke-Wright,
                   2-opt, tabu_search, solve, exact_solve.
instances.py       Loaders for the case-study network and the Augerat (1995)
                   benchmark set.
sweep.py           Main experiment. 160 configurations x 10 seeds.
sweep_s.py         Stop limit extended to S = 6, 7, 8, closed convention.
sweep_s_open.py    The same under the open-route convention.
sweep_ratio.py     Designed experiment on the load-to-capacity ratio.
analyze.py         Builds the article tables from the CSV results.
make_figures.py    Builds Figures 3 to 5 as PDF and 600 dpi PNG.
data/vw40.py       Case-study network: plant and 39 suppliers.
data/augerat/      Seven Augerat A-set benchmark instances.
results/           CSV output of the experiments, as reported in the paper.
figures/           Figures 3 to 5.
```

## Reproducing the results

A single case-study run, which prints the emission model, the supplier split,
the tabu-search solution and the certified optimum:

```
python milkrun_vrp.py
```

The full set of experiments, in order. Approximate wall-clock times on a single
core are given; nothing here is parallelised:

```
python sweep.py           # ~39 min  -> results/results_raw.csv, results/validation.csv
python sweep_s.py         # ~12 min  -> results/results_extra_S.csv
python sweep_s_open.py    # ~15 min  -> results/results_extra_S_open.csv
python sweep_ratio.py     # ~15 min  -> results/results_ratio.csv
python analyze.py         # prints the article tables
python make_figures.py    # writes figures/figure3-5 .pdf and .png
```

The CSV files under `results/` are the ones the article was written from, so
`analyze.py` and `make_figures.py` run without re-executing the sweeps.

## Results files

| File | Rows | Content |
|---|---|---|
| `results_raw.csv` | 160 | One row per instance, stop limit, objective and route convention, with mean and dispersion across 10 seeds |
| `validation.csv` | 8 | Tabu search against the certified optimum on the instances small enough to enumerate |
| `results_extra_S.csv` | 24 | Stop limits 6 to 8, closed convention |
| `results_extra_S_open.csv` | 24 | Stop limits 6 to 8, open convention |
| `results_ratio.csv` | 180 | Designed experiment: four base instances rescaled to five target load-to-capacity ratios, stop limits 2 to 10 |

## Validation

The metaheuristic is not asserted to be good; it is measured. `exact_solve`
enumerates every feasible subset of routed suppliers, obtains the optimal
sequence of each by complete permutation, and solves the partition by dynamic
programming over bit masks. On the eight instances small enough for this to be
tractable, the tabu search reaches the certified optimum in every case, with a
gap of 0.0000 per cent. Across the full sweep the coefficient of variation over
ten seeds stays between 0.4 and 1.3 per cent, with a maximum of 3.0 per cent.

## Reproducibility

Every run is seeded explicitly and draws its randomness from a local generator,
so a result is fully determined by its seed. The sweeps use seeds 0 to 9; the
standalone script uses 1001 to 1005.

## Data

`data/vw40.py` holds the road distance matrix in kilometres and the
consignment weights and volumes of one collection cycle. The benchmark files
under `data/augerat/` are the A-set of Augerat et al. (1995) in TSPLIB format;
their demands are rescaled so that vehicle capacity equals the 24,000 kg of the
case-study vehicle, which is what makes the load-to-capacity rule comparable
across instances.

## Citation

See `CITATION.cff`. If you use this code, please cite the article.

## License

MIT. See `LICENSE`.

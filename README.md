# hawkes-nbbo-events

Code and derived data for *Event definition and the branching ratio: Hawkes estimates of
price-change clustering in the consolidated quote feed* (Hyoeun Lee, University of Illinois
Urbana-Champaign). Archived releases: Zenodo, https://doi.org/[to be assigned at the first release].
Cite the paper and this repository (`CITATION.cff`).

The quote and trade data are the NYSE Trade and Quote database (Millisecond TAQ), accessed through
Wharton Research Data Services (WRDS) under licence, and cannot be redistributed. Everything else is
here: the queries that extract the data, the code that builds the event series and fits the models,
and the fitted parameters, bootstrap replicates and simulation rows from which every table and
figure of the paper is produced. A reader without WRDS access can rebuild every table and figure
from `results/`; a reader with WRDS access can rerun the pipeline from the raw tables.

## Layout

| Path | What it holds |
|---|---|
| `hawkes_taq/` | The package: event construction from NBBO changes (`events.py`), the gated bivariate Hawkes model with K exponential kernels, its likelihood, fitting, residuals and simulation (`hawkes.py`), the simulation study (`simstudy.py`, `simulate_nbbo.py`), the WRDS pull (`fetch.py`) |
| `sql/` | The WRDS queries behind the data checks and the per-exchange NBBO rebuild (`14_build_lf_nbbo.py`), and the trade pull (`15_pull_trades.py`) |
| `scripts/` | Fitting and analysis scripts; `scripts/cluster/` holds the batch files of the runs that need a cluster |
| `results/` | The derived files, with a manifest in `results/README.md` |
| `tests/` | Unit tests of the event construction and the likelihood (`pytest`) |

## Environment

Python 3.11 with numpy, pandas, scipy and matplotlib; numba is optional and speeds the likelihood
about seventyfold; the `wrds` package is needed only for the pull. `pip install -r requirements.txt`.
The tests run in about a minute: `pytest tests`.

## Rebuilding the tables and figures (no TAQ access needed)

```
python scripts/paper_tables.py                       # all fourteen numerical tables, one block per table
python scripts/paper_tables.py main masses horizon   # a selection, by label
python scripts/make_paper_figures.py --fits-e1m results/fits/fits_e1m_delta_K4.csv \
    --fits-e0 results/fits_e0_k4_panel.csv --panel-sd results/blockboot_panel_sd.csv \
    --by-day results/by_day_fixed_timescales.csv --priceunits-fine results/a8_priceunits_fine.csv \
    --out figures                                    # Figures 2, 3 and 4 as PDF, EPS and PNG
```

`paper_tables.py` prints each table in the paper's row order and rounding so that it can be set
against `paper.tex` line by line. The map from each table and figure to its inputs:

| Table or figure | Function or script | Inputs under `results/` |
|---|---|---|
| Table 1 (example) and Table 2 (specification) | none, text only | |
| Table 3, baseline resolution, INTC May 2017 | `tab_baseline` | `baseline_bins_panel.csv`, `bootstrap/bins_201705/`, `bootstrap/parametric_201705_summary.csv` |
| Table 4, recovery in the base experiment | `tab_sim` | `simulation/batch1_replicates.csv` |
| Table 5, raw-event distortion, simulation and data | `tab_sim` | `simulation/batch1_replicates.csv`, `common_window_k4.csv`, `fits_e0_k4_panel.csv`, the main panel, `panel/gapshares_monthly.csv` |
| Table 6, event statistics, May 2017 | `tab_events` | `panel/gapshares_monthly.csv`, `panel/tick_regime.csv`, `fits/fits_K3_nodelta_panel.csv`, `fits/fits_e1m_delta_K4.csv` |
| Table 7, Lee (2023) replication | `tab_ibm` | `fits/lee2023_replication_fits.csv` |
| Table 8, kernel mass within one and ten seconds | `tab_horizon` | `fits_e0_k4_panel.csv`, the main panel, `fits/fits_K3_nodelta_panel.csv` |
| Table 9, threshold curve | `tab_threshold` | `fits/threshold_curve/`, `fits/fits_e1m_delta_K4.csv` |
| Table 10, parametric against block bootstrap | `tab_blockboot` | `bootstrap/parametric_201705/`, `blockboot_may_blocks.csv`, the May fits |
| Tables 11 and 12, main panel | `tab_main`, `tab_masses` | the main panel (`fits/fits_e1m_delta_K4.csv` with BAC from `fits/fits_e1m_h2_K3.csv`), `blockboot_panel_sd.csv` |
| Table 13, kernel by time of day | `tab_byhour` | `panel/by_hour/`, `by_hour_BAC_h2.csv` |
| Table 14, tick regime | `tab_regime` | `panel/tick_regime.csv`, the raw and the main panel |
| Table B.1, fully gated quantities | `tab_fullygated` | `fully_gated_201705.csv` |
| Table C.1, price units | `tab_priceunits` | `a8_priceunits_k4.csv` |
| Figure 1, brief states (diagram) | `docs/overview/figures/make_lock_figure.py --paper` in the development repository | none |
| Figure 2, fitted kernels | `make_paper_figures.py`, `fig_kernel` | the raw and the main panel |
| Figure 3, price path by horizon | `fig_priceunits` | `a8_priceunits_fine.csv` |
| Figure 4, n(1 s) against activity | `fig_months` | `blockboot_panel_sd.csv`, `by_day_fixed_timescales.csv` |

The numbers quoted in the text come from the same files; the results document of the development
repository records each one with its source file and section.

## Rerunning the pipeline (WRDS access needed)

1. **Pull the NBBO changes.** `hawkes_taq.fetch.fetch_nbbo_changes` saves one file per stock-day
   (`t`, `bid`, `ask`) from `taqmsec.complete_nbbo`; `README.md` shows the loop. The sample is the eight
   stocks INTC, MU, MDLZ, BAC, WFC, T, TGT and ATVI from October 31, 2016 to December 29, 2017, 10:00 to
   15:30, early-close days excluded (`hawkes_taq.pipeline.HALF_DAYS`). `sql/15_pull_trades.py` pulls the
   trades used in the trade match, and `sql/14_build_lf_nbbo.py` rebuilds the NBBO from the per-exchange
   quotes in SIP time and in exchange time.
2. **Fit.** `scripts/fit_panel.py` builds E0, E1 and E1m from the NBBO files and fits any specification;
   the batch files in `scripts/cluster/` give the exact arguments of every run in the paper (model order,
   threshold, refractory period, baseline bins, start grid). One stock-month of the main specification
   takes two to ten minutes on one core; the raw K = 4 fits of the busiest months take hours. The whole
   panel was fitted on a cluster with these batch files.
   The check scripts under `scripts/checks/` that read the NBBO and trade files take their paths on the
   command line or, for the revision and misfit checks, from the environment variable `HAWKES_DATA`,
   the directory that holds the pulled files in the layout the docstrings name (`may/NBBO/<SYM>/`,
   `nbbo/<quarter>/<SYM>/`, `trades_in/trades/<SYM>/`, `delta/K4.csv` for the main fits, and so on).
3. **Bootstraps, simulation and checks.** `scripts/bootstrap_ks.py` (parametric), `scripts/block_bootstrap.py`
   (day blocks), `scripts/simulate_study.py` with `scripts/calibrate_brief_states.py` (the simulation study),
   `scripts/fit_by_hour.py`, `scripts/fit_by_day.py`, `scripts/fit_without_outliers.py`, and the checks under
   `scripts/checks/`, each with its usage line in its docstring.

A smoke test of the full chain on one stock-month, from the NBBO files to a fit and a table row:

```
python scripts/fit_panel.py --nbbo DATA/nbbo_2017Q2 --syms INTC --months 201705 \
    --definitions E1m --K 4 --starts full --delta 0.001 --out DATA/smoke.csv
```

The row in `DATA/smoke.csv` should match the INTC, 201705 row of `results/fits/fits_e1m_delta_K4.csv`
in every column but `minutes`.

## Licence

Code: MIT (`LICENSE`). Derived data in `results/`: Creative Commons Attribution 4.0 (CC BY 4.0).

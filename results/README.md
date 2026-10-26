# results/: the derived files behind every table and figure

Licence: Creative Commons Attribution 4.0 (CC BY 4.0); the code is under MIT (`LICENSE` at the root).

No TAQ data are in this directory. Every file is a fitted parameter set, a bootstrap replicate, a
simulation row or a summary statistic. `scripts/paper_tables.py` rebuilds the fourteen numerical
tables of the paper from these files and `scripts/make_paper_figures.py` the three data figures; see
`REPLICATION.md` at the repository root for the map from each table and figure to its inputs.

Conventions. A fit file has one row per fit with `memory_s` (the K memories $1/\beta_k$ in seconds,
fastest first), `excitation_per_scale` ($a_k + c_k$), `effective_excitation` ($(a_k + c_k)e^{-\beta_k\delta}$),
`self_per_scale` ($a_k$), `cross_per_scale` ($c_k$), `branching_ratio` (the ungated total, eq. (br)),
`delta` (the refractory period in seconds, 0 for raw events), `loglik`, `aic`, `bic`, `ks_up`, `ks_down`
and `minutes` (wall time). Vectors are stored as "a; b; c". Months are `YYYYMM`; `sym` is the ticker.
Event definitions: `E0` raw, `E1` locks merged, `E1m` clean.

## Fits (`fits/` and the top level)

| File | Content | Produced by |
|---|---|---|
| `fits/fits_e1m_delta_K4.csv` | Main clean panel: E1m, K = 4, h = delta = 1 ms, 30-minute baselines, full start grid, 112 stock-months | `scripts/fit_panel.py` (`scripts/cluster/fit_e1m_delta.sbatch`) |
| `fits/fits_e1m_h2_K3.csv`, `fits_e1m_h2_K4.csv` | The same at h = delta = 2 ms (BAC's main specification is the K = 3 file) | `fit_e1m_h2.sbatch` |
| `fits/fits_e1m_delta_K3.csv` | Clean panel at K = 3, 1 ms | `fit_e1m_delta.sbatch` |
| `fits/fits_e1m_nodelta_K3.csv`, `_K4.csv` | Clean panel without the refractory period (Section 6.2.2's "K = 3 and none") | `fit_e1m_full.sbatch` |
| `fits_e0_k4_panel.csv` | Raw panel: E0, K = 4, default grid plus two ladder starts, 112 stock-months | `fit_e0_k4_panel.sbatch` |
| `fits_e0_K3_panel.csv` | Raw panel at K = 3 (first round) | `fit_e0_k4.sbatch` |
| `fits/fits_K3_nodelta_panel.csv` | First-round panel, K = 3, no refractory period, E0, E1 and E1m (the E1 rows are Table 8's "locks merged only") | first-round `fit_panel.py` |
| `fits_e0_k4_fullgrid_201705.csv` | Raw K = 4, May 2017, full grid plus ladder (the check of Section 6.2.2) | `fit_e0_k4_fullgrid.sbatch` |
| `fits/threshold_curve/` | E1m with h = delta = 0.25, 0.5, 2, 5 ms, K = 4, eight stocks, May and November 2017; `curve.csv` joins them with the 1 ms fits (Table 9) | `fit_threshold_curve.sbatch` |
| `fits_e1m_K5_201705.csv`, `fits_e1m_K5_BAC_h2_201705.csv`, `k5_check_201705.csv` | K = 5 on the eight May stocks and the comparison with K = 4 (Sections 3.2 and 6.5) | `scripts/checks/misfit/k5_check.py` |
| `fits/fits_exchange_time_201705.csv`, `_201711.csv` | Fits on the NBBO rebuilt in SIP time and in exchange time (Section 6.3.2) | `sql/14_build_lf_nbbo.py`, then `fit_panel.py` |
| `fits/lee2023_replication_fits.csv`, `_stats.csv` | IBM and AAPL, January 2017, daily fits in Lee (2023)'s specification and ours (Table 7) | `scripts/checks/lee2023_replication.py` |
| `fits/fits_lee2024_grid_201705.csv` | The 0.1-second grid filter of Lee (2024) fitted on the four simulation stocks (Section 6.3.3) | `scripts/checks/grid_filter_lee2024.py` |
| `fits/refit_without_half_days_K4.csv`, `fits/refractory_check_201705.csv` | Early-close days removed; the refractory-period check of results section 10 | `fit_panel.py` |
| `fully_gated_201705.csv` | Appendix B's table | `scripts/checks/fully_gated.py` |

## Bootstraps (`bootstrap/` and the top level)

| File | Content | Produced by |
|---|---|---|
| `bootstrap/parametric_201705/<SYM>_201705/` | Parametric bootstrap of the main fit, May 2017, 99 replicates: `fit.csv` (the fit, one row), `replicates.csv` (one row per refit), `summary.csv` (estimate, bootstrap mean, standard error, percentiles per parameter) | `scripts/bootstrap_ks.py` (`bootstrap_se.sbatch`) |
| `bootstrap/parametric_201705_summary.csv` | The May bootstrap condensed: per component memory ratio, excitation bias and standard error, effective branching ratio bias and standard error | `bootstrap_ks.py` summary |
| `bootstrap/parametric_201711/`, `boot_201711_summary.csv` | The same for November 2017 (Sections 6.3.4 and 6.5) | `bootstrap_se_nov.sbatch`; `scripts/checks/revision/cluster5_summaries.py` |
| `bootstrap/bins_201705/INTC_201705_B{5,10,60,110,330}/` | INTC May 2017 refitted with 5-, 10-, 60-, 110- and 330-minute baselines, each with 30 bootstrap replicates (Table 3's bias column) | `bootstrap_bins.sbatch` |
| `bootstrap/block_panel/<SYM>_<month>.csv` | Day-block bootstrap, five-day blocks, 100 replicates, clean fit, every stock-month; `blockboot_panel_sd.csv` is its summary (Table 12's standard errors) | `scripts/block_bootstrap.py` (`block_bootstrap_panel.sbatch`); `scripts/checks/revision/a7_rawk4_panel.py` |
| `bootstrap/block_may_b{1,5,10}/` | Paired raw and clean day-block bootstrap, May 2017, blocks of 1, 5 and 10 days; `blockboot_may_blocks.csv` is its summary (Table 10) | `block_bootstrap_blocks.sbatch`; `a7_rawk4_panel.py` |

## Simulation study (`simulation/` and the top level)

| File | Content | Produced by |
|---|---|---|
| `simulation/batch1_replicates.csv` | One row per (replicate, event definition, K, delta, pooling) of the first batch: experiments `base`, `long`, `share0`, `share50`, `subms`, four stocks, 50 replicate-months each; true and fitted memories, excitations, n(1 s), n(10 s), totals, baseline events, KS statistics (Tables 4 and 5, Section 5.3) | `scripts/simulate_study.py` (`sim_truth.sbatch`, `sim_run.sbatch`), rows collected by `scripts/summarize_simulation.py` |
| `simulation/batch2_replicates.csv` | The second batch: `actscale`, `powerlaw`, `subms_clean` | `sim_run2.sbatch` |
| `sim_share0_truth.csv` | The share0 experiment refitted on the true clean events, for the bias decomposition of Section 5.2 | `sim_share0_truth.sbatch` |
| `simulation/truth/truth_<SYM>.npz` | The kernels and baselines that generate the simulated moves (the fitted May 2017 clean models) | `simulate_study.py truth` |
| `simulation/calibration/calibration_summary.csv` | The brief-state calibration per stock-month: shares of direct, through-lock and through-widening moves, flickers per move, chain durations | `scripts/calibrate_brief_states.py` |

The per-move brief-state chain files that the simulation samples durations from (`chains_<SYM>_<month>.csv.gz`) are not in the package; they are rebuilt from the NBBO files by `calibrate_brief_states.py`.

## Panel checks (`panel/` and the top level)

| File | Content | Produced by |
|---|---|---|
| `panel/by_hour/by_hour_<SYM>.csv`, `by_hour_BAC_h2.csv` | Kernel fitted by window of the day for every stock-month (Table 13); BAC at its main specification | `scripts/fit_by_hour.py` (`fit_by_hour.sbatch`, `fit_by_hour_bac2ms.sbatch`) |
| `score_windows_201705.csv` | The by-hour kernels scored on their own windows (Section 6.5) | `scripts/checks/misfit/score_windows.py` |
| `by_day_fixed_timescales.csv`, `extreme_days_calendar.csv` | Day-level fits with the month's timescales held fixed, the calendar of extreme days and the count rule (Section 6.4.4) | `scripts/fit_by_day.py`; `scripts/checks/revision/a2_day_level.py` |
| `panel/outliers/outliers_<SYM>.csv` | Refits without the count-flagged days | `scripts/fit_without_outliers.py` (`outliers.sbatch`) |
| `panel/baseline_bins/<SYM>_<month>_B<bins>.csv`, `baseline_bins_panel.csv` | Eight stocks, May and November 2017, baselines of 5 to 330 minutes (Section 4.3) | `fit_baseline_bins.sbatch`; `cluster5_summaries.py` |
| `panel/brief_states/<SYM>/calibration_summary.csv`, `brief_states_panel.csv` | Brief-state shares and durations for all 112 stock-months | `calibrate_brief_states.py` (`brief_states_panel.sbatch`) |
| `panel/gapshares_monthly.csv` | Share of gaps under 1 ms and median gap per stock-month for E0, E1 and E1m, and the share of locked or crossed states (Table 6, the counting relation) | `scripts/checks/gap_shares.py` |
| `panel/tick_regime.csv` | Time at a one-tick spread, locked and two-tick states, NBBO changes a day and price per stock-month (Table 14; the regime rule is applied in `paper_tables.py`) | `scripts/checks/tick_regime.py` |
| `common_window_k4.csv`, `panel/horizon_k4.csv`, `panel/panel_k4_joined.csv` | Raw and clean kernel mass on common windows and the counting relation, 112 stock-months (Table 8, Section 6.2.4) | `scripts/checks/common_window.py`; `a7_rawk4_panel.py` |
| `a8_priceunits_k4.csv`, `a8_priceunits_fine.csv` | Price units: observed and model-implied mid change after an up move, three horizons (Appendix C) and thirty (Figure 3) | `scripts/checks/revision/a8_priceunits_k4.py`, `a8_priceunits_fine.py` |
| `panel/a9a10_may.csv` | Exact immigrant accounting and the follower discriminant, May 2017 (Section 6.2.1) | `scripts/checks/revision/a9a10_followers_accounting.py` |
| `panel/a1_activity_measures.csv`, `a3_locktail_by_month.csv`, `a3_nov_windows.csv`, `a3_thresholds.csv`, `a3_tod_may.csv` | Activity measures, the lock tail by month, November windows in both clocks, the 2 ms refit comparison, the tail by time of day (Section 6.4.5) | `scripts/checks/revision/a3_*.py` |
| `panel/event_qc_201705.csv`, `event_stats_201705_201711.csv`, `event_stats_exchange_time_201705.csv` | Event counts and state statistics per stock-day and per version of the NBBO (SIP time, exchange time) | `scripts/build_events.py`; `sql/14_build_lf_nbbo.py` |
| `panel/match_clock_events.csv`, `match_clock_events_201705.csv` | Event-level matching between the SIP-time and exchange-time NBBO (Section 6.3.2) | `scripts/checks/match_clock_events.py` |
| `lock_pairs_201705.csv` | Lock durations by venue pair in both clocks (Section 2.1) | `scripts/checks/lock_pairs.py` |
| `trades_match_strict_201705.csv` | The trade match for the brief positive-spread states, loose and strict (Section 2.1) | `scripts/checks/trades_match.py`, `trades_match_strict.py` |
| `panel/misfit_logs/` | The residual diagnostics of Section 6.5 as the scripts printed them | `scripts/checks/misfit/diag.py`, `burnin.py`, `finebin.py`, `altern.py` |

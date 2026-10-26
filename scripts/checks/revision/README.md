# Revision checks (mock referee report of Oct 13, 2026)

Scripts behind the revision items in `docs/review_mock_20261013_triage.md`; results in
`docs/results_panel_201611_201712.md` §27 onward. They read the session's saved fits and
state data from fixed paths (the `S` constant at the top of each file); point `S` at a folder
holding the panel fit files (`delta/K4.csv`, `delta/K3.csv`, `delta/h2_K4.csv`, `delta/h2_K3.csv`),
the calibration chain tables (`sim/cal/chains_SYM_YYYYMM.csv.gz`), the May NBBO changes
(`may/NBBO/SYM/YYYYMMDD.csv.gz`) and the November reconstruction (`lf_nov_in/lfnbbo/{sip,ex}`).

- `../common_window.py` (A5): raw and clean kernel mass on common windows; horizon-wise identity.
- `a3_thresholds_locktail.py` (A3 parts A, B; A4): activity gradients at h = 1 and 2 ms, the
  lock-duration tail by month from the chain tables, and the month-trend regressions.
- `a3_tod_tail.py` (A3 part C): lock-duration tail by window of the day and by day, May 2017.
- `a3_nov_windows.py` (A3 part D): kernel by window of the day in November 2017, SIP time
- `a8_priceunits.py` (A8): price units with the model side computed on simulated paths, day-clustered standard errors of the data means, and the isolated-cascade value for comparison.
- `a9a10_followers_accounting.py` (A10, A9): exact immigrant accounting from the states and the follower-count and size discriminant, May 2017.
- `a11_lee_identity.py` (A11): the identity under the Lee (2023) specification for IBM and AAPL.
  against exchange time.

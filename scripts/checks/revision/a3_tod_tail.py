"""A3 part 3: brief-state duration tail by window of the day and by day, May 2017, from the states."""
import os
import sys
from pathlib import Path, glob, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hawkes_taq import events as ev
from hawkes_taq.hawkes import SESSION_END, SESSION_START
S = os.environ.get("HAWKES_DATA", "DATA")   # root of the NBBO, trade and intermediate files (see REPLICATION.md)
ORDER = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]
W = [("10-11", 0, 3600), ("11-14:30", 3600, SESSION_END - SESSION_START - 3600), ("14:30-15:30", SESSION_END - SESSION_START - 3600, 1e9)]
rows = []
for sym in ORDER:
    for f in sorted(glob.glob(f"{S}/may/NBBO/{sym}/*.csv.gz")):
        st = ev.build_states(pd.read_csv(f), SESSION_END)
        t = st.t.to_numpy() - SESSION_START; sp = st.spread.to_numpy(); dur = st.dur.to_numpy()
        nonpos = sp <= 0
        for name, a, b in W:
            m = (t >= a) & (t < b); lk = m & nonpos
            rows.append(dict(sym=sym, day=f.split("/")[-1][:8], window=name, states=int(m.sum()), locks=int(lk.sum()),
                             lock_med_ms=1e3 * np.median(dur[lk]) if lk.sum() else np.nan,
                             lock_gt1=float((dur[lk] > 1e-3).mean()) if lk.sum() else np.nan,
                             lock_gt2=float((dur[lk] > 2e-3).mean()) if lk.sum() else np.nan,
                             gaps_lt1=float((dur[m] < 1e-3).mean())))
d = pd.DataFrame(rows); d.to_csv(f"{S}/a5/a3_tod_may.csv", index=False)
d["hours"] = d.window.map({"10-11": 1, "11-14:30": 3.5, "14:30-15:30": 1}); d["states_per_hour"] = d.states / d.hours
g = d.groupby(["sym", "window"]).agg(states_per_hour=("states_per_hour", "median"), lock_med_ms=("lock_med_ms", "median"),
                                     lock_gt1=("lock_gt1", "median"), lock_gt2=("lock_gt2", "median"), gaps_lt1=("gaps_lt1", "median"))
print(g.round(3).to_string())
# across days within stock: does the lock tail rise with the day's activity?
dd = d.groupby(["sym", "day"]).agg(states=("states", "sum"), locks=("locks", "sum"), lock_gt1=("lock_gt1", "mean"), lock_med_ms=("lock_med_ms", "mean")).reset_index()
dd["la"] = np.log(dd.states)
for c in ["la", "lock_gt1", "lock_med_ms"]: dd[c + "_d"] = dd[c] - dd.groupby("sym")[c].transform("mean")
print(f"\nacross days within stock (May): corr(lock share > 1 ms, log states/day) {np.corrcoef(dd.lock_gt1_d, dd.la_d)[0,1]:+.2f}; corr(lock median, log states/day) {np.corrcoef(dd.lock_med_ms_d, dd.la_d)[0,1]:+.2f}")
print(dd.groupby("sym").apply(lambda g: pd.Series({"corr_gt1": np.corrcoef(g.lock_gt1, g.la)[0,1], "corr_med": np.corrcoef(g.lock_med_ms, g.la)[0,1]})).reindex(ORDER).round(2).to_string())

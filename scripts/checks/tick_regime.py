import sys
from pathlib import Path, glob, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hawkes_taq import events as ev, hawkes as hk
from hawkes_taq.pipeline import HALF_DAYS
S = sys.argv[1] if len(sys.argv) > 1 else "."  # folder with nbbo/<quarter>/<SYM>/*.csv.gz
rows = []
for sym in ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]:
    files = sorted(f for q in ["2016Q4", "2017Q1", "2017Q2", "2017Q3", "2017Q4"] for f in glob.glob(f"{S}/nbbo/{q}/{sym}/*.csv.gz"))
    by_month = {}
    for f in files:
        d = f.split("/")[-1][:8]
        if d in HALF_DAYS or d < "20161101": continue
        by_month.setdefault(d[:6], []).append(f)
    for month, fs in sorted(by_month.items()):
        t1 = t0 = t2 = tlock = 0.0; n = 0; nlock = 0; price = []; days = 0
        for f in fs:
            s = ev.build_states(pd.read_csv(f), hk.SESSION_END); dur = s.dur.to_numpy(); sp = s.spread.to_numpy()
            t1 += dur[sp == 1].sum(); tlock += dur[sp <= 0].sum(); t2 += dur[sp >= 2].sum()
            n += len(s); nlock += int((sp <= 0).sum()); price.append(float((s.bid + s.ask).mean() / 2 * ev.TICK)); days += 1
        T = t1 + tlock + t2
        rows.append(dict(sym=sym, month=month, days=days, price=round(np.mean(price), 1), changes_per_day=round(n / days),
                         pct_time_1tick=round(100 * t1 / T, 2), pct_time_locked=round(100 * tlock / T, 2), pct_time_2plus=round(100 * t2 / T, 2),
                         pct_changes_lock=round(100 * nlock / n, 1)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{S}/tick_regime.csv", index=False)

"""Share of sub-millisecond gaps and median gap per stock-month for E0, E1 and E1m, and the share of
locked or crossed states (results/panel/gapshares_monthly.csv; Table 6 and Section 6.2 of the paper).

    python scripts/checks/gap_shares.py --nbbo DATA/nbbo_2016Q4 DATA/nbbo_2017Q1 ... --out results/panel/gapshares_monthly.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hawkes_taq import events as ev, hawkes as hk  # noqa: E402

SYMS = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbbo", nargs="+", required=True, help="directories with <SYM>/<date>.csv.gz NBBO change files")
    p.add_argument("--syms", nargs="+", default=SYMS)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for sym in a.syms:
        files = sorted([f for r in a.nbbo for f in (Path(r) / sym).glob("*.csv.gz")], key=lambda f: f.name)
        by_month = {}
        for f in files:
            by_month.setdefault(f.name[:6], []).append(f)
        for month, fs in by_month.items():
            if month == "201610":
                continue
            acc = {"E0": [], "E1": [], "E1m": []}
            nonpos = states = 0
            for f in fs:
                s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
                nonpos += int((s.spread <= 0).sum()); states += len(s)
                for k, e in (("E0", ev.events_e0(s)), ("E1", ev.events_e1(s)), ("E1m", ev.events_e1(s, min_state=0.001))):
                    acc[k].append(np.diff(e["t"].to_numpy()))
            r = {"sym": sym, "month": month, "n_days": len(fs), "pct_states_nonpos": 100 * nonpos / states}
            for k, gs in acc.items():
                g = np.concatenate(gs)
                r[f"gap_lt1ms_{k}"] = float((g < 1e-3).mean())
                r[f"median_gap_{k}"] = float(np.median(g))
            rows.append(r)
            print(sym, month, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

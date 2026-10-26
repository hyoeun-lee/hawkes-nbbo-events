"""Kernel fitted by time of day, for the whole panel (results §23.B extended).

For each stock-month, E1m (1 ms merge, 1 ms dead time, K = 4, 30-minute baselines) is fitted
separately on three windows of every day, each window treated as its own session with a 60 s
burn-in: 10:00-11:00, 11:00-14:30 and 14:30-15:30. Fits start from the main fit's decay rates
(--fits, the panel file fits_e1m_delta_K4.csv), as in the May 2017 check.

    python scripts/fit_by_hour.py --nbbo DATA/nbbo_2016Q4 ... --fits DATA/fits_e1m_delta_K4.csv \\
        --syms INTC --out DATA/by_hour_INTC.csv

One row per (stock, month, window); resumable.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hawkes_taq import hawkes as hk  # noqa: E402
import fit_panel as fp  # noqa: E402

MONTHS = ["201611", "201612"] + [f"2017{m:02d}" for m in range(1, 13)]
BURN = 60.0


def windows(T):
    return [("10:00-11:00", 0.0, 3600.0), ("11:00-14:30", 3600.0, T - 3600.0), ("14:30-15:30", T - 3600.0, T)]


def piece(days, a, b):
    out = []
    for d in days:
        m = (d.t >= a) & (d.t < b)
        if m.sum() > 10:
            out.append(hk.Day(t=d.t[m] - a, types=d.types[m], T=b - a, label=d.label))
    return out


def mass(f, lo, H):
    return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--fits", required=True, help="main panel fits (start values)")
    p.add_argument("--syms", nargs="+", required=True)
    p.add_argument("--months", nargs="+", default=MONTHS)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--min-state", type=float, default=0.001,
                   help="persistence threshold h for E1m (BAC: 0.002 with --delta 0.002, --K 3)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fits = pd.read_csv(a.fits, dtype={"month": str})
    fits = fits[(fits.definition == "E1m") & (fits.K == a.K)]
    done = set()
    if Path(a.out).exists():
        d = pd.read_csv(a.out, dtype={"month": str})
        done = set(zip(d.sym, d.month, d.window))
    rows = []
    for sym in a.syms:
        for month in a.months:
            row0 = fits[(fits.sym == sym) & (fits.month == month)]
            if row0.empty:
                print(sym, month, "no main fit", flush=True)
                continue
            beta0 = 1.0 / np.array([float(x) for x in str(row0.iloc[0].memory_s).split(";")])
            days = fp.build_days(fp.month_files(a.nbbo, sym, month), "E1m", a.min_state)
            if not days:
                continue
            T = days[0].T
            for name, lo, hi in windows(T):
                if (sym, month, name) in done:
                    continue
                t0 = time.time()
                ds = piece(days, lo, hi)
                f = hk.fit(ds, K=a.K, kernel="symmetric", bin_minutes=30, starts=[beta0],
                           baseline="shared", delta=a.delta, burn=BURN)
                row = {"sym": sym, "month": month, "window": name, "n_days": len(ds), "n_events": f.n_events,
                       "events_per_hour": round(f.n_events / len(ds) / ((hi - lo) / 3600.0), 1),
                       "memory_s": "; ".join(f"{m:.4g}" for m in f.timescales()),
                       "excitation_per_scale": "; ".join(f"{x:.4f}" for x in f.alpha.sum(axis=0)),
                       "branching_ratio": round(float(f.branching_ratio), 4),
                       "n1": round(mass(f, a.delta, 1.0), 4), "n10": round(mass(f, a.delta, 10.0), 4),
                       "loglik": round(float(f.loglik), 2), "minutes": round((time.time() - t0) / 60, 2)}
                pd.DataFrame([row]).to_csv(a.out, mode="a", header=not Path(a.out).exists(), index=False)
                print(sym, month, name, row["n1"], row["n10"], row["branching_ratio"], f"{row['minutes']} min", flush=True)
                rows.append(row)


if __name__ == "__main__":
    main()

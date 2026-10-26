"""Outlier days: flag days with an extreme number of NBBO changes and refit the stock-month
without them (results §24).

A day is flagged when its number of NBBO price changes (E0 events) exceeds --factor times the
stock-month median. Flagged stock-months are refitted (E1m, K = 4, 1 ms dead time, 30-minute
baselines, started from the main fit) without those days; the main fit's values are copied into
the same row for comparison.

    python scripts/fit_without_outliers.py --nbbo DATA/nbbo_2016Q4 ... --fits DATA/fits_e1m_delta_K4.csv \\
        --syms INTC --out DATA/outliers_INTC.csv
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


def mass_from_row(r, lo, H):
    m = np.array([float(x) for x in str(r.memory_s).split(";")])
    e = np.array([float(x) for x in str(r.excitation_per_scale).split(";")])
    return float(np.sum(e * (np.exp(-lo / m) - np.exp(-H / m))))


def mass(f, lo, H):
    return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--fits", required=True)
    p.add_argument("--syms", nargs="+", required=True)
    p.add_argument("--months", nargs="+", default=MONTHS)
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fits = pd.read_csv(a.fits, dtype={"month": str})
    fits = fits[(fits.definition == "E1m") & (fits.K == a.K)]
    done = set()
    if Path(a.out).exists():
        d = pd.read_csv(a.out, dtype={"month": str})
        done = set(zip(d.sym, d.month))
    for sym in a.syms:
        for month in a.months:
            if (sym, month) in done:
                continue
            files = fp.month_files(a.nbbo, sym, month)
            if not files:
                continue
            raw = fp.build_days(files, "E0", 0.001)
            counts = pd.Series({d.label: len(d.t) for d in raw})
            med = counts.median()
            flagged = sorted(counts[counts > a.factor * med].index)
            row = {"sym": sym, "month": month, "n_days": len(raw), "median_e0_per_day": int(med),
                   "max_e0_per_day": int(counts.max()), "flagged": " ".join(flagged), "n_flagged": len(flagged)}
            r0 = fits[(fits.sym == sym) & (fits.month == month)]
            if not r0.empty:
                r0 = r0.iloc[0]
                row.update(main_memory_s=r0.memory_s, main_br=r0.branching_ratio,
                           main_n1=round(mass_from_row(r0, a.delta, 1.0), 4), main_n10=round(mass_from_row(r0, a.delta, 10.0), 4))
            if flagged and not r0.empty:
                t0 = time.time()
                beta0 = 1.0 / np.array([float(x) for x in str(r0.memory_s).split(";")])
                clean = [d for d in fp.build_days(files, "E1m", 0.001) if d.label not in flagged]
                f = hk.fit(clean, K=a.K, kernel="symmetric", bin_minutes=30, starts=[beta0], baseline="shared", delta=a.delta)
                row.update(refit_n_days=len(clean), refit_n_events=f.n_events,
                           refit_memory_s="; ".join(f"{m:.4g}" for m in f.timescales()),
                           refit_br=round(float(f.branching_ratio), 4),
                           refit_n1=round(mass(f, a.delta, 1.0), 4), refit_n10=round(mass(f, a.delta, 10.0), 4),
                           minutes=round((time.time() - t0) / 60, 2))
            pd.DataFrame([row]).to_csv(a.out, mode="a", header=not Path(a.out).exists(), index=False)
            print(sym, month, "flagged:", flagged or "none", {k: v for k, v in row.items() if k.startswith(("main_n", "refit_n1", "refit_n10", "refit_br"))}, flush=True)


if __name__ == "__main__":
    main()

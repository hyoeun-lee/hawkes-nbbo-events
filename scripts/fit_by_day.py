"""Day-level excitation with the month's timescales held fixed (revision item A2).

For each stock-month the main fit's decay rates are kept and every day is fitted on its own:
the day's baseline bins and excitation weights are free, the timescales are not (hawkes.fit_fixed,
one concave problem per day). The output has one row per day: the day's kernel masses n(1 s),
n(10 s) and total on (delta, H], branching ratio, mean baseline, event counts (clean and raw),
the calendar flags of --calendar (date, sym or ALL, class) and the count flag of the earlier
outlier rule (raw count above --factor times the stock-month median). Days chosen from the
calendar are chosen without looking at the data, which answers the objection that count-flagged
days are selected on the outcome.

Specification: E1m with h = delta = 1 ms and K = 4 (--fits), or, for the stocks in --h2-syms,
h = delta = 2 ms and K = 3 (--fits-h2), the main specification of the paper.

    python scripts/fit_by_day.py --nbbo DATA/nbbo_2016Q4 ... --fits DATA/fits_e1m_delta_K4.csv \\
        --fits-h2 DATA/fits_e1m_h2_K3.csv --h2-syms BAC \\
        --calendar results/extreme_days_calendar.csv --out by_day.csv
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hawkes_taq import hawkes as hk  # noqa: E402
from hawkes_taq.pipeline import SAMPLE_STOCKS  # noqa: E402
import fit_panel as fp  # noqa: E402

MONTHS = ["201611", "201612"] + [f"2017{m:02d}" for m in range(1, 13)]


def mass(f, lo, H):
    """Kernel mass on (lo, H] per event, both directions summed (symmetric kernel)."""
    return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))


def mass_from_row(r, lo, H):
    m = np.array([float(x) for x in str(r.memory_s).split(";")])
    e = np.array([float(x) for x in str(r.excitation_per_scale).split(";")])
    return float(np.sum(e * (np.exp(-lo / m) - np.exp(-H / m))))


def job(roots, sym, month, fits_row, h, K, factor, calendar):
    t0 = time.time()
    files = fp.month_files(roots, sym, month)
    if not files:
        return []
    beta = 1.0 / np.array([float(x) for x in str(fits_row.memory_s).split(";")])
    raw = {d.label: len(d.t) for d in fp.build_days(files, "E0", h)}
    med = float(np.median(list(raw.values())))
    clean = fp.build_days(files, "E1m", h)
    rows = []
    for d in clean:
        f = hk.fit_fixed([d], beta, bin_minutes=30, delta=h)
        cls = calendar.get((d.label, sym), []) + calendar.get((d.label, "ALL"), [])
        rows.append({"sym": sym, "month": month, "date": d.label, "h": h, "K": K,
                     "n_events": len(d.t), "n_e0": raw.get(d.label, np.nan),
                     "e0_month_median": med, "count_flag": raw.get(d.label, 0) > factor * med,
                     "calendar": ";".join(sorted(set(cls))),
                     "n1": round(mass(f, h, 1.0), 4), "n10": round(mass(f, h, 10.0), 4),
                     "total": round(float(f.alpha.sum()), 4),
                     "branching_ratio": round(float(f.branching_ratio), 4),
                     "eff_excitation": round(float(f.effective_excitation().sum()), 4),
                     "mu_mean": round(float(f.mu.mean()), 5), "mu_first_bin": round(float(f.mu[0, 0]), 5),
                     "excitation_per_scale": "; ".join(f"{a:.4f}" for a in f.alpha.sum(axis=0)),
                     "loglik": round(float(f.loglik), 2),
                     "month_n1": round(mass_from_row(fits_row, h, 1.0), 4),
                     "month_n10": round(mass_from_row(fits_row, h, 10.0), 4),
                     "month_memory_s": fits_row.memory_s})
    print(f"{sym} {month}: {len(rows)} days in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def read_calendar(path):
    cal = {}
    if path is None:
        return cal
    c = pd.read_csv(path, dtype=str)
    for r in c.itertuples():
        cal.setdefault((r.date, r.sym), []).append(r.cls)
    return cal


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--fits", required=True, help="main fits, E1m, h = 1 ms, K = 4")
    p.add_argument("--fits-h2", default=None, help="fits at h = 2 ms, K = 3, for --h2-syms")
    p.add_argument("--h2-syms", nargs="*", default=["BAC"])
    p.add_argument("--syms", nargs="+", default=SAMPLE_STOCKS)
    p.add_argument("--months", nargs="+", default=MONTHS)
    p.add_argument("--calendar", default=None, help="CSV with columns date (YYYYMMDD), sym (ticker or ALL), cls")
    p.add_argument("--factor", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    fits = {(1e-3, 4): pd.read_csv(a.fits, dtype={"month": str})}
    if a.fits_h2:
        fits[(2e-3, 3)] = pd.read_csv(a.fits_h2, dtype={"month": str})
    calendar = read_calendar(a.calendar)
    out = Path(a.out)
    done = set()
    if out.exists():
        prev = pd.read_csv(out, dtype={"month": str})
        done = set(zip(prev.sym, prev.month))
    jobs = []
    for sym in a.syms:
        h, K = (2e-3, 3) if sym in a.h2_syms else (1e-3, 4)
        F = fits[(h, K)]
        for month in a.months:
            if (sym, month) in done:
                continue
            r = F[(F.sym == sym) & (F.month == month) & (F.definition == "E1m") & (F.K == K)]
            if r.empty:
                print("no main fit for", sym, month, flush=True)
                continue
            jobs.append((sym, month, r.iloc[0], h, K))
    print(f"{len(jobs)} stock-months", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {pool.submit(job, a.nbbo, s, m, r, h, K, a.factor, calendar): (s, m) for s, m, r, h, K in jobs}
        for fut in as_completed(futs):
            try:
                rows = fut.result()
            except Exception as e:
                print("FAILED", futs[fut], repr(e)[:200], flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()

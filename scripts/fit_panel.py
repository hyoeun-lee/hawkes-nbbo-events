"""Fit Hawkes models for many stock-months, building events from saved NBBO changes.

Input: folders of NBBO changes as saved on WRDS (<root>/<SYM>/<YYYYMMDD>.csv.gz, columns
t, bid, ask). Events are built here, so event definitions can change without a WRDS run.

Example:
    python scripts/fit_panel.py --nbbo nbbo/2016Q4 nbbo/2017Q1 --months 201705 \\
        --definitions E0 E1 E1m --K 3 4 --workers 4 --out fits.csv

Definitions: E0 raw mid changes; E1 locks/crosses < 5 ms merged; E1m additionally every
state < --min-state seconds (default 1 ms) merged. One row per (stock, month, definition, K)
is appended to the output CSV as each fit finishes, so a run can be resumed. Use a separate
--out file for each --kernel/--baseline/--starts/--delta specification (resuming keys on stock,
month, definition and K only).
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

from hawkes_taq import events as ev  # noqa: E402
from hawkes_taq import hawkes as hk  # noqa: E402
from hawkes_taq.pipeline import HALF_DAYS, SAMPLE_STOCKS  # noqa: E402

END = hk.SESSION_END


def month_files(roots, sym, month, exclude=HALF_DAYS):
    """Saved NBBO files of one stock-month, without early-close days (pipeline.HALF_DAYS)."""
    files = []
    for r in roots:
        files += [f for f in sorted(Path(r, sym).glob(f"{month}*.csv.gz")) if f.name[:8] not in exclude]
    return sorted(files, key=lambda f: f.name)


def build_days(files, definition, min_state):
    days = []
    for f in files:
        s = ev.build_states(pd.read_csv(f), END)
        if definition == "E0":
            e = ev.events_e0(s)
        elif definition == "E1":
            e = ev.events_e1(s)
        elif definition == "E1m":
            e = ev.events_e1(s, min_state=min_state)
        else:
            raise ValueError(definition)
        e["t_fit"] = ev.spread_ties(e["t"])
        d = hk.day_from_events(e, label=f.name[:8])
        if len(d.t):
            days.append(d)
    return days


def extra_starts(path, sym, month, definition, K):
    """Two extra starting points built from the saved (K - 1)-component fit of the same series:
    its decay rates plus one faster rate (ten times the fastest) or one slower rate (a third of
    the slowest). A K-component fit started this way can do no worse than the (K - 1) fit."""
    d = pd.read_csv(path, dtype={"month": str})
    r = d[(d.sym == sym) & (d.month == month) & (d.definition == definition) & (d.K == K - 1)]
    if r.empty:
        return []
    b = 1.0 / np.array([float(x) for x in str(r.iloc[0].memory_s).split(";")])
    return [np.sort(np.append(b, b.max() * 10.0))[::-1], np.sort(np.append(b, b.min() / 3.0))[::-1]]


def fit_job(roots, sym, month, definition, K, min_state, bin_minutes, oos, kernel="symmetric",
            baseline="shared", starts="default", delta=0.0, extra_from=None):
    t0 = time.time()
    days = build_days(month_files(roots, sym, month), definition, min_state)
    if starts == "full" and extra_from:
        st = list(hk.full_starts(K)) + extra_starts(extra_from, sym, month, definition, K)
        starts = "full+ladder"
    elif starts == "full":
        st = hk.full_starts(K)
    elif extra_from:
        st = hk._default_starts(K) + extra_starts(extra_from, sym, month, definition, K)
        starts = "default+ladder"
    else:
        st = None
    f = hk.fit(days, K=K, bin_minutes=bin_minutes, kernel=kernel, baseline=baseline, starts=st,
               delta=delta)
    ks = hk.ks_residuals(days, f)
    A = f.alpha_matrix()
    row = {"sym": sym, "month": month, "definition": definition, "K": K,
           "kernel": kernel, "baseline": baseline, "starts": starts, "delta": delta, "bin_minutes": bin_minutes,
           "n_days": len(days), "n_events": f.n_events,
           "memory_s": "; ".join(f"{m:.4g}" for m in f.timescales()),
           "excitation_per_scale": "; ".join(f"{a:.4f}" for a in A.sum(axis=(0, 1)) / 2),
           "effective_excitation": "; ".join(f"{a:.4f}" for a in f.effective_excitation()),
           "self_per_scale": "; ".join(f"{a:.4f}" for a in A[0, 0]),
           "cross_per_scale": "; ".join(f"{a:.4f}" for a in A[0, 1]),
           "branching_ratio": round(float(f.branching_ratio), 4),
           "loglik": round(float(f.loglik), 2), "aic": round(float(f.aic), 2),
           "bic": round(float(f.bic), 2),
           "ks_up": round(float(ks["up"].statistic), 4),
           "ks_down": round(float(ks["down"].statistic), 4)}
    if oos and len(days) > 3:
        f_even = hk.fit(days[::2], K=K, bin_minutes=bin_minutes, kernel=kernel, baseline=baseline,
                        starts=st, delta=delta)
        row["oos_loglik"] = round(float(hk.refit_baseline(days[1::2], f_even)), 2)
    row["minutes"] = round((time.time() - t0) / 60, 2)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nbbo", nargs="+", required=True, help="folders of saved NBBO changes")
    p.add_argument("--months", nargs="+", required=True, help="YYYYMM")
    p.add_argument("--syms", nargs="+", default=SAMPLE_STOCKS)
    p.add_argument("--definitions", nargs="+", default=["E0", "E1", "E1m"])
    p.add_argument("--K", nargs="+", type=int, default=[3])
    p.add_argument("--min-state", type=float, default=0.001)
    p.add_argument("--kernel", choices=["symmetric", "full"], default="symmetric")
    p.add_argument("--baseline", choices=["shared", "separate"], default="shared",
                   help="one baseline shared by up and down moves, or one per direction")
    p.add_argument("--starts", choices=["default", "full"], default="default",
                   help="full: every grid combination of decay rates plus empirical ladders "
                        "(about 3-4x slower; use when local optima are a concern)")
    p.add_argument("--extra-starts-from", default=None,
                   help="CSV of fits with K - 1 components for the same series; adds two starts built "
                        "from that fit to the default grid (see extra_starts)")
    p.add_argument("--delta", type=float, default=0.0,
                   help="refractory period after each event, seconds (0.001 matches E1m; E1m only)")
    p.add_argument("--bin-minutes", type=float, default=30)
    p.add_argument("--oos", action="store_true", help="also fit even days and score odd days")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    done = set()
    if out.exists():
        prev = pd.read_csv(out, dtype={"month": str})
        done = set(zip(prev["sym"], prev["month"], prev["definition"], prev["K"]))
    jobs = [(s, m, d, k) for m in args.months for s in args.syms for d in args.definitions
            for k in args.K if (s, m, d, k) not in done]
    print(f"{len(jobs)} fits to run", flush=True)
    # a fresh process per fit keeps memory from accumulating across fits
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(fit_job, args.nbbo, s, m, d, k, args.min_state, args.bin_minutes,
                               args.oos, args.kernel, args.baseline, args.starts,
                               args.delta, args.extra_starts_from): (s, m, d, k)
                   for s, m, d, k in jobs}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                print("FAILED", futures[fut], repr(e)[:200], flush=True)
                continue
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(row, flush=True)


if __name__ == "__main__":
    main()

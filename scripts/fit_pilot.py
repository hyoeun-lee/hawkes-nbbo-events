"""Fit K = 1, 2, 3 exponential kernels to E0 and E1 events for one stock and compare.

Example (from the repository root, after build_events.py):
    python scripts/fit_pilot.py --events events_pilot --sym INTC

Prints, for each event definition and K: the decay rates (as memory 1/beta in seconds),
total excitation per timescale, branching ratio, log-likelihood, AIC, BIC, KS statistics of
the time-rescaled residuals, and the out-of-sample log-likelihood (kernel fitted on even
days, scored on odd days with baselines re-estimated). Saves the table as CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_taq import hawkes as hk  # noqa: E402


def fit_table(days, definition, Ks, bin_minutes):
    rows = []
    train, test = days[::2], days[1::2]
    for K in Ks:
        f = hk.fit(days, K=K, bin_minutes=bin_minutes)
        ks = hk.ks_residuals(days, f)
        f_train = hk.fit(train, K=K, bin_minutes=bin_minutes)
        A = f.alpha_matrix()
        rows.append({
            "definition": definition, "K": K, "n_events": f.n_events,
            "memory_s": "; ".join(f"{m:.4g}" for m in f.timescales()),
            "excitation_per_scale": "; ".join(f"{a:.3f}" for a in A.sum(axis=(0, 1)) / 2),
            "branching_ratio": round(float(f.branching_ratio), 3),
            "loglik": round(float(f.loglik), 1), "aic": round(float(f.aic), 1), "bic": round(float(f.bic), 1),
            "ks_up": round(float(ks["up"].statistic), 4), "ks_down": round(float(ks["down"].statistic), 4),
            "oos_loglik": round(float(hk.refit_baseline(test, f_train)), 1) if test else None,
        })
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True, help="output folder of build_events.py")
    p.add_argument("--sym", default="INTC")
    p.add_argument("--definitions", nargs="+", default=["E0", "E1"])
    p.add_argument("--K", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--bin-minutes", type=float, default=30)
    p.add_argument("--out", default=None, help="CSV path (default: <events>/fit_<sym>.csv)")
    args = p.parse_args()

    tables = []
    for definition in args.definitions:
        days = hk.load_days(args.events, args.sym, definition)
        days = [d for d in days if len(d.t) > 0]
        print(f"{definition}: {len(days)} days, {sum(len(d.t) for d in days)} events", flush=True)
        tables.append(fit_table(days, definition, args.K, args.bin_minutes))
    table = pd.concat(tables, ignore_index=True)
    out = args.out or str(Path(args.events) / f"fit_{args.sym}.csv")
    table.to_csv(out, index=False)
    print(table.to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()

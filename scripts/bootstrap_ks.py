"""Parametric bootstrap for one stock-month: calibrated KS tests and standard errors.

Example:
    python scripts/bootstrap_ks.py --nbbo nbbo/2017Q2 --sym INTC --month 201705 \\
        --definition E1m --K 4 --delta 0.001 --starts full --n-boot 99 \\
        --workers 8 --out-dir boot/INTC_201705

Fits the model, then repeatedly simulates from the fit, refits (starting from the fitted decay
rates) and recomputes the KS statistics. Prints the observed statistics with bootstrap p-values.

With --out-dir, the parameters of every refit are saved, one row per replicate, so standard
errors and percentile intervals can be computed for every parameter (memories, excitation per
scale, effective excitation, branching ratios). Files in --out-dir:

    fit.csv         the fit to the data (one row)
    replicates.csv  one row per replicate (appended as they finish; rerunning resumes)
    summary.csv     per parameter: estimate, bootstrap mean, standard error, 2.5% and 97.5%
                    percentiles; KS p-values

Each replicate has its own random stream (seed, replicate number), so --workers runs them in
parallel and an interrupted run can be resumed. Runtime is roughly n_boot fits divided by the
number of workers.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fit_panel import build_days, month_files  # noqa: E402
from hawkes_taq import hawkes as hk  # noqa: E402


def param_row(f):
    """The fitted parameters of one model as a flat dict (components ordered fast to slow)."""
    row = {"loglik": float(f.loglik), "branching_ratio": float(f.branching_ratio),
           "effective_br": float(f.effective_excitation().sum()),
           "baseline_events_per_day": float(np.sum(f.mu * np.diff(f.edges)) * 2 / f.mu.shape[0])}
    A = f.alpha_matrix()
    eff = f.effective_excitation()
    for k in range(f.K):
        row[f"memory_{k + 1}"] = float(1.0 / f.beta[k])
        row[f"excitation_{k + 1}"] = float(A[0, :, k].sum())
        row[f"effective_excitation_{k + 1}"] = float(eff[k])
        if f.kernel == "symmetric":
            row[f"self_{k + 1}"] = float(A[0, 0, k])
            row[f"cross_{k + 1}"] = float(A[0, 1, k])
        else:
            for i, a in enumerate("ud"):
                for j, c in enumerate("ud"):
                    row[f"alpha_{a}{c}_{k + 1}"] = float(A[i, j, k])
    return row


_DAYS, _FIT = None, None


def _init(days, fit_):
    global _DAYS, _FIT
    _DAYS, _FIT = days, fit_


def _replicate(seed, b):
    f_b, ks = hk.bootstrap_replicate(_DAYS, _FIT, seed, b)
    return {"rep": b, "seed": seed, "ks_up": float(ks["up"]), "ks_down": float(ks["down"]),
            **param_row(f_b)}


def summarize(fit_row, reps, obs):
    """Standard errors and percentile intervals for every parameter, plus KS p-values."""
    rows = []
    for col, est in fit_row.items():
        if col not in reps:
            continue
        x = reps[col].to_numpy(dtype=float)
        rows.append({"parameter": col, "estimate": est, "boot_mean": float(np.mean(x)),
                     "se": float(np.std(x, ddof=1)), "q025": float(np.quantile(x, 0.025)),
                     "q975": float(np.quantile(x, 0.975))})
        if col.startswith("memory_"):         # memories are skewed: also on the log scale
            rows.append({"parameter": "log_" + col, "estimate": float(np.log(est)),
                         "boot_mean": float(np.mean(np.log(x))),
                         "se": float(np.std(np.log(x), ddof=1)),
                         "q025": float(np.quantile(np.log(x), 0.025)),
                         "q975": float(np.quantile(np.log(x), 0.975))})
    pv = hk.bootstrap_pvalues(obs, {"up": reps["ks_up"], "down": reps["ks_down"]})
    for side in ("up", "down"):
        rows.append({"parameter": f"ks_{side}", "estimate": pv[side]["ks"],
                     "boot_mean": pv[side]["boot_median"], "se": np.nan,
                     "q025": np.nan, "q975": pv[side]["boot_95"], "p_boot": pv[side]["p_boot"]})
    return pd.DataFrame(rows), pv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--sym", required=True)
    p.add_argument("--month", required=True)
    p.add_argument("--definition", default="E1m")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--kernel", choices=["symmetric", "full"], default="symmetric")
    p.add_argument("--baseline", choices=["shared", "separate"], default="shared")
    p.add_argument("--starts", choices=["default", "full"], default="default",
                   help="starting decay rates for the fit to the data (replicates start from it)")
    p.add_argument("--bin-minutes", type=float, default=30,
                   help="length of the baseline bins in minutes (330 = one level per day)")
    p.add_argument("--n-boot", type=int, default=99)
    p.add_argument("--min-state", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.0, help="refractory period, seconds")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out-dir", help="save every replicate's parameters and a summary here")
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else None
    days = build_days(month_files(args.nbbo, args.sym, args.month), args.definition, args.min_state)
    starts = hk.full_starts(args.K) if args.starts == "full" else None
    f = hk.fit(days, K=args.K, kernel=args.kernel, baseline=args.baseline, starts=starts,
               delta=args.delta, bin_minutes=args.bin_minutes)
    obs = {k: float(r.statistic) for k, r in hk.ks_residuals(days, f).items()}
    fit_row = {"ks_up": obs["up"], "ks_down": obs["down"], **param_row(f)}
    print(f"fit: memory {np.round(f.timescales(), 5)}, branching ratio {f.branching_ratio:.3f}, "
          f"effective {f.effective_excitation().sum():.3f}, loglik {f.loglik:.1f}, "
          f"BIC {f.bic:.1f}", flush=True)

    rep_file = None
    done = set()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        meta = {"sym": args.sym, "month": args.month, "definition": args.definition, "K": args.K,
                "kernel": args.kernel, "baseline": args.baseline, "delta": args.delta,
                "min_state": args.min_state, "bin_minutes": args.bin_minutes,
                "n_events": f.n_events, "n_days": len(days), "bic": float(f.bic)}
        pd.DataFrame([{**meta, **fit_row}]).to_csv(out / "fit.csv", index=False)
        rep_file = out / "replicates.csv"
        if rep_file.exists():
            prev = pd.read_csv(rep_file)
            done = set(prev.loc[prev["seed"] == args.seed, "rep"].astype(int))
    todo = [b for b in range(args.n_boot) if b not in done]
    print(f"{len(done)} replicates done, {len(todo)} to run", flush=True)

    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(days, f)) as pool:
        futures = [pool.submit(_replicate, args.seed, b) for b in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            if rep_file:
                pd.DataFrame([row]).to_csv(rep_file, mode="a", header=not rep_file.exists(),
                                           index=False)
            print(f"replicate {row['rep']} done ({n}/{len(todo)}): branching ratio "
                  f"{row['branching_ratio']:.3f}, KS {row['ks_up']:.4f} / {row['ks_down']:.4f}",
                  flush=True)

    reps = pd.read_csv(rep_file) if rep_file else pd.DataFrame(rows)
    reps = reps[reps["seed"] == args.seed].drop_duplicates("rep")
    summary, pv = summarize(fit_row, reps, obs)
    if out:
        summary.to_csv(out / "summary.csv", index=False)
    for side, r in pv.items():
        print(f"{args.sym} {args.month} {args.definition} K={args.K} {args.kernel}/{args.baseline} "
              f"delta={args.delta} "
              f"{side}: KS {r['ks']:.4f}, "
              f"bootstrap p {r['p_boot']:.3f} (bootstrap median {r['boot_median']:.4f}, "
              f"95th pct {r['boot_95']:.4f})")
    est = summary.set_index("parameter")
    keys = [f"memory_{k + 1}" for k in range(args.K)] + ["effective_br", "branching_ratio"]
    print("standard errors (n = %d replicates):" % len(reps))
    for k in keys:
        print(f"  {k}: {est.loc[k, 'estimate']:.5g} (se {est.loc[k, 'se']:.3g}, "
              f"95% interval {est.loc[k, 'q025']:.5g} to {est.loc[k, 'q975']:.5g})")


if __name__ == "__main__":
    main()

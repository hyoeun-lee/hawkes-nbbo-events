"""Summarize simulation-study results (rows written by scripts/simulate_study.py run).

For each stock, experiment, event definition, K, dead time and pooling: median and RMSE of the
log ratio of each fitted memory to the true one (when K equals the true number of components),
bias of the effective branching ratio against the truth, baseline events against the truth,
KS statistics, and for raw events the accounting identity
    BR(E0) ~ share of E0 gaps < 1 ms + BR(E1m) * N(E1m) / N(E0).
Also the share of replicates in which BIC picks K = 3, 4 or 5 for E1m; bias of the fitted kernel
mass within 1 s and 10 s (n1, n10) against the truth (rows from the second batch onward); and for
the actscale experiment (one kernel, baselines scaled by 0.5-2), how the estimates move with
activity although the true kernel does not.

    python scripts/summarize_simulation.py results/*.csv --out summary.csv
"""

import argparse
import glob

import numpy as np
import pandas as pd


def parse(col):
    return col.astype(str).str.split("; ").apply(lambda v: np.array([float(x) for x in v]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--out")
    args = p.parse_args()
    files = [f for g in args.files for f in glob.glob(g)]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "error" in d:
        print("failed fits:", int(d["error"].notna().sum()))
        d = d[d["error"].isna()]
    d["mem"], d["tmem"] = parse(d["memory_s"]), parse(d["true_memory_s"])
    d["teff"] = d["true_br"]

    rows = []
    keys = ["sym", "experiment", "definition", "K", "delta", "pool"]
    for k, g in d.groupby(keys):
        r = dict(zip(keys, k), n_reps=len(g))
        r["br_eff_median"] = g["effective_br"].median()
        r["true_br"] = g["true_br"].iloc[0]
        r["br_eff_bias"] = (g["effective_br"] - g["true_br"]).median()
        r["baseline_ratio"] = (g["baseline_events"] / g["true_baseline_events"]).median()
        r["ks_up"], r["ks_down"] = g["ks_up"].median(), g["ks_down"].median()
        if "n1" in g and g["n1"].notna().any():
            r["n1_median"], r["true_n1"] = g["n1"].median(), g["true_n1"].iloc[0]
            r["n1_bias"] = (g["n1"] - g["true_n1"]).median()
            r["n10_median"], r["true_n10"] = g["n10"].median(), g["true_n10"].iloc[0]
            r["n10_bias"] = (g["n10"] - g["true_n10"]).median()
        same = g[g["mem"].apply(len) == g["tmem"].apply(len)]
        if len(same):
            lr = np.vstack([np.log(m / t) for m, t in zip(same["mem"], same["tmem"])])
            for j in range(lr.shape[1]):
                r[f"logratio_mem{j + 1}_median"] = float(np.median(lr[:, j]))
                r[f"logratio_mem{j + 1}_rmse"] = float(np.sqrt(np.mean(lr[:, j] ** 2)))
        r["fastest_memory_median_ms"] = 1e3 * float(np.median([m.min() for m in g["mem"]]))
        r["minutes_median"] = g["minutes"].median()
        rows.append(r)
    s = pd.DataFrame(rows)

    # accounting identity, per replicate (E0 K=4 vs E1m K=4 with the dead time, all days)
    e0 = d[(d.definition == "E0") & (d.K == 4) & (d.pool == "all")].set_index(["sym", "experiment", "rep"])
    em = d[(d.definition == "E1m") & (d.K == 4) & (d.delta > 0) & (d.pool == "all")] \
        .set_index(["sym", "experiment", "rep"])
    j = e0[["branching_ratio", "e0_gap_lt1ms", "n_e0", "n_e1m"]].join(em[["effective_br"]], how="inner")
    if len(j):
        j["identity_pred"] = j.e0_gap_lt1ms + j.effective_br * j.n_e1m / j.n_e0
        j["identity_err"] = j.branching_ratio - j.identity_pred
        print("Accounting identity, BR(E0) minus prediction (median by stock and experiment):")
        print(j.groupby(level=["sym", "experiment"]).identity_err.median().round(4).to_string())

    # K selection by BIC (E1m with the dead time, all days)
    kb = d[(d.definition == "E1m") & (d.delta > 0) & (d.pool == "all")]
    if kb.K.nunique() > 1:
        best = kb.loc[kb.groupby(["sym", "experiment", "rep"]).bic.idxmin()]
        print("\nK chosen by BIC (share of replicates):")
        print(best.groupby(["sym", "experiment"]).K.value_counts(normalize=True).round(2).to_string())

    # actscale: one kernel, baselines scaled; does the estimate move with activity?
    act = d[(d.experiment == "actscale") & (d.definition == "E1m") & (d.K == 4) & (d.delta > 0)]
    if len(act) and "baseline_scale" in act:
        print("\nactscale (true kernel fixed): E1m K=4 estimates by baseline scale (medians):")
        cols = [c for c in ("n_e1m", "effective_br", "n1", "n10") if c in act]
        print(act.groupby(["sym", "baseline_scale"])[cols].median().round(4).to_string())
        for sym, g in act.groupby("sym"):
            for c in ("effective_br", "n1", "n10"):
                if c in g and g[c].notna().sum() > 2:
                    r = np.corrcoef(np.log(g["n_e1m"]), g[c])[0, 1]
                    print(f"  {sym}: correlation of log events with {c}: {r:.2f}")

    pd.set_option("display.width", 250)
    print("\n" + s.round(4).to_string())
    if args.out:
        s.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

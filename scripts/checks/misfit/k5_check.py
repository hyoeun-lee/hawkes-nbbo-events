"""Model order at the top: K = 5 against K = 4 on the real data (revision item A12).

For each stock-month in --fits5 (E1m fits with K = 5), the K = 4 fit of the same stock-month
in --fits4 is compared on BIC and AIC, the horizon masses n(1 s) and n(10 s), the total and the
fifth component's memory and weight. With --nbbo the burst diagnostic of results §23.C is added:
the rate of events 10-100 s after an event relative to the unconditional rate, in the data and in
three months simulated from each fit (K = 4 and K = 5), to see whether a fifth component absorbs
the bursts.

    python scripts/checks/misfit/k5_check.py --fits4 DATA/fits_e1m_delta_K4.csv --fits5 fits_K5.csv \\
        [--nbbo DATA/nbbo_2017May --out k5_check.csv]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hawkes_taq import hawkes as hk  # noqa: E402
import fit_panel as fp  # noqa: E402
from score_windows import baseline_refit, parse  # noqa: E402
from response import response  # noqa: E402


def mass_row(r, lo, H):
    m, e = parse(r.memory_s), parse(r.excitation_per_scale)
    return float(np.sum(e * (np.exp(-lo / m) - np.exp(-H / m))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits4", required=True)
    p.add_argument("--fits5", required=True)
    p.add_argument("--nbbo", nargs="*", default=None)
    p.add_argument("--sims", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    f4 = pd.read_csv(a.fits4, dtype={"month": str}); f5 = pd.read_csv(a.fits5, dtype={"month": str})
    rows = []
    for r5 in f5.itertuples():
        r4 = f4[(f4.sym == r5.sym) & (f4.month == r5.month) & (f4.definition == r5.definition) & (f4.K == r5.K - 1)].iloc[0]
        h = float(r5.delta)
        m5, e5 = parse(r5.memory_s), parse(r5.excitation_per_scale); m4 = parse(r4.memory_s)
        # the component of the K = 5 fit farthest (in log memory) from any K = 4 memory is "the new one"
        dist = [min(abs(np.log(m / m4))) for m in m5]
        k_new = int(np.argmax(dist))
        row = {"sym": r5.sym, "month": r5.month, "h": h, "K4_bic": r4.bic, "K5_bic": r5.bic, "dBIC_5_minus_4": r5.bic - r4.bic,
               "dAIC_5_minus_4": r5.aic - r4.aic, "dloglik": r5.loglik - r4.loglik,
               "K4_n1": mass_row(r4, h, 1.0), "K5_n1": mass_row(r5, h, 1.0), "K4_n10": mass_row(r4, h, 10.0), "K5_n10": mass_row(r5, h, 10.0),
               "K4_total": float(parse(r4.effective_excitation).sum()), "K5_total": float(parse(r5.effective_excitation).sum()),
               "K4_memory_s": r4.memory_s, "K5_memory_s": r5.memory_s, "K5_excitation": r5.excitation_per_scale,
               "new_memory_s": m5[k_new], "new_weight": e5[k_new], "K5_minutes": r5.minutes, "K5_starts": r5.starts}
        if a.nbbo:
            days = fp.build_days(fp.month_files(a.nbbo, r5.sym, r5.month), "E1m", h)
            real = response(days)
            for tag, r in (("K4", r4), ("K5", r5)):
                beta = 1.0 / parse(r.memory_s); alpha = np.vstack([parse(r.self_per_scale), parse(r.cross_per_scale)])
                fit = baseline_refit(days, beta, alpha, h, 0.0)
                sims = [response(hk.simulate_like(days, fit, np.random.default_rng([23, b, len(r5.sym)]))) for b in range(a.sims)]
                row[f"{tag}_loglik_recon"] = fit.loglik
                for w, name in enumerate(["10-100ms", "0.1-1s", "1-10s", "10-100s"]):
                    row[f"{tag}_sim_same_{name}"] = float(np.mean([s[0][w] for s in sims]))
                    row[f"{tag}_sim_opp_{name}"] = float(np.mean([s[1][w] for s in sims]))
            for w, name in enumerate(["10-100ms", "0.1-1s", "1-10s", "10-100s"]):
                row[f"real_same_{name}"] = float(real[0][w]); row[f"real_opp_{name}"] = float(real[1][w])
        rows.append(row)
        print(f"{r5.sym} {r5.month}: dBIC {row['dBIC_5_minus_4']:+.1f} dAIC {row['dAIC_5_minus_4']:+.1f} dloglik {row['dloglik']:+.1f} | "
              f"n1 {row['K4_n1']:.4f} -> {row['K5_n1']:.4f} | n10 {row['K4_n10']:.4f} -> {row['K5_n10']:.4f} | total {row['K4_total']:.3f} -> {row['K5_total']:.3f} | "
              f"new component {row['new_memory_s']:.4g} s, weight {row['new_weight']:.4f} | K5 memories {r5.memory_s}", flush=True)
        if a.nbbo:
            print(f"   10-100 s ratio (same+opp)/2: real {(row['real_same_10-100s'] + row['real_opp_10-100s']) / 2:.2f} | "
                  f"K4 sim {(row['K4_sim_same_10-100s'] + row['K4_sim_opp_10-100s']) / 2:.2f} | K5 sim {(row['K5_sim_same_10-100s'] + row['K5_sim_opp_10-100s']) / 2:.2f}", flush=True)
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

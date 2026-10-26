"""Paired day-block bootstrap of raw (E0) and clean (E1m) fits for one stock-month (review point E).

Trading days are resampled in blocks of --block consecutive days, with replacement, and the
same resampled days are used for both event definitions, so the raw/clean contrasts (kernel
mass within 1 s and 10 s, totals) get model-free standard errors that respect day-to-day
dependence. Each replicate refits E1m (K = 4, 1 ms dead time; --h and --K-clean change the threshold and
order, for BAC at 2 ms with K = 3) and E0 (K = 4), starting from the main fits' decay rates;
--definitions clean refits E1m only, for model-free standard errors of a month's n(1 s) and n(10 s). Replicates are independent streams (seed, b), so the work can be split
across array tasks with --rep-start/--rep-count and resumed.

    python scripts/block_bootstrap.py --nbbo DATA/nbbo_2017Q2 --sym INTC --month 201705 \\
        --fits-clean DATA/fits_e1m_delta_K4.csv --fits-raw DATA/fits_e0_k4_INTC.csv \\
        --rep-start 0 --rep-count 10 --out DATA/blockboot/INTC_201705.csv
"""

import argparse
import math
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


def mass(f, lo, H):
    return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))


def start_beta(path, sym, month, definition, K):
    d = pd.read_csv(path, dtype={"month": str})
    r = d[(d.sym == sym) & (d.month == month) & (d.definition == definition) & (d.K == K)]
    if r.empty:
        raise SystemExit(f"no {definition} K={K} fit for {sym} {month} in {path}")
    return 1.0 / np.array([float(x) for x in str(r.iloc[0].memory_s).split(";")])


def block_indices(n, block, rng):
    """Indices of a block-bootstrap sample of n days: blocks of `block` consecutive days drawn
    with replacement from the n - block + 1 possible starts, concatenated and cut to n."""
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def fit_summary(days, beta0, K, delta, tag):
    f = hk.fit(days, K=K, kernel="symmetric", bin_minutes=30, starts=[beta0], baseline="shared", delta=delta)
    return {f"{tag}_n_events": f.n_events,
            f"{tag}_memory_s": "; ".join(f"{m:.4g}" for m in f.timescales()),
            f"{tag}_excitation": "; ".join(f"{x:.4f}" for x in f.alpha.sum(axis=0)),
            f"{tag}_br": round(float(f.branching_ratio), 4),
            f"{tag}_n1": round(mass(f, delta, 1.0), 4), f"{tag}_n10": round(mass(f, delta, 10.0), 4),
            f"{tag}_n1ms": round(mass(f, 0.0, 0.001), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--sym", required=True)
    p.add_argument("--month", required=True)
    p.add_argument("--fits-clean", required=True)
    p.add_argument("--fits-raw", default=None, help="raw (E0) fits; needed unless --definitions clean")
    p.add_argument("--K", type=int, default=4, help="components for both fits unless overridden")
    p.add_argument("--K-clean", type=int, default=None)
    p.add_argument("--K-raw", type=int, default=None)
    p.add_argument("--h", type=float, default=0.001,
                   help="merge threshold and refractory period of the clean series (seconds)")
    p.add_argument("--definitions", choices=["both", "clean"], default="both",
                   help="clean: refit E1m only (no raw fits, no ratios)")
    p.add_argument("--block", type=int, default=5)
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--rep-count", type=int, default=99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    K_clean = a.K_clean or a.K
    K_raw = a.K_raw or a.K
    paired = a.definitions == "both"
    files = fp.month_files(a.nbbo, a.sym, a.month)
    clean = fp.build_days(files, "E1m", a.h)
    n = len(clean)
    b_clean = start_beta(a.fits_clean, a.sym, a.month, "E1m", K_clean)
    if paired:
        if a.fits_raw is None:
            raise SystemExit("--fits-raw is needed for the paired bootstrap")
        raw = fp.build_days(files, "E0", 0.001)
        assert [d.label for d in clean] == [d.label for d in raw], "day lists differ between definitions"
        b_raw = start_beta(a.fits_raw, a.sym, a.month, "E0", K_raw)
    done = set()
    if Path(a.out).exists():
        done = set(pd.read_csv(a.out).rep)
    for b in range(a.rep_start, a.rep_start + a.rep_count):
        if b in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng([a.seed, b])
        idx = block_indices(n, a.block, rng)
        row = {"sym": a.sym, "month": a.month, "rep": b, "block": a.block, "h": a.h,
               "K_clean": K_clean, "days": " ".join(clean[i].label for i in idx)}
        row.update(fit_summary([clean[i] for i in idx], b_clean, K_clean, a.h, "clean"))
        if paired:
            row.update(fit_summary([raw[i] for i in idx], b_raw, K_raw, 0.0, "raw"))
            row["ratio_n1"] = round(row["raw_n1"] / row["clean_n1"], 4)
            row["ratio_n10"] = round(row["raw_n10"] / row["clean_n10"], 4)
        row["minutes"] = round((time.time() - t0) / 60, 2)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(a.out, mode="a", header=not Path(a.out).exists(), index=False)
        print(a.sym, a.month, "rep", b, "clean n1", row["clean_n1"], "raw n1", row.get("raw_n1", "-"),
              "ratio", row.get("ratio_n1", "-"), f"{row['minutes']} min", flush=True)


if __name__ == "__main__":
    main()

"""Measure how clean price moves pass through brief states, per stock-month (simulation inputs).

For every pair of consecutive surviving E1m states (the states E1m keeps), the states between
them are the merged brief states ("chain"). Each pair is classified:

- clean move (the mid-price differs between the two surviving states):
  direct (no chain), via lock (the chain contains a locked or crossed state),
  via widening (the chain has only states with spread of 2 ticks or more), or via other
  (positive-spread chain containing a one-tick state, e.g. both quotes moving in turn);
- flicker (the mid-price is the same before and after the chain): a brief state that returns
  to the previous price, again split into lock, widening and other.

Also reported per stock-month: share of time at one tick excluding locked and crossed states
(the large-tick definition), and the Dayri-Rosenbaum eta on E1m moves,
eta = N_continuations / (2 N_alternations).

Writes a summary table (one row per stock-month) and, with --durations, the chain durations
and sizes per stock-month for resampling in the simulation. No TAQ data are written to the
repository; outputs go to --out.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_taq import events as ev  # noqa: E402
from hawkes_taq import hawkes as hk  # noqa: E402
from hawkes_taq.pipeline import HALF_DAYS  # noqa: E402


def classify_day(states, max_nonpos=0.005, min_state=0.001):
    s = states
    dur, spread, mid2, t = (s[c].to_numpy() for c in ("dur", "spread", "mid2", "t"))
    transient = ((spread <= 0) & (dur < max_nonpos)) | (dur == 0) | (dur < min_state)
    kept = np.flatnonzero(~transient)
    rows = []
    for prev, cur in zip(kept[:-1], kept[1:]):
        d = int(mid2[cur] - mid2[prev])
        chain = np.arange(prev + 1, cur)
        if len(chain) == 0:
            if d != 0:
                rows.append(("move", "direct", 0.0, 0, d, mid2[prev], spread[prev], spread[cur]))
            continue
        sp = spread[chain]
        kind = "lock" if (sp <= 0).any() else ("widen" if (sp >= 2).all() else "other")
        length = t[cur] - t[chain[0]]
        rows.append(("move" if d != 0 else "flicker", kind, length, len(chain), d, mid2[prev],
                     spread[prev], spread[cur]))
    out = pd.DataFrame(rows, columns=["what", "kind", "chain_s", "n_states", "d_mid2", "mid2_before",
                                      "spread_before", "spread_after"])
    return out, kept


def stock_month(files):
    parts, eta_c, eta_a, t1, tpos, nlong, nE0 = [], 0, 0, 0.0, 0.0, 0, 0
    for f in files:
        s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
        nE0 += len(ev.events_e0(s))
        c, kept = classify_day(s)
        c["day"] = f.name[:8]
        parts.append(c)
        pos = s["spread"] > 0
        tpos += s.loc[pos, "dur"].sum()
        t1 += s.loc[s["spread"] == 1, "dur"].sum()
        nlong += int(((s["spread"] <= 0) & (s["dur"] >= 0.005)).sum())
        mv = np.sign(c.loc[c.what == "move", "d_mid2"].to_numpy())
        same = mv[1:] == mv[:-1]
        eta_c += int(same.sum())
        eta_a += int((~same).sum())
    return pd.concat(parts, ignore_index=True), dict(
        share_time_1tick_excl_nonpos=t1 / tpos, eta=eta_c / (2 * eta_a) if eta_a else np.nan,
        n_long_nonpos=nlong, n_e0=nE0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--syms", nargs="+", required=True)
    p.add_argument("--months", nargs="+", required=True)
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--durations", action="store_true", help="also save chain-level tables")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in args.syms:
        for month in args.months:
            files = sorted([f for r in args.nbbo for f in Path(r, sym).glob(f"{month}*.csv.gz")
                            if f.name[:8] not in HALF_DAYS], key=lambda f: f.name)
            if not files:
                continue
            c, extra = stock_month(files)
            mv, fl = c[c.what == "move"], c[c.what == "flicker"]
            n = len(mv)
            row = {"sym": sym, "month": month, "days": len(files), "n_clean_moves": n,
                   "n_flickers": len(fl), "n_e0": extra["n_e0"],
                   "share_direct": (mv.kind == "direct").mean(),
                   "share_via_lock": (mv.kind == "lock").mean(),
                   "share_via_widen": (mv.kind == "widen").mean(),
                   "share_via_other": (mv.kind == "other").mean(),
                   "flickers_per_move": len(fl) / n if n else np.nan,
                   "flicker_share_lock": (fl.kind == "lock").mean() if len(fl) else np.nan,
                   "flicker_share_widen": (fl.kind == "widen").mean() if len(fl) else np.nan,
                   "share_moves_size1tick": (mv.d_mid2.abs() == 2).mean(),
                   "share_moves_halftick": (mv.d_mid2.abs() == 1).mean(),
                   "median_lock_chain_ms": 1e3 * mv.loc[mv.kind == "lock", "chain_s"].median(),
                   "p90_lock_chain_ms": 1e3 * mv.loc[mv.kind == "lock", "chain_s"].quantile(0.9),
                   "median_widen_chain_ms": 1e3 * mv.loc[mv.kind == "widen", "chain_s"].median(),
                   "median_flicker_ms": 1e3 * fl["chain_s"].median() if len(fl) else np.nan}
            row.update(extra)
            rows.append(row)
            if args.durations:
                c.to_csv(out / f"chains_{sym}_{month}.csv.gz", index=False)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "calibration_summary.csv", index=False)


if __name__ == "__main__":
    main()

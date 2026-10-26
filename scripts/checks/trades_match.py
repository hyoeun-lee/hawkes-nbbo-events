"""Depletion against flicker: are brief positive-spread NBBO states preceded by a trade?

A sub-millisecond positive-spread state begins when one side of the NBBO moves away (the ask rises
or the bid falls) and ends within 1 ms. If a trade took the whole displayed quantity at that price
just before, the state is depletion-and-refill (trading); if no trade is near, it is a quote that
was placed or pulled and replaced (flicker). For each state we look for a trade at the price that
disappeared, on the side that moved, in a window around the state's start (SIP time for both; trade
reports reach the SIP on their own path, so the window is generous and asymmetric). Controls: the
same statistic for persistent moves (states lasting at least 1 ms that moved the mid), and the share
expected by chance from the trade rate.

Usage: python scripts/checks/trades_match.py <nbbo_dir> <trades_dir> [SYMS...]
  nbbo_dir/<SYM>/<date>.csv.gz  (t, bid, ask; the project's NBBO change files)
  trades_dir/<SYM>/<date>.csv.gz (t, t_ex, ex, price, size, cond; from sql/15_pull_trades.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hawkes_taq import events as ev  # noqa: E402
from hawkes_taq.hawkes import SESSION_END  # noqa: E402

TICK = ev.TICK
WINDOWS = [(-0.003, 0.0005), (-0.010, 0.001), (-0.001, 0.0002)]   # (before, after) in seconds


def classify(states):
    """Which side moved into each state, and the price that disappeared (in ticks)."""
    bid, ask = states.bid.to_numpy(), states.ask.to_numpy()
    side = np.full(len(states), "", dtype=object)
    gone = np.zeros(len(states), dtype=np.int64)
    up_ask = np.r_[False, ask[1:] > ask[:-1]]      # ask moved away: the old ask was taken or pulled
    down_bid = np.r_[False, bid[1:] < bid[:-1]]    # bid moved away: the old bid was hit or pulled
    side[up_ask] = "ask"; gone[up_ask] = np.r_[0, ask[:-1]][up_ask]
    side[down_bid] = "bid"; gone[down_bid] = np.r_[0, bid[:-1]][down_bid]
    both = up_ask & down_bid
    side[both] = "both"
    return side, gone


def matched(t0, price_ticks, tr_t, tr_p, lo, hi):
    """True where a trade at price_ticks occurs in [t0 + lo, t0 + hi]."""
    out = np.zeros(len(t0), dtype=bool)
    for p in np.unique(price_ticks):
        sel = price_ticks == p
        tt = tr_t[tr_p == p]
        if len(tt) == 0:
            continue
        a = np.searchsorted(tt, t0[sel] + lo, side="left")
        b = np.searchsorted(tt, t0[sel] + hi, side="right")
        out[sel] = b > a
    return out


def one_day(nbbo, trades):
    st = ev.build_states(nbbo, SESSION_END)
    side, gone = classify(st)
    brief = (st.dur.to_numpy() < 0.001)
    pos = st.spread.to_numpy() > 0
    moved = np.r_[False, np.diff(st.mid2.to_numpy()) != 0]
    tr_t = trades.t.to_numpy(); tr_p = np.rint(trades.price.to_numpy() / TICK).astype(np.int64)
    groups = {
        "brief positive-spread (<1 ms)": brief & pos & (side != "") & (side != "both"),
        "brief widening (2+ ticks, <1 ms)": brief & (st.spread.to_numpy() >= 2) & (side != "") & (side != "both"),
        "brief one-tick state (<1 ms, mid moved)": brief & (st.spread.to_numpy() == 1) & moved & (side != "") & (side != "both"),
        "persistent move (>=1 ms, mid moved)": (~brief) & moved & (side != "") & (side != "both"),
    }
    t0 = st.t.to_numpy()
    rows = []
    rate = len(tr_t) / (SESSION_END - 36000.0)     # trades per second (any price)
    for name, g in groups.items():
        n = int(g.sum())
        r = {"group": name, "n": n}
        for lo, hi in WINDOWS:
            m = matched(t0[g], gone[g], tr_t, tr_p, lo, hi)
            r[f"match[{lo*1e3:+.0f},{hi*1e3:+.1f}] ms"] = float(m.mean()) if n else np.nan
        # chance: a trade at any price in the widest window
        lo, hi = WINDOWS[1]
        r["any trade in widest window"] = float(matched(t0[g], np.zeros(n, dtype=np.int64),
                                                       tr_t, np.zeros(len(tr_t), dtype=np.int64), lo, hi).mean()) if n else np.nan
        r["chance (rate x window)"] = float(min(1.0, rate * (hi - lo)))
        rows.append(r)
    return pd.DataFrame(rows)


def main(nbbo_dir, trades_dir, syms, exclude_trf=False):
    """exclude_trf: drop trades reported through the FINRA trade reporting facilities (TAQ exchange
    code D), which are off-exchange and cannot take out a displayed quote (revision item A22)."""
    for sym in syms:
        parts, n_all, n_trf = [], 0, 0
        for f in sorted(Path(nbbo_dir, sym).glob("*.csv.gz")):
            g = Path(trades_dir, sym, f.name)
            if not g.exists():
                continue
            tr = pd.read_csv(g)
            n_all += len(tr); n_trf += int((tr.ex == "D").sum())
            if exclude_trf:
                tr = tr[tr.ex != "D"]
            parts.append(one_day(pd.read_csv(f), tr))
        if not parts:
            print(sym, "no matching days"); continue
        d = pd.concat(parts)
        w = d.groupby("group", sort=False).apply(
            lambda x: pd.Series({c: np.average(x[c], weights=x.n) if c != "n" else x.n.sum() for c in d.columns if c != "group"}))
        print(f"\n{sym}: {len(parts)} days; trades {n_all}, of which TRF {n_trf} ({n_trf / max(n_all, 1):.1%})"
              f"{'; TRF excluded' if exclude_trf else ''}")
        print(w.round(3).to_string())


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--exclude-trf"]
    main(argv[0], argv[1], argv[2:] or ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"],
         exclude_trf="--exclude-trf" in sys.argv)

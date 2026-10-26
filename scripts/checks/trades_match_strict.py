"""Strict trade match for the brief positive-spread states (second mock review, C1).

The match of trades_match.py pairs a sub-millisecond positive-spread state with a trade at the
price that disappeared, on any exchange, within [-3, +0.5] ms of the state's start in SIP time.
This script tightens it in two steps, using the NBBO rebuilt from the per-exchange quotes with the
venue of the best bid and ask (sql/14_build_lf_nbbo.py --venues; lfnbbo_venues/{sip,ex}/SYM/date.csv.gz
with t, bid, ask, bid_ex, ask_ex), so that the venue whose quote vanished is known:

  loose          trade at the vanished price, any exchange (TRF excluded), [-3, +0.5] ms;
  same venue     the trade is on the exchange that held the vanished quote, same window;
  same venue,    same venue and the trade's participant timestamp within [-0.5, +0.1] ms of the
  tight          quote update's exchange timestamp (exchange clock) or the SIP times (SIP clock).

Battalio et al. (2026, Appendix B.2) describe the quote update that follows an execution as carrying
the trade's own exchange timestamp, so in the exchange clock the strict match should sit at an offset
near zero. The offset (quote update time minus trade time) of the nearest same-venue trade is
reported as quantiles. Displayed sizes are not in the venue files, so no size condition is applied.
The venue files tag the holder of the best quote as of the state's start, choosing the
alphabetically first venue among ties, so when a venue leaves the best without changing the NBBO
the tag is stale; the same-venue shares are therefore lower bounds. The execution signature that
does not depend on the tag is the identical timestamp: a trade at the vanished price, on any venue,
whose participant timestamp equals the quote update's exchange timestamp within 10 us (reported as
exact_10us, and within 100 us). Controls as before: persistent moves (states of at least 1 ms that moved the mid), and chance levels
(a trade at any price on any venue, and on the same venue, in the loose window).

    python scripts/checks/trades_match_strict.py --venues DATA/lfnbbo_venues --trades DATA/trades \\
        --syms INTC BAC MDLZ TGT --out results/trades_match_strict_201705.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TICK = 0.01
SESSION_START, SESSION_END = 36000.0, 55800.0
LOOSE = (-0.003, 0.0005)
TIGHT = (-0.0005, 0.0001)


def states_with_venues(df):
    """Price states from an NBBO file with venue columns: the venue that held the quote which
    vanished when a side moved away, taken from the row immediately before the change."""
    df = df.sort_values("t").reset_index(drop=True)
    bid = np.rint(df.bid.to_numpy() / TICK).astype(np.int64)
    ask = np.rint(df.ask.to_numpy() / TICK).astype(np.int64)
    prev_bid_ex = df.bid_ex.shift(1).to_numpy()
    prev_ask_ex = df.ask_ex.shift(1).to_numpy()
    change = np.ones(len(df), dtype=bool)
    change[1:] = (bid[1:] != bid[:-1]) | (ask[1:] != ask[:-1])
    t = df.t.to_numpy(dtype=float)
    pb = np.r_[bid[0], bid[:-1]]
    pa = np.r_[ask[0], ask[:-1]]
    s = pd.DataFrame({"t": t, "bid": bid, "ask": ask, "prev_bid": pb, "prev_ask": pa,
                      "prev_bid_ex": prev_bid_ex, "prev_ask_ex": prev_ask_ex})[change].reset_index(drop=True)
    s["dur"] = np.r_[np.diff(s.t.to_numpy()), SESSION_END - s.t.iloc[-1]]
    s["spread"] = s.ask - s.bid
    s["mid2"] = s.bid + s.ask
    s["moved"] = np.r_[False, np.diff(s.mid2.to_numpy()) != 0]
    up_ask = s.ask.to_numpy() > s.prev_ask.to_numpy()
    down_bid = s.bid.to_numpy() < s.prev_bid.to_numpy()
    side = np.where(up_ask & ~down_bid, "ask", np.where(down_bid & ~up_ask, "bid", ""))
    s["side"] = side
    s["gone"] = np.where(side == "ask", s.prev_ask, np.where(side == "bid", s.prev_bid, 0))
    s["venue"] = np.where(side == "ask", s.prev_ask_ex, np.where(side == "bid", s.prev_bid_ex, ""))
    s = s[(s.t >= SESSION_START) & (s.t < SESSION_END)]
    return s


def nearest_offset(t0, tt, lo, hi):
    """For each t0, the offset t0 - (trade time) of the trade nearest to t0 inside [t0+lo, t0+hi];
    NaN where none."""
    out = np.full(len(t0), np.nan)
    if len(tt) == 0:
        return out
    a = np.searchsorted(tt, t0 + lo, side="left")
    b = np.searchsorted(tt, t0 + hi, side="right")
    has = b > a
    for i in np.where(has)[0]:
        c = tt[a[i]:b[i]]
        out[i] = t0[i] - c[np.argmin(np.abs(c - t0[i]))]
    return out


def match(states, trades, tcol, by_venue, window):
    """Offset of the nearest trade at the vanished price (and on the vanished quote's venue if
    by_venue) within the window; NaN where no trade."""
    t0 = states.t.to_numpy()
    off = np.full(len(states), np.nan)
    tr_p = np.rint(trades.price.to_numpy() / TICK).astype(np.int64)
    tr_t = trades[tcol].to_numpy()
    tr_ex = trades.ex.to_numpy()
    keys = list(zip(states.gone.to_numpy(), states.venue.to_numpy())) if by_venue else list(states.gone.to_numpy())
    for key in set(keys):
        sel = np.array([k == key for k in keys])
        if by_venue:
            m = (tr_p == key[0]) & (tr_ex == key[1])
        else:
            m = tr_p == key
        tt = np.sort(tr_t[m])
        off[sel] = nearest_offset(t0[sel], tt, *window)
    return off


def chance(states, trades, tcol, by_venue, window):
    t0 = states.t.to_numpy()
    tr_t = trades[tcol].to_numpy()
    if not by_venue:
        return float(~np.isnan(nearest_offset(t0, np.sort(tr_t), *window))).mean() if False else \
            float(np.mean(~np.isnan(nearest_offset(t0, np.sort(tr_t), *window))))
    out = np.zeros(len(states), dtype=bool)
    for v in set(states.venue):
        sel = states.venue.to_numpy() == v
        tt = np.sort(tr_t[trades.ex.to_numpy() == v])
        out[sel] = ~np.isnan(nearest_offset(t0[sel], tt, *window))
    return float(out.mean())


def one_day(nbbo, trades, tcol):
    st = states_with_venues(nbbo)
    tr = trades[trades.ex != "D"].sort_values(tcol)
    groups = {
        "brief positive-spread": (st.dur < 0.001) & (st.spread > 0) & (st.side != ""),
        "persistent move": (st.dur >= 0.001) & st.moved & (st.side != ""),
    }
    rows = []
    for name, g in groups.items():
        s = st[g]
        r = {"group": name, "n": len(s)}
        if len(s) == 0:
            rows.append(r); continue
        loose = match(s, tr, tcol, False, LOOSE)
        venue = match(s, tr, tcol, True, LOOSE)
        tight = match(s, tr, tcol, True, TIGHT)
        r["loose"] = float(np.mean(~np.isnan(loose)))
        r["same_venue"] = float(np.mean(~np.isnan(venue)))
        r["same_venue_tight"] = float(np.mean(~np.isnan(tight)))
        v = venue[~np.isnan(venue)] * 1e3
        r["offset_ms_q10"], r["offset_ms_q50"], r["offset_ms_q90"] = (np.percentile(v, [10, 50, 90]) if len(v) else (np.nan,) * 3)
        r["offset_in_0_100us"] = float(np.mean((v >= 0) & (v <= 0.1))) if len(v) else np.nan
        # where the trade is when it is not on the vanished quote's venue
        other = ~np.isnan(loose) & np.isnan(venue)
        r["other_venue_only"] = float(np.mean(other))
        r["no_trade"] = float(np.mean(np.isnan(loose)))
        o = loose[other] * 1e3
        r["other_offset_ms_q50"] = float(np.median(o)) if len(o) else np.nan
        r["other_offset_positive"] = float(np.mean(o > 0)) if len(o) else np.nan
        r["same_venue_exact0"] = float(np.mean(np.abs(v) < 1e-3)) if len(v) else np.nan
        # execution signature: a trade at the vanished price, on any venue, whose participant timestamp
        # equals the quote update's exchange timestamp (within 10 us; also within 100 us)
        r["exact_10us"] = float(np.mean(np.abs(loose) <= 10e-6))
        r["exact_100us"] = float(np.mean(np.abs(loose) <= 100e-6))
        r["chance_any_venue"] = chance(s, tr, tcol, False, LOOSE)
        r["chance_same_venue"] = chance(s, tr, tcol, True, LOOSE)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--venues", required=True)
    p.add_argument("--trades", required=True)
    p.add_argument("--syms", nargs="+", default=["INTC", "BAC", "MDLZ", "TGT"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for clock, tcol in (("ex", "t_ex"), ("sip", "t")):
        for sym in a.syms:
            parts = []
            for f in sorted(Path(a.venues, clock, sym).glob("*.csv.gz")):
                g = Path(a.trades, sym, f.name)
                if not g.exists():
                    continue
                parts.append(one_day(pd.read_csv(f), pd.read_csv(g), tcol))
            if not parts:
                print(clock, sym, "no days"); continue
            d = pd.concat(parts)
            for name, x in d.groupby("group", sort=False):
                x = x[x.n > 0]
                w = x.n.to_numpy()
                r = {"clock": clock, "sym": sym, "group": name, "days": len(x), "n": int(w.sum())}
                for c in ["loose", "same_venue", "same_venue_tight", "offset_in_0_100us", "other_venue_only", "no_trade",
                          "other_offset_positive", "same_venue_exact0", "exact_10us", "exact_100us", "chance_any_venue", "chance_same_venue"]:
                    r[c] = float(np.average(x[c], weights=w))
                for c in ["offset_ms_q10", "offset_ms_q50", "offset_ms_q90", "other_offset_ms_q50"]:
                    r[c] = float(np.median(x[c]))
                rows.append(r)
                print(f"{clock} {sym} {name}: n {r['n']}, loose {r['loose']:.3f}, same venue {r['same_venue']:.3f}, "
                      f"same venue tight {r['same_venue_tight']:.3f}, offset q50 {r['offset_ms_q50']:.3f} ms "
                      f"[q10 {r['offset_ms_q10']:.3f}, q90 {r['offset_ms_q90']:.3f}], in [0, 0.1] ms {r['offset_in_0_100us']:.2f}, "
                      f"exact 0 {r['same_venue_exact0']:.2f} | other venue only {r['other_venue_only']:.3f} (offset q50 {r['other_offset_ms_q50']:.3f} ms, "
                      f"positive {r['other_offset_positive']:.2f}), no trade {r['no_trade']:.3f} | exact timestamp (any venue) "
                      f"10us {r['exact_10us']:.3f}, 100us {r['exact_100us']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

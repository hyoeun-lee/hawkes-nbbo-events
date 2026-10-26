"""Lock durations by venue pair, in SIP time and in exchange time (revision item A18).

Reads the NBBO-change files with venue codes written by sql/14_build_lf_nbbo.py with
venues=True (lfnbbo_venues/{sip,ex}/SYM/date.csv.gz: t, bid, ask, bid_ex, ask_ex). A locked
state is one with bid >= ask; its duration runs to the next NBBO change; its pair is
(venue of the best bid, venue of the best ask). For each clock and stock: locks per day, median
and 90th-percentile duration, share over 1 ms, overall and by pair (pairs with at least 50 locks),
and the same for pairs whose two venues sit in the same data center.

Usage: python scripts/checks/lock_pairs.py <venues_dir> <out.csv> [SYMS...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 2016-17 data centers (TAQ exchange codes): Mahwah (NYSE group), Carteret (Nasdaq group),
# Secaucus (Cboe/BATS group and IEX's matching engine)
CENTER = {"N": "Mahwah", "P": "Mahwah", "A": "Mahwah",
          "Q": "Carteret", "T": "Carteret", "B": "Carteret", "X": "Carteret",
          "Z": "Secaucus", "Y": "Secaucus", "K": "Secaucus", "J": "Secaucus"}
# Venues not listed (C, M, V, W) sit elsewhere or their 2016-17 location is not certain here;
# pairs involving them count as neither same nor different data center.


def locks(df, start=36000.0, end=55800.0):
    df = df.sort_values("t").reset_index(drop=True)
    dur = np.r_[np.diff(df.t.to_numpy()), end - df.t.iloc[-1]]
    d = df.assign(dur=dur)
    d = d[(d.t >= start) & (d.t < end) & (d.bid >= d.ask)]
    known = [b in CENTER and a in CENTER for b, a in zip(d.bid_ex, d.ask_ex)]
    d = d.assign(pair=d.bid_ex.astype(str) + "/" + d.ask_ex.astype(str), known=known,
                 same_center=[k and CENTER[b] == CENTER[a] for k, b, a in zip(known, d.bid_ex, d.ask_ex)],
                 same_exchange=(d.bid_ex == d.ask_ex))
    return d


def summarize(d, n_days):
    ms = d.dur.to_numpy() * 1e3
    return {"locks_per_day": len(d) / max(n_days, 1), "median_ms": float(np.median(ms)) if len(ms) else np.nan,
            "p90_ms": float(np.percentile(ms, 90)) if len(ms) else np.nan,
            "share_over_1ms": float((ms > 1.0).mean()) if len(ms) else np.nan, "n": len(d)}


def main(venues_dir, out, syms):
    rows = []
    for clock in ("sip", "ex"):
        for sym in syms:
            files = sorted(Path(venues_dir, clock, sym).glob("*.csv.gz"))
            if not files:
                print(clock, sym, "no files"); continue
            d = pd.concat([locks(pd.read_csv(f)) for f in files], ignore_index=True)
            rows.append({"clock": clock, "sym": sym, "pair": "all", **summarize(d, len(files))})
            rows.append({"clock": clock, "sym": sym, "pair": "same data center", **summarize(d[d.same_center], len(files))})
            rows.append({"clock": clock, "sym": sym, "pair": "different data centers", **summarize(d[d.known & ~d.same_center], len(files))})
            rows.append({"clock": clock, "sym": sym, "pair": "same exchange", **summarize(d[d.same_exchange], len(files))})
            for pair, g in d.groupby("pair"):
                if len(g) >= 50:
                    rows.append({"clock": clock, "sym": sym, "pair": pair, **summarize(g, len(files))})
            print(f"{clock} {sym}: {len(files)} days, {len(d)} locks, median {np.median(d.dur) * 1e3:.2f} ms, "
                  f"same-center share {d.same_center.mean():.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(res[res.pair.isin(["all", "same data center", "different data centers", "same exchange"])].round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["INTC", "MDLZ", "BAC", "TGT"])

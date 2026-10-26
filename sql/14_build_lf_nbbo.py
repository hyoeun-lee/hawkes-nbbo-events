"""Latency-free NBBO (Holden, Pierson & Wu, 2023) from the per-exchange quote table, run on WRDS.

For each stock-day, every exchange's regular quotes (taqmsec.cqm_YYYYMMDD, qu_cond 'R') are
processed in one of two orders, each exchange's best bid and ask is tracked, and the best across
exchanges is taken:

  sip  ordered by the SIP timestamp (time_m): should reproduce complete_nbbo (validation);
  ex   ordered by the exchange's own timestamp (part_time): the latency-free NBBO.

A zero price means the exchange has no bid (ask); it is marked -inf (+inf) so forward filling
never brings back a withdrawn price. Output per stock-day: NBBO changes (t, bid, ask) from 10:00
to 15:30, in the same format as the complete_nbbo files, so the event pipeline runs unchanged.

Run in a WRDS Jupyter session where `db = wrds.Connection()` exists. Output stays on WRDS / local
disk (TAQ-derived data are not committed).
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd

SQL_QUOTES = """
select time_m, time_m_nano, part_time, part_time_nano, ex, bid, ask, qu_seqnum
from taqmsec.cqm_{date}
where sym_root = '{sym}' and sym_suffix is null
  and time_m between '09:30:00' and '15:30:00'
  and qu_cond = 'R'
"""


def secs(col, nano):
    t = pd.to_timedelta(col.astype(str)).dt.total_seconds().to_numpy()
    return t + nano.fillna(0).to_numpy() * 1e-9


def nbbo_changes(df, order, start=36000.0, end=55800.0):
    """NBBO from per-exchange quotes processed in the given time order."""
    d = df.sort_values([order, "qu_seqnum"], kind="stable").reset_index(drop=True)
    exs = np.array(sorted(d.ex.unique()))
    col = pd.Series(d.ex).map({e: i for i, e in enumerate(exs)}).to_numpy()
    n, m = len(d), len(exs)
    B = np.full((n, m), np.nan)
    A = np.full((n, m), np.nan)
    bid = d.bid.astype(float).to_numpy()
    ask = d.ask.astype(float).to_numpy()
    B[np.arange(n), col] = np.where(bid > 0, bid, -np.inf)
    A[np.arange(n), col] = np.where(ask > 0, ask, np.inf)
    B = pd.DataFrame(B).ffill().fillna(-np.inf).to_numpy()
    A = pd.DataFrame(A).ffill().fillna(np.inf).to_numpy()
    nbb, nbo = B.max(axis=1), A.min(axis=1)
    bex, aex = exs[B.argmax(axis=1)], exs[A.argmin(axis=1)]
    t = d[order].to_numpy()
    keep = np.ones(n, bool)
    keep[1:] = (nbb[1:] != nbb[:-1]) | (nbo[1:] != nbo[:-1])
    out = pd.DataFrame({"t": t[keep], "bid": nbb[keep], "ask": nbo[keep],
                        "bid_ex": bex[keep], "ask_ex": aex[keep]})
    out = out[np.isfinite(out.bid) & np.isfinite(out.ask)]
    first = out[out.t <= start].tail(1).assign(t=start)
    return pd.concat([first, out[(out.t > start) & (out.t <= end)]], ignore_index=True)


def price_changes_only(raw):
    """Keep quotes where that exchange's bid or ask changes (size-only updates do not move the NBBO)."""
    raw = raw.sort_values(["ex", "t_sip", "qu_seqnum"], kind="stable")
    pb = raw.groupby("ex").bid.shift()
    pa = raw.groupby("ex").ask.shift()
    return raw[(raw.bid != pb) | (raw.ask != pa)]


def save_day(db, sym, date, out="lfnbbo", venues=False):
    """venues=True also writes the venue codes of the best bid and ask (bid_ex, ask_ex) to
    <out>_venues/{sip,ex}/SYM/date.csv.gz, for lock durations by venue pair in both clocks
    (revision item A18). The event pipeline keeps reading the three-column files."""
    raw = db.raw_sql(SQL_QUOTES.format(sym=sym, date=date))
    raw = raw.assign(t_sip=secs(raw.time_m, raw.time_m_nano),
                     t_ex=secs(raw.part_time, raw.part_time_nano))
    raw = price_changes_only(raw)
    for name, order in [("sip", "t_sip"), ("ex", "t_ex")]:
        full = nbbo_changes(raw, order)
        ch = full[["t", "bid", "ask"]]
        p = Path(out) / name / sym / f"{date}.csv.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        ch.to_csv(p, index=False)
        if venues:
            q = Path(f"{out}_venues") / name / sym / f"{date}.csv.gz"
            q.parent.mkdir(parents=True, exist_ok=True)
            full[["t", "bid", "ask", "bid_ex", "ask_ex"]].to_csv(q, index=False)
    n = len(raw)
    del raw
    gc.collect()
    return n


if __name__ == "__main__":
    import wrds
    db = wrds.Connection()
    import sys
    # usage: python sql/14_build_lf_nbbo.py [--venues] [YYYYMM] [SYM ...]
    # (default: May 2017, the four simulation stocks; --venues also writes lfnbbo_venues/ for A18)
    venues = "--venues" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--venues"]
    month = argv[0] if argv else "201705"
    syms = argv[1:] or ["INTC", "MDLZ", "BAC", "TGT"]
    y, m = int(month[:4]), int(month[4:])
    skip = {"20170529", "20171123", "20171124"}                      # Memorial Day; Thanksgiving and its half day
    dates = [d.strftime("%Y%m%d") for d in pd.bdate_range(f"{y}-{m:02d}-01", pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0))
             if d.strftime("%Y%m%d") not in skip]
    for sym in syms:
        for date in dates:
            print(sym, date, save_day(db, sym, date, venues=venues), flush=True)

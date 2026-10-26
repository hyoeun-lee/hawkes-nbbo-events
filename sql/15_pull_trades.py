"""Pull trades for the stock-days of one month from WRDS Millisecond TAQ (taqmsec.ctm_YYYYMMDD).

Used for the depletion-against-flicker check (results §24): are sub-millisecond positive-spread
NBBO states preceded by a trade at the price that disappeared? Output per stock-day:
trades/<SYM>/<date>.csv.gz with t (SIP time, seconds after midnight, with nanoseconds when the
table has them), t_ex (exchange / participant time, when present), ex, price, size, cond.
Regular-way trades only (tr_corr = '00'); 09:55-15:30 so that trades just before 10:00 are there.

Run in a WRDS Jupyter session:  python sql/15_pull_trades.py 201705 INTC MU MDLZ BAC WFC T TGT ATVI
Output stays on WRDS / local disk (TAQ-derived data are not committed).
"""

import sys
from pathlib import Path

import pandas as pd

FULL = """
select time_m, time_m_nano, part_time, part_time_nano, ex, price, size, tr_scond
from taqmsec.ctm_{date}
where sym_root = '{sym}' and sym_suffix is null
  and time_m between '09:55:00' and '15:30:00'
  and tr_corr = '00'
"""
BASIC = """
select time_m, ex, price, size, tr_scond
from taqmsec.ctm_{date}
where sym_root = '{sym}' and sym_suffix is null
  and time_m between '09:55:00' and '15:30:00'
  and tr_corr = '00'
"""


def secs(col, nano=None):
    t = pd.to_timedelta(col.astype(str)).dt.total_seconds().to_numpy()
    if nano is not None:
        t = t + nano.fillna(0).to_numpy() * 1e-9
    return t


def pull_day(db, sym, date, out="trades"):
    p = Path(out) / sym / f"{date}.csv.gz"
    if p.exists():
        return "exists"
    try:
        raw = db.raw_sql(FULL.format(date=date, sym=sym))
        t = secs(raw.time_m, raw.time_m_nano)
        t_ex = secs(raw.part_time, raw.part_time_nano) if raw.part_time.notna().any() else None
    except Exception as e:  # older layout without nanosecond / participant columns
        print("  full layout failed:", str(e).splitlines()[0], "-> basic columns")
        raw = db.raw_sql(BASIC.format(date=date, sym=sym))
        t, t_ex = secs(raw.time_m), None
    df = pd.DataFrame({"t": t, "t_ex": t_ex if t_ex is not None else float("nan"), "ex": raw.ex,
                       "price": raw.price, "size": raw["size"], "cond": raw.tr_scond.fillna("")})
    df = df.sort_values("t", kind="stable")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return len(df)


def month_dates(month):
    y, m = int(month[:4]), int(month[4:])
    days = pd.bdate_range(f"{y}-{m:02d}-01", pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0))
    skip = {"20170529", "20170704", "20170904", "20171123", "20171124",   # holidays and the half day
            "20161124", "20161125", "20161226", "20170102", "20170116", "20170220", "20170414", "20171225"}
    return [d.strftime("%Y%m%d") for d in days if d.strftime("%Y%m%d") not in skip]


if __name__ == "__main__":
    import wrds
    db = wrds.Connection()
    month = sys.argv[1] if len(sys.argv) > 1 else "201705"
    syms = sys.argv[2:] or ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]
    print(db.raw_sql(f"select * from taqmsec.ctm_{month_dates(month)[0]} limit 2").columns.tolist())
    for sym in syms:
        for date in month_dates(month):
            print(sym, date, pull_day(db, sym, date), flush=True)

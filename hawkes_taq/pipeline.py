"""Build and save event series for many stock-days."""

from pathlib import Path

import pandas as pd

from .events import (build_states, events_e0, events_e1, events_e2, qc_summary,
                     spread_ties)
from .fetch import fetch_nbbo_changes

# Early-close sessions (13:00 close) in the sample. The 10:00-15:30 window would include 2.5 hours
# of after-hours quotes (wide, stale spreads), so these days are excluded from all fits.
HALF_DAYS = frozenset({"20161125", "20170703", "20171124"})

SAMPLE_STOCKS = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T",  # core, large-tick
                 "TGT", "ATVI"]                              # contrast


def _seconds(hhmmss):
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return 3600 * h + 60 * m + s


def build_stock_day(db, sym, date, start="10:00:00", end="15:30:00",
                    max_nonpos=0.005, stamp="exit", min_state=0.0):
    """Fetch one stock-day and return ({'E0': df, 'E1': df, 'E2': df}, qc_row).

    Each event table has the raw time t (seconds after midnight) and t_fit, the time with
    ties spread within one microsecond, for use in likelihoods.
    """
    nbbo = fetch_nbbo_changes(db, sym, date, start, end)
    states = build_states(nbbo, end_time=_seconds(end))
    e0, e1, e2 = (events_e0(states), events_e1(states, max_nonpos, stamp, min_state),
                  events_e2(states))
    for e in (e0, e1, e2):
        e.insert(1, "t_fit", spread_ties(e["t"]))
    qc = {"sym": sym, "date": date, **qc_summary(nbbo, states, e0, e1, e2, max_nonpos)}
    return {"E0": e0, "E1": e1, "E2": e2}, qc


def run(db, syms, dates, out_dir="data/events", **kwargs):
    """Build events for every stock and date; save CSVs and return the QC table.

    Files go to out_dir/<E>/<sym>/<date>.csv.gz, and the QC table to out_dir/qc.csv.
    Dates without a TAQ table (holidays, weekends) are recorded in the QC table and skipped.
    """
    out = Path(out_dir)
    rows = []
    for date in dates:
        for sym in syms:
            try:
                events, qc = build_stock_day(db, sym, date, **kwargs)
            except Exception as e:  # missing table, empty day, connection error
                rows.append({"sym": sym, "date": date, "error": str(e).splitlines()[0][:120]})
                continue
            for name, df in events.items():
                path = out / name / sym / f"{date}.csv.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
            rows.append(qc)
    qc_table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    qc_table.to_csv(out / "qc.csv", index=False)
    return qc_table


def weekdays(start, end):
    """Weekday dates as 'YYYYMMDD' strings; holidays are skipped later by run()."""
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(start, end)]

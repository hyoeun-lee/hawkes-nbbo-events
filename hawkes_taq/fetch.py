"""Pull NBBO price changes for one stock-day from WRDS Millisecond TAQ."""

import datetime as dt

import numpy as np
import pandas as pd

# NBBO price changes only (rows where best bid or best ask differs from the previous row),
# from the completed NBBO table, which includes quotes missing from nbbom.
# complete_nbbo has no sequence number, so rows sharing a timestamp come back in arbitrary
# order; build_states counts them so they can be checked.
NBBO_CHANGES_SQL = """
with q as (
    select time_m, best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.complete_nbbo_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '{start}' and '{end}'
      and best_bid > 0 and best_ask > 0
    window w as (order by time_m)
)
select time_m, best_bid, best_ask
from q
where pb is null or best_bid <> pb or best_ask <> pa
order by time_m
"""


def _to_seconds(col):
    """Seconds after midnight from a column of datetime.time or timedelta values."""
    if pd.api.types.is_timedelta64_dtype(col):
        return col.dt.total_seconds().to_numpy()
    return np.array([
        v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6 if isinstance(v, dt.time)
        else pd.Timedelta(v).total_seconds()
        for v in col
    ])


def fetch_nbbo_changes(db, sym, date, start="10:00:00", end="15:30:00"):
    """Return a DataFrame with columns t (seconds after midnight), bid, ask.

    db is an open wrds.Connection; date is 'YYYYMMDD'.
    """
    raw = db.raw_sql(NBBO_CHANGES_SQL.format(sym=sym, date=date, start=start, end=end))
    return pd.DataFrame({
        "t": _to_seconds(raw["time_m"]),
        "bid": raw["best_bid"].astype(float).to_numpy(),
        "ask": raw["best_ask"].astype(float).to_numpy(),
    })

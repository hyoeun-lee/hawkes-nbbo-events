# WRDS queries

Queries behind [`../docs/data_checks.md`](../docs/data_checks.md). TAQ files are templates with `{date}` (`YYYYMMDD`) and `{sym}` (ticker root) placeholders, for the WRDS Millisecond TAQ tables `taqmsec.nbbom_{date}` and `taqmsec.complete_nbbo_{date}`. `10_crsp_candidates.sql` has no placeholders.

| File | What it measures |
|---|---|
| `01_timestamp_resolution.sql` | Sub-millisecond and nanosecond content of the timestamps (one minute) |
| `02_sip_lag_and_gaps.sql` | NBBO change count, SIP-minus-exchange lag, gaps between changes |
| `03_format_break.sql` | Rows and NBBO changes around the late-October 2016 format change |
| `04_daily_counts.sql` | Daily NBBO change count and spread state after each change |
| `05_lag_by_activity.sql` | SIP lag grouped by recent activity |
| `06_lag_by_venue.sql` | SIP lag by venue |
| `07_spread_after_change.sql` | Spread (ticks) after each change; zero or null quote check |
| `08_state_durations.sql` | Duration of 1-tick, locked, 2+ tick and crossed states |
| `09_lock_duration_by_venue_pair.sql` | Lock duration by the venues at the best bid and best ask |
| `10_crsp_candidates.sql` | Stock selection, stage 1: liquid stocks priced $10–60 in 2017 (CRSP) |
| `11_tick_profile.sql` | Stock selection, stage 2: NBBO changes, share of time at one tick, lock share (`complete_nbbo`) |
| `12_same_exchange_locks.sql` | Locked/crossed NBBO states where one exchange holds both sides, against all locks |
| `13_lf_nbbo_feasibility.sql` | Quote-table (`cqm`) columns and one stock-day's size and exchange-timestamp coverage (`part_time`, `part_time_nano`) |
| `14_build_lf_nbbo.py` | Python, run on WRDS: NBBO rebuilt from `cqm` regular quotes in SIP-time and exchange-time order (exchange-timestamp reconstruction); month and stocks as arguments (default May 2017, four simulation stocks) (`data_checks.md` §15) |
| `15_pull_trades.py` | Python, run on WRDS: regular-way trades (`ctm`) for one month's stock-days, for the depletion-against-flicker check (`scripts/checks/trades_match.py`) |

## Running

```python
import wrds
from pathlib import Path

db = wrds.Connection()

def run(name, date, sym='INTC'):
    sql = Path(f'sql/{name}.sql').read_text().format(date=date, sym=sym)
    return db.raw_sql(sql)

print(run('08_state_durations', '20170517').to_string())
```

Notes:

- Use `mod()` in SQL, not `%`. psycopg2 treats `%` as a parameter placeholder.
- Postgres arrays (from `percentile_cont(array[...])`) come back as text, so print them directly.
- `time_m_nano` and `part_time` don't exist in older tables (see the data checks), so queries using them fail before 2016.
- A full trading day for one stock runs in well under a second; 61 days of `04_daily_counts.sql` took about 35 seconds.

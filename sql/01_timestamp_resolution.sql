-- Share of rows with a sub-millisecond part in time_m, and with a nonzero nanosecond remainder.
-- One minute of one stock. time_m_nano exists only from 2016 on; drop that line for earlier dates.
-- Use mod() rather than the percent operator, which psycopg2 treats as a parameter placeholder.
select count(*) as n_rows,
       sum((mod(extract(microseconds from time_m)::bigint, 1000) <> 0)::int) as n_subms,
       sum((time_m_nano <> 0)::int) as n_nano_nonzero
from taqmsec.nbbom_{date}
where sym_root = '{sym}' and sym_suffix is null
  and time_m between '10:00:00' and '10:01:00';

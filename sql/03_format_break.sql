-- Rows, NBBO changes and nonzero nanosecond remainders in 10:00-10:30 (format-break check, late Oct 2016).
select count(*) as n_rows,
       sum(case when best_bid <> pb or best_ask <> pa then 1 else 0 end) as n_nbbo_changes,
       sum((time_m_nano <> 0)::int) as n_nano_nonzero
from (
    select time_m, time_m_nano, best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.nbbom_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '10:30:00'
    window w as (order by time_m, qu_seqnum)
) t;

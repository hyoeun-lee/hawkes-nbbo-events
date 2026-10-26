-- NBBO price changes: count, SIP-minus-exchange lag quantiles, and gaps between changes (ms).
with q as (
    select time_m, part_time, best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.nbbom_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '15:30:00'
    window w as (order by time_m, qu_seqnum)
),
c as (
    select time_m, part_time,
           lag(time_m) over (order by time_m) as prev_t
    from q
    where best_bid <> pb or best_ask <> pa
)
select count(*) as n_events,
       percentile_cont(array[0.05,0.25,0.5,0.75,0.95]) within group
         (order by extract(epoch from (time_m - part_time)) * 1000) as sip_delay_ms,
       percentile_cont(array[0.05,0.25,0.5,0.75,0.95]) within group
         (order by extract(epoch from (time_m - prev_t)) * 1000) as gap_ms
from c;

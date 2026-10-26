-- SIP lag at each NBBO change, grouped by the gap between the two preceding changes.
-- The preceding gap is used because the gap ending at event i contains event i's own lag.
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
           lag(time_m, 1) over (order by time_m) as t1,
           lag(time_m, 2) over (order by time_m) as t2
    from q
    where (best_bid <> pb or best_ask <> pa) and part_time is not null
),
b as (
    select extract(epoch from (time_m - part_time)) * 1000 as lag_ms,
           extract(epoch from (t1 - t2)) * 1000           as prior_gap_ms
    from c
    where t2 is not null
)
select case when prior_gap_ms < 0.5  then '1: <0.5 ms'
            when prior_gap_ms < 5    then '2: 0.5-5 ms'
            when prior_gap_ms < 1000 then '3: 5 ms-1 s'
            else                          '4: >1 s' end as activity,
       count(*) as n,
       percentile_cont(0.5) within group (order by lag_ms) as lag_median_ms,
       percentile_cont(0.9) within group (order by lag_ms) as lag_p90_ms,
       avg(lag_ms) as lag_mean_ms
from b
group by 1
order by 1;

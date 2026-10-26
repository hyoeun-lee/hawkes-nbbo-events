-- How long the NBBO stays in each spread state after a change (ms).
with q as (
    select time_m, best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.nbbom_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '15:30:00'
    window w as (order by time_m, qu_seqnum)
),
c as (
    select time_m, round((best_ask - best_bid) * 100) as spread_ticks,
           lead(time_m) over (order by time_m) as next_t
    from q
    where best_bid <> pb or best_ask <> pa
)
select case when spread_ticks < 0 then 'crossed'
            when spread_ticks = 0 then 'locked'
            when spread_ticks = 1 then '1 tick'
            else '2+ ticks' end as state,
       count(*) as n,
       percentile_cont(array[0.1, 0.5, 0.9]) within group
         (order by extract(epoch from (next_t - time_m)) * 1000) as duration_ms_p10_p50_p90
from c
where next_t is not null
group by 1
order by n desc;

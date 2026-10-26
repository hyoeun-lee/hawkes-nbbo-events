-- Stage 2 of stock selection: NBBO changes, time-weighted share of time at a one-tick spread,
-- and number of changes that lock or cross, from the completed NBBO table.
with q as (
    select time_m, best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.complete_nbbo_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '15:30:00'
      and best_bid > 0 and best_ask > 0
    window w as (order by time_m)
),
c as (
    select time_m, round((best_ask - best_bid) * 100) as sp_ticks,
           lead(time_m) over (order by time_m) as next_t
    from q
    where pb is null or best_bid <> pb or best_ask <> pa
)
select count(*) as n_changes,
       sum(extract(epoch from (next_t - time_m))) as total_s,
       sum(case when sp_ticks = 1 then extract(epoch from (next_t - time_m)) else 0 end) as s_1tick,
       sum(case when sp_ticks <= 0 then 1 else 0 end) as n_lock_cross
from c
where next_t is not null;

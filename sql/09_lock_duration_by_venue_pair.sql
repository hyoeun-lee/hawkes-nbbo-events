-- Median duration of locked NBBO states, by the venues at the best bid and best ask.
with q as (
    select time_m, best_bid, best_ask, best_bidex, best_askex,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.nbbom_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '15:30:00'
    window w as (order by time_m, qu_seqnum)
),
c as (
    select time_m, best_bid, best_ask, best_bidex, best_askex,
           lead(time_m) over (order by time_m) as next_t
    from q
    where best_bid <> pb or best_ask <> pa
)
select best_bidex as bid_venue, best_askex as ask_venue, count(*) as n_locks,
       percentile_cont(0.5) within group
         (order by extract(epoch from (next_t - time_m)) * 1000) as median_lock_ms
from c
where best_bid = best_ask and next_t is not null
group by 1, 2
order by n_locks desc
limit 15;

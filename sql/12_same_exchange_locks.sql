-- Locked or crossed NBBO states where the same exchange holds the best bid and the best ask,
-- against all locked or crossed states (Favreau & Garvey, 2022, drop these as data errors).
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
    where pb is null or best_bid <> pb or best_ask <> pa
)
select (best_bidex = best_askex) as same_exchange,
       case when best_bid = best_ask then 'locked' else 'crossed' end as state,
       count(*) as n,
       percentile_cont(0.5) within group
         (order by extract(epoch from (next_t - time_m)) * 1000) as median_ms
from c
where best_bid >= best_ask and next_t is not null
group by 1, 2
order by 1, 2;

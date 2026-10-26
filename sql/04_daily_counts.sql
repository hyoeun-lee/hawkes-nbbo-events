-- Per-day NBBO change count and the spread state after each change.
select count(*) as n_changes,
       sum((round((best_ask - best_bid) * 100) = 1)::int) as n_1tick,
       sum((best_ask = best_bid)::int) as n_locked,
       sum((best_ask < best_bid)::int) as n_crossed,
       avg((best_ask + best_bid) / 2) as avg_mid
from (
    select best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.nbbom_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '15:30:00'
    window w as (order by time_m, qu_seqnum)
) t
where best_bid <> pb or best_ask <> pa;

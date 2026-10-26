-- Distribution of the spread (in ticks) after each NBBO change, with a check for zero or null quotes.
with q as (
    select best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.nbbom_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '15:30:00'
    window w as (order by time_m, qu_seqnum)
)
select round((best_ask - best_bid) * 100) as spread_ticks,
       count(*) as n_events,
       sum(case when best_bid = 0 or best_ask = 0 or best_bid is null or best_ask is null then 1 else 0 end) as n_zero_or_null
from q
where best_bid <> pb or best_ask <> pa
group by 1
order by n_events desc
limit 15;

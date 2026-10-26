-- SIP lag at NBBO changes, by the venue (ex) of the quote that triggered the row.
with q as (
    select time_m, part_time, ex, best_bid, best_ask,
           lag(best_bid) over w as pb, lag(best_ask) over w as pa
    from taqmsec.nbbom_{date}
    where sym_root = '{sym}' and sym_suffix is null
      and time_m between '10:00:00' and '15:30:00'
    window w as (order by time_m, qu_seqnum)
)
select ex, count(*) as n,
       percentile_cont(0.1) within group (order by extract(epoch from (time_m - part_time)) * 1000) as lag_p10_ms,
       percentile_cont(0.5) within group (order by extract(epoch from (time_m - part_time)) * 1000) as lag_median_ms,
       percentile_cont(0.9) within group (order by extract(epoch from (time_m - part_time)) * 1000) as lag_p90_ms
from q
where (best_bid <> pb or best_ask <> pa) and part_time is not null
group by ex
order by n desc;

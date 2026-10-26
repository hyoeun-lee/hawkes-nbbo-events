-- Feasibility of a latency-free NBBO (Holden, Pierson & Wu, 2023; Battalio et al., 2026):
-- does the per-exchange quote table carry exchange (participant) timestamps, and how large is
-- one stock-day? Run the two statements separately.

-- (a) columns of the quote table for one day
select column_name, data_type
from information_schema.columns
where table_schema = 'taqmsec' and table_name = 'cqm_{date}'
order by ordinal_position;

-- (b) size and timestamp coverage for one stock-day, 10:00-15:30
select count(*) as n_quotes,
       count(distinct ex) as n_exchanges,
       sum(case when participant_timestamp is null then 1 else 0 end) as n_missing_part_time
from taqmsec.cqm_{date}
where sym_root = '{sym}' and sym_suffix is null
  and time_m between '10:00:00' and '15:30:00';

-- Stage 1 of stock selection: liquid US common stocks priced $10-60 in 2017, ranked by average daily dollar volume.
-- If crsp.dsf is not permitted, use crsp_a_stock.dsf and crsp_a_stock.dsenames.
select n.ticker, d.permno,
       avg(abs(d.prc))         as avg_price,
       avg(abs(d.prc) * d.vol) as avg_dollar_vol,
       count(*)                as n_days
from crsp.dsf d
join crsp.dsenames n
  on d.permno = n.permno
 and d.date between n.namedt and coalesce(n.nameendt, '9999-12-31')
where d.date between '2017-01-01' and '2017-12-31'
  and n.shrcd in (10, 11)
  and n.exchcd in (1, 2, 3)
group by n.ticker, d.permno
having avg(abs(d.prc)) between 10 and 60 and count(*) > 240
order by avg_dollar_vol desc
limit 40;

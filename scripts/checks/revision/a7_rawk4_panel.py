"""Raw K = 4 panel (cluster job 11424604) and block bootstraps (11424605, 11424637): checks and table inputs."""
import glob
import numpy as np, pandas as pd
S = os.environ.get("HAWKES_DATA", "DATA")   # root of the NBBO, trade and intermediate files (see REPLICATION.md)
C = f'{S}/cluster4'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
order = ['INTC', 'MU', 'MDLZ', 'BAC', 'WFC', 'T', 'TGT', 'ATVI']

def load(f, delta_col=True):
    d = pd.read_csv(f, dtype={'month': str})
    d['mem'] = d.memory_s.str.split('; ').apply(lambda v: np.array(v, float))
    d['exc'] = d.excitation_per_scale.str.split('; ').apply(lambda v: np.array(v, float))
    if 'delta' not in d: d['delta'] = 0.0
    return d

def mass(row, H, lo=None):
    lo = row.delta if lo is None else lo
    return float(np.sum(row.exc * (np.exp(-lo / row.mem) - np.exp(-H / row.mem))))

def add_masses(d):
    d['n1'] = d.apply(lambda r: mass(r, 1.0), axis=1)
    d['n10'] = d.apply(lambda r: mass(r, 10.0), axis=1)
    d['n1ms'] = d.apply(lambda r: mass(r, 0.001, 0.0), axis=1)          # mass below 1 ms from zero
    d['n1_from1ms'] = d.apply(lambda r: mass(r, 1.0, 0.001), axis=1)    # (1 ms, 1 s]
    d['n10_from1ms'] = d.apply(lambda r: mass(r, 10.0, 0.001), axis=1)
    d['n1_from5ms'] = d.apply(lambda r: mass(r, 1.0, 0.005), axis=1)
    d['n10_from5ms'] = d.apply(lambda r: mass(r, 10.0, 0.005), axis=1)
    d['eff_br'] = d.apply(lambda r: float(np.sum(r.exc * np.exp(-r.delta / r.mem))), axis=1)
    d['fast_mem'] = d.mem.apply(lambda m: m.min()); d['slow_mem'] = d.mem.apply(lambda m: m.max())
    d['epd'] = d.n_events / d.n_days
    return d

raw4 = add_masses(load(f'{C}/fits_e0_k4_panel.csv'))
raw3 = add_masses(load(str(Path(__file__).resolve().parents[3]) + '/results/fits_e0_K3_panel.csv'))
k4 = add_masses(load(f'{S}/delta/K4.csv')); h2k3 = add_masses(load(f'{S}/delta/h2_K3.csv'))
clean = pd.concat([k4[k4.sym != 'BAC'], h2k3[h2k3.sym == 'BAC']]).sort_values(['sym', 'month'])
old4 = add_masses(pd.concat([load(f) for f in glob.glob(f'{S}/e0k4/fits_e0_k4_*.csv')]))
print('raw4', len(raw4), 'raw3', len(raw3), 'clean', len(clean), 'old raw4', len(old4))
print('raw4 starts:', raw4.starts.unique(), '| minutes median/max', raw4.minutes.median(), raw4.minutes.max())

# --- 1. optimum checks
j = raw4.set_index(['sym', 'month']).join(raw3.set_index(['sym', 'month'])[['loglik', 'bic', 'n1', 'n10', 'branching_ratio', 'slow_mem']], rsuffix='_3')
j['dll'] = j.loglik - j.loglik_3; j['dbic'] = j.bic - j.bic_3
print('\n[1] K4 vs K3 loglik gain: min', round(j.dll.min(), 1), 'median', round(j.dll.median(), 1), '| K4 below K3 in', int((j.dll < 0).sum()), 'stock-months | BIC prefers K4 in', int((j.dbic < 0).sum()), 'of', len(j))
jo = raw4.set_index(['sym', 'month']).join(old4.set_index(['sym', 'month'])[['loglik', 'n1', 'n10', 'branching_ratio', 'slow_mem', 'n1ms']], rsuffix='_old', how='inner')
jo['dll'] = jo.loglik - jo.loglik_old
print('[1b] new vs old raw K4 (May, Nov): n', len(jo), '| dll min', round(jo.dll.min(), 1), 'median', round(jo.dll.median(), 1), 'max', round(jo.dll.max(), 1), '| worse in', int((jo.dll < -0.5).sum()))
print(jo[['dll', 'n1', 'n1_old', 'n10', 'n10_old', 'branching_ratio', 'branching_ratio_old', 'slow_mem', 'slow_mem_old']].round(3).to_string())

# --- 2. horizon table (tab:horizon) inputs
m = raw4.set_index(['sym', 'month']).join(clean.set_index(['sym', 'month'])[['n1', 'n10', 'n1_from5ms', 'n10_from5ms', 'eff_br', 'branching_ratio', 'n_events', 'epd', 'delta']], rsuffix='_c')
m['r1'] = m.n1 / m.n1_c; m['r10'] = m.n10 / m.n10_c; m['rbr'] = m.branching_ratio / m.branching_ratio_c
m['r1_5ms'] = m.n1_from5ms / m.n1_from5ms_c; m['r10_5ms'] = m.n10_from5ms / m.n10_from5ms_c
q = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}--{s.quantile(.75):.2f}]"
q3 = lambda s: f"{s.median():.3f} [{s.quantile(.25):.3f}--{s.quantile(.75):.3f}]"
print('\n[2] tab:horizon, 112 stock-months (raw K=4; clean K=4, BAC h=2ms K=3)')
print('  Clean  n1', q(m.n1_c), ' n10', q(m.n10_c), ' total', q(m.branching_ratio_c), ' eff', q(m.eff_br))
print('  Raw    n1', q(m.n1), ' n10', q(m.n10), ' total', q(m.branching_ratio))
print('  Raw first ms', q(m.n1ms), ' raw (1ms,1s]', q(m.n1_from1ms), ' clean n1 (from h)', q(m.n1_c))
print('  Raw/clean n1', q(m.r1), 'range', round(m.r1.min(), 2), round(m.r1.max(), 2), '| n10', q(m.r10), 'range', round(m.r10.min(), 2), round(m.r10.max(), 2), '| total', q(m.rbr))
print('  stock-months with ratio>1: n1', int((m.r1 > 1).sum()), 'n10', int((m.r10 > 1).sum()))
print('  Common window (5 ms, H]: raw/clean n1', q(m.r1_5ms), ' n10', q(m.r10_5ms), '| raw (5ms,1s]', q3(m.n1_from5ms), 'clean', q3(m.n1_from5ms_c))
print('  Clean n1 by stock (median):', m.groupby(level='sym').n1_c.median().reindex(order).round(3).to_dict())
print('  Raw n1 by stock (median):', m.groupby(level='sym').n1.median().reindex(order).round(3).to_dict())
print('  Raw/clean n1 by stock (median):', m.groupby(level='sym').r1.median().reindex(order).round(2).to_dict())
print('  Raw fast memory (ms) median by stock:', (m.groupby(level='sym').fast_mem.median().reindex(order) * 1e3).round(3).to_dict())
print('  Raw slow memory (s) median by stock:', m.groupby(level='sym').slow_mem.median().reindex(order).round(1).to_dict(), '| clean:', m.groupby(level='sym').apply(lambda x: x.name and clean[clean.sym == x.name].slow_mem.median()).reindex(order).round(1).to_dict())
print('  Raw sub-ms mass by stock (median):', m.groupby(level='sym').n1ms.median().reindex(order).round(3).to_dict())
print('  Raw components under 5 ms: count per stock-month', raw4.mem.apply(lambda v: int((v < 0.005).sum())).value_counts().to_dict())

# --- 3. identity n0 = s + n_m N_m/N_0 with equal K
g = pd.read_csv(f'{S}/gapshares_monthly.csv', dtype={'month': str}).set_index(['sym', 'month'])
m = m.join(g[['gap_lt1ms_E0']])
m['pred'] = m.gap_lt1ms_E0 + m.branching_ratio_c * m.n_events_c / m.n_events
m['err'] = m.branching_ratio - m.pred
print('\n[3] identity, 112 stock-months, equal K: median error', round(m.err.median(), 3), 'MAE', round(m.err.abs().median(), 3), '| by stock median err', m.groupby(level='sym').err.median().reindex(order).round(3).to_dict())
print('  s median', round(m.gap_lt1ms_E0.median(), 3), ' N_m/N_0 median', round((m.n_events_c / m.n_events).median(), 3), '| raw BR range', round(m.branching_ratio.min(), 2), round(m.branching_ratio.max(), 2), '| clean BR range', round(m.branching_ratio_c.min(), 2), round(m.branching_ratio_c.max(), 2), 'clean eff range', round(m.eff_br.min(), 2), round(m.eff_br.max(), 2))
m['imm_raw'] = m.n_events * (1 - m.branching_ratio) / m.n_days; m['imm_clean'] = m.n_events_c * (1 - m.branching_ratio_c) / m.n_days
m['imm_ratio'] = m.imm_raw / m.imm_clean
print('  immigrants/day raw over clean: median', round(m.imm_ratio.median(), 3), '| by stock', m.groupby(level='sym').imm_ratio.median().reindex(order).round(2).to_dict())
# activity relation on raw: corr of raw n1 with log activity within stock
raw4['lepd'] = np.log(raw4.epd)
cw = []
for s in order:
    x = raw4[raw4.sym == s]; cw.append(np.corrcoef(x.lepd, x.n1)[0, 1])
print('  raw n1 vs log activity within stock, corr by stock:', dict(zip(order, np.round(cw, 2))), 'median', round(float(np.median(cw)), 2))
m.reset_index().to_csv(f'{S}/a7/horizon_k4.csv', index=False)

# --- 4. block bootstrap panel: SD of clean n1, n10 by stock-month
bb = pd.concat([pd.read_csv(f, dtype={'month': str}) for f in sorted(glob.glob(f'{C}/blockboot_panel/*.csv'))])
sd = bb.groupby(['sym', 'month']).agg(sd_n1=('clean_n1', 'std'), sd_n10=('clean_n10', 'std'), sd_br=('clean_br', 'std'), med_n1=('clean_n1', 'median'), n=('rep', 'size'), h=('h', 'first'), K=('K_clean', 'first'))
print('\n[4] block bootstrap panel: reps per stock-month', sd.n.min(), sd.n.max(), '| h by stock', sd.groupby(level='sym').h.first().to_dict())
pt = clean.set_index(['sym', 'month'])[['n1', 'n10', 'eff_br']].join(sd)
pt['bias_n1'] = pt.med_n1 - pt.n1
print('  SD of clean n1 by stock: median [min--max] over months')
for s in order:
    x = pt.loc[s]
    print(f"   {s:5s} n1 {x.n1.median():.3f} sd {x.sd_n1.median():.4f} [{x.sd_n1.min():.4f}--{x.sd_n1.max():.4f}] | n10 {x.n10.median():.3f} sd {x.sd_n10.median():.4f} [{x.sd_n10.min():.4f}--{x.sd_n10.max():.4f}] | eff br {x.eff_br.median():.3f} sd {x.sd_br.median():.3f} | block median minus point n1: {x.bias_n1.median():+.4f}")
print('  relative SD n1 (sd/point) median across 112:', round((pt.sd_n1 / pt.n1).median(), 3), 'max', round((pt.sd_n1 / pt.n1).max(), 3))
pt.reset_index().to_csv(f'{S}/a7/blockboot_panel_sd.csv', index=False)

# --- 5. May block lengths 1, 5, 10
print('\n[5] May 2017 paired bootstrap by block length (clean sd n1 / raw sd n1 / ratio n1 median [2.5, 97.5])')
rows = []
for b in (1, 5, 10):
    d = pd.concat([pd.read_csv(f, dtype={'month': str}) for f in sorted(glob.glob(f'{C}/blockboot_b{b}/*.csv'))])
    for s in order:
        x = d[d.sym == s]
        rows.append(dict(block=b, sym=s, n=len(x), sd_clean_n1=x.clean_n1.std(), sd_clean_n10=x.clean_n10.std(), sd_raw_n1=x.raw_n1.std(),
                         ratio1_med=x.ratio_n1.median(), ratio1_lo=x.ratio_n1.quantile(.025), ratio1_hi=x.ratio_n1.quantile(.975),
                         ratio10_med=x.ratio_n10.median(), ratio10_lo=x.ratio_n10.quantile(.025), ratio10_hi=x.ratio_n10.quantile(.975)))
bl = pd.DataFrame(rows); bl.to_csv(f'{S}/a7/blockboot_may_blocks.csv', index=False)
for s in order:
    x = bl[bl.sym == s].set_index('block')
    pe = m.loc[(s, '201705')]
    print(f"   {s:5s} point ratio n1 {pe.r1:.2f} n10 {pe.r10:.2f} | " + ' | '.join(f"b{b}: sd {x.loc[b].sd_clean_n1:.4f}/{x.loc[b].sd_raw_n1:.4f} r1 {x.loc[b].ratio1_med:.2f} [{x.loc[b].ratio1_lo:.2f},{x.loc[b].ratio1_hi:.2f}] r10 {x.loc[b].ratio10_med:.2f} [{x.loc[b].ratio10_lo:.2f},{x.loc[b].ratio10_hi:.2f}]" for b in (1, 5, 10)))

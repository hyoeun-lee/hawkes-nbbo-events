"""Lee (2023) replication: IBM and AAPL, January 2017, event statistics and fits under his specification
(one day at a time, constant baseline, K = 2 or 3) and ours (the month, 30-minute bins), on raw (E0) and
clean (E1m) events; the attribution of events to each kernel as the average of lambda_k / lambda over
events (his Table 3). Writes lee2023_replication_stats.csv and lee2023_replication_fits.csv (Table 7).

    python scripts/checks/lee2023_replication.py --nbbo DATA/nbbo_ibm --out results/fits
"""
import argparse
import sys, math
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from hawkes_taq import events as ev, hawkes as hk
from fit_panel import build_days

_p = argparse.ArgumentParser(); _p.add_argument('--nbbo', required=True); _p.add_argument('--out', required=True)
_a = _p.parse_args() if __name__ == '__main__' else None
ROOT = Path(_a.nbbo) if _a else None
OUT = Path(_a.out) if _a else None
SYMS = ['IBM', 'AAPL']
STARTS = {2: [[3e3, 30.0], [1e3, 10.0], [1e4, 1.0], [300.0, 3.0]],
          3: [[3e3, 30.0, 0.5], [1e3, 10.0, 0.1], [1e4, 100.0, 1.0], [300.0, 3.0, 0.05]]}


def files(sym):
    return sorted((ROOT / sym).glob('*.csv.gz'))


def stats():
    rows = []
    for sym in SYMS:
        acc = {'E0': [], 'E1': [], 'E1m': []}; n = {k: 0 for k in acc}
        nonpos = states = 0; dur_lock = []; dur_other = []
        for f in files(sym):
            s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
            nonpos += int((s.spread <= 0).sum()); states += len(s)
            dur_lock.append(s.dur[s.spread <= 0].to_numpy()); dur_other.append(s.dur[s.spread > 0].to_numpy())
            for k, e in (('E0', ev.events_e0(s)), ('E1', ev.events_e1(s)), ('E1m', ev.events_e1(s, min_state=0.001))):
                e = e[e.type.isin(['up', 'down'])]
                acc[k].append(np.diff(e.t.to_numpy())); n[k] += len(e)
        nd = len(files(sym)); dl = np.concatenate(dur_lock); do = np.concatenate(dur_other)
        r = {'sym': sym, 'days': nd, 'pct_states_nonpos': 100 * nonpos / states,
             'median_lock_ms': 1e3 * np.median(dl) if len(dl) else np.nan,
             'pct_open_states_lt1ms': 100 * (do < 1e-3).mean()}
        for k, gs in acc.items():
            g = np.concatenate(gs)
            r[f'events_per_day_{k}'] = n[k] / nd
            r[f'gap_lt1ms_{k}'] = 100 * (g < 1e-3).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def cause_shares(days, f):
    """Lee (2023) Table 3: average over events of mu/lambda and lambda_k/lambda."""
    A = f.alpha_matrix()            # (2, 2, K): A[i, j, k]
    K = f.K; tot = np.zeros(K + 1); n = 0
    for d, day in enumerate(days):
        mu = f.mu_by_type(d); edges = f.edges
        R = np.zeros((K, 2))        # decayed sum per kernel and source type, without alpha
        last = 0.0
        for t, i in zip(day.t, day.types):
            R *= np.exp(-f.beta * (t - last))[:, None]; last = t
            b = min(np.searchsorted(edges, t, side='right') - 1, len(edges) - 2)
            comp = np.array([f.beta[k] * (A[i, :, k] @ R[k]) for k in range(K)])
            lam = mu[b, i] + comp.sum()
            tot[0] += mu[b, i] / lam; tot[1:] += comp / lam; n += 1
            R[:, i] += 1.0
    return tot / n


def job(args):
    sym, definition, K, mode, fs = args
    days = build_days(fs, definition, 0.001)
    if mode == 'lee':      # one day, constant baseline
        f = hk.fit(days, K=K, bin_minutes=330, starts=STARTS[K])
    else:                  # ours: pooled month, 30-minute bins
        f = hk.fit(days, K=K, bin_minutes=30, starts=STARTS[K])
    sh = cause_shares(days, f)
    r = {'sym': sym, 'definition': definition, 'K': K, 'mode': mode,
         'day': fs[0].name[:8] if mode == 'lee' else 'month',
         'n_events': f.n_events, 'branching': f.branching_ratio, 'bic': f.bic,
         'share_base': sh[0]}
    for k in range(K):
        r[f'mem{k+1}_ms'] = 1e3 / f.beta[k]
        r[f'exc{k+1}'] = float(f.alpha_matrix()[0, :, k].sum())
        r[f'share{k+1}'] = sh[k + 1]
    ks = hk.ks_residuals(days, f)
    r['ks_up'] = ks['up'].statistic; r['ks_down'] = ks['down'].statistic
    return r


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    st = stats(); st.to_csv(OUT / 'lee2023_replication_stats.csv', index=False)
    print(st.round(2).T.to_string(), flush=True)
    jobs = []
    for sym in SYMS:
        for definition in ['E0', 'E1m']:
            for K in [2, 3]:
                jobs.append((sym, definition, K, 'ours', files(sym)))
                for f in files(sym):
                    jobs.append((sym, definition, K, 'lee', [f]))
    rows = []
    with ProcessPoolExecutor(4, max_tasks_per_child=1) as ex:
        for i, r in enumerate(ex.map(job, jobs)):
            rows.append(r)
            if r['mode'] == 'ours' or r['day'] == '20170103':
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if i % 20 == 0:
                pd.DataFrame(rows).to_csv(OUT / 'lee2023_replication_fits.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'lee2023_replication_fits.csv', index=False)
    print('done')

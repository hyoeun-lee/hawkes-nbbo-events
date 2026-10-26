"""Lee (2024) 0.1 s filter on our May 2017 NBBO data, fitted with K = 1 and K = 4."""
import sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from fit_panel import month_files
from hawkes_taq import events as ev, hawkes as hk
S = sys.argv[1] if len(sys.argv) > 1 else "."  # folder with nbbo/2017Q2 and the output

def grid_events(states, dt=0.1, start=hk.SESSION_START, end=hk.SESSION_END):
    """Sample the mid every dt; when it differs from the previous sample, record one event at the
    time of the last change to that price (start of the state in force), signed by the change."""
    t = states["t"].to_numpy(); m = states["mid2"].to_numpy()
    g = start + dt*np.arange(1, int(round((end-start)/dt))+1)
    i = np.searchsorted(t, g, side="right") - 1
    i0 = np.searchsorted(t, start, side="right") - 1
    idx = np.concatenate([[i0], i]); mids = m[idx]
    ch = np.nonzero(mids[1:] != mids[:-1])[0]
    tt = t[idx[1:][ch]]; d = mids[1:][ch] - mids[:-1][ch]
    return pd.DataFrame({"t": tt, "type": np.where(d > 0, "up", "down"), "size": np.abs(d)})

def days_for(sym, kind):
    out=[]
    for f in month_files([f"{S}/nbbo/2017Q2"], sym, "201705"):
        s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
        e = grid_events(s) if kind=="grid" else (ev.events_e0(s) if kind=="E0" else ev.events_e1(s, min_state=0.001))
        e = e.copy(); e["t_fit"] = ev.spread_ties(e["t"].to_numpy())
        d = hk.day_from_events(e, label=f.name[:8]); out.append((d, e))
    return out

def job(a):
    sym, K = a
    pairs = days_for(sym, "grid"); days=[p[0] for p in pairs]
    sizes = np.concatenate([p[1]["size"].to_numpy() for p in pairs])
    gaps = np.concatenate([np.diff(d.t) for d in days if len(d.t)>1])
    st = hk.full_starts(K) if K in (3,4) else None
    f = hk.fit(days, K=K, starts=st)
    mem = f.timescales(); exc = f.alpha_matrix()[0].sum(axis=0)
    n = lambda H: float(np.sum(exc*(1-np.exp(-H/mem))))
    ks = hk.ks_residuals(days, f)
    return dict(sym=sym, K=K, n_events=f.n_events, share_size_gt1=float((sizes>1).mean()),
                gaps_lt_100ms=float((gaps<0.1).mean()), gaps_lt_1ms=float((gaps<0.001).mean()),
                memory="; ".join(f"{m:.4g}" for m in mem), excitation="; ".join(f"{x:.4f}" for x in exc),
                n1=n(1), n10=n(10), BR=float(f.branching_ratio), bic=float(f.bic),
                ks_up=float(ks["up"].statistic), ks_down=float(ks["down"].statistic))

if __name__=="__main__":
    jobs=[(s,K) for s in ["INTC","BAC","MDLZ","TGT"] for K in (1,4)]
    rows=[]
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(job, jobs):
            rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(f"{S}/grid_filter_lee2024.csv", index=False); print("ALLDONE", flush=True)

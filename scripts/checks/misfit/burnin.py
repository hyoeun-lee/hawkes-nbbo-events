import sys
from pathlib import Path, time, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, ".")
import hawkes_taq.hawkes as hk
from recon import load_days, reconstruct, S
from diag import per_event, diagnostics
from scipy import stats
k = pd.read_csv(S + "/delta/K4.csv", dtype={"month": str}); k = k[(k.definition == "E1m") & (k.K == 4) & (k.month == "201705")]
def mass(f, lo, H): return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))
def first_bin_mean(days, f, burn):
    """mean residual of up and down moves between burn and the end of the first bin, and in the last bin."""
    ev = per_event(days, f)   # with f.burn > 0, residuals exclude events before burn (per_event copies burn? it rebuilds the fit)
    out = []
    for i in (0, 1):
        e = ev[ev.type == i]
        out.append((round(e[e.t < 1800].r.mean(), 3), round(e[e.t >= days[0].T - 1800].r.mean(), 3)))
    return out
sym = sys.argv[1]; burn = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
row = k[k.sym == sym].iloc[0]; days = load_days(sym)
f0 = reconstruct(days, row)
# the original fit scored on the same events (residuals from burn on)
f0b = hk.HawkesFit(**{**f0.__dict__, "burn": burn})
t0 = time.time()
fb = hk.fit(days, K=4, kernel="symmetric", bin_minutes=30, starts=[f0.beta], baseline="shared", delta=0.001, burn=burn)
print(sym, f"burn {burn:.0f}s fit in {(time.time()-t0)/60:.1f} min", flush=True)
pickle.dump(fb, open(f"fit_burn_{sym}.pkl", "wb"))
for tag, f in (("main, all events", f0), ("main, scored from burn", f0b), ("refit with burn-in", fb)):
    ks = hk.ks_residuals(days, f)
    fb_mean = first_bin_mean(days, f, f.burn)
    print(f"{sym} {tag:24s} loglik {f.loglik:.1f} | mem s {np.round(1/f.beta,4)} | n1 {mass(f,0.001,1):.4f} n10 {mass(f,0.001,10):.4f} total {f.branching_ratio:.3f}"
          f" | KS up {ks['up'].statistic:.4f} down {ks['down'].statistic:.4f} | first-bin mean up/down {fb_mean[0][0]}/{fb_mean[1][0]} last-bin {fb_mean[0][1]}/{fb_mean[1][1]}", flush=True)

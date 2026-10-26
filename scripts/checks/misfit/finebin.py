import sys
from pathlib import Path, time, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, ".")
import hawkes_taq.hawkes as hk
from recon import load_days, reconstruct, S, parse
from response import response
from diag import per_event, diagnostics
k = pd.read_csv(S + "/delta/K4.csv", dtype={"month": str}); k = k[(k.definition == "E1m") & (k.K == 4) & (k.month == "201705")]
def mass(f, lo, H): return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))
sym = sys.argv[1]; row = k[k.sym == sym].iloc[0]; days = load_days(sym)
fits = {30: reconstruct(days, row)}
for bm in (10, 5):
    t0 = time.time()
    fits[bm] = hk.fit(days, K=4, kernel="symmetric", bin_minutes=bm, starts=[fits[30].beta], baseline="shared", delta=0.001)
    print(sym, bm, f"min fitted in {(time.time()-t0)/60:.1f} min", flush=True)
pickle.dump(fits, open(f"fits_fine_{sym}.pkl", "wb"))
for bm, f in fits.items():
    ks = hk.ks_residuals(days, f); rs, ro = response(days)
    sims = [response(hk.simulate_like(days, f, np.random.default_rng([31, b]))) for b in range(2)]
    d = diagnostics(per_event(days, f), f.edges)
    print(f"{sym} bins={bm:2d} loglik {f.loglik:.1f} AIC {f.aic:.1f} BIC {f.bic:.1f} | mem s {np.round(1/f.beta,4)} | n1 {mass(f,0.001,1):.4f} n10 {mass(f,0.001,10):.4f} total {f.branching_ratio:.3f} eff {f.effective_excitation().sum():.3f}"
          f" | KS up {ks['up'].statistic:.4f} down {ks['down'].statistic:.4f} | ac up {d['up']['ac']} | first/last bin mean up {d['up']['bin_mean'][0]}/{d['up']['bin_mean'][-1]}"
          f" | resp same real {np.round(rs,2)} sim {np.round(np.mean([s[0] for s in sims],axis=0),2)}", flush=True)

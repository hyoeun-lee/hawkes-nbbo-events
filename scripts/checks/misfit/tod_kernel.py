import sys
from pathlib import Path, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, ".")
import hawkes_taq.hawkes as hk
from recon import load_days, reconstruct, S
k = pd.read_csv(S + "/delta/K4.csv", dtype={"month": str}); k = k[(k.definition == "E1m") & (k.K == 4) & (k.month == "201705")]
def mass(f, lo, H): return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))
def piece(days, a, b):
    """the window [a, b) of each day as its own session (times shifted to start at 0)"""
    out = []
    for d in days:
        m = (d.t >= a) & (d.t < b)
        out.append(hk.Day(t=d.t[m] - a, types=d.types[m], T=b - a, label=d.label))
    return out
sym = sys.argv[1]; row = k[k.sym == sym].iloc[0]; days = load_days(sym); f0 = reconstruct(days, row)
T = days[0].T
windows = [("10:00-11:00", 0.0, 3600.0), ("11:00-14:30", 3600.0, T - 3600.0), ("14:30-15:30", T - 3600.0, T), ("11:00-15:30", 3600.0, T)]
print(f"{sym} whole month: n1 {mass(f0,0.001,1):.4f} n10 {mass(f0,0.001,10):.4f} total {f0.branching_ratio:.3f} mem {np.round(1/f0.beta,3)}", flush=True)
for name, a, b in windows:
    ds = piece(days, a, b)
    f = hk.fit(ds, K=4, kernel="symmetric", bin_minutes=30, starts=[f0.beta], baseline="shared", delta=0.001, burn=60.0)
    n = sum(len(d.t) for d in ds)
    print(f"{sym} {name:12s} events/day {n/len(ds):7.0f} | n1 {mass(f,0.001,1):.4f} n10 {mass(f,0.001,10):.4f} total {f.branching_ratio:.3f} | mem {np.round(1/f.beta,3)} | exc {np.round(f.alpha.sum(axis=0),3)}", flush=True)

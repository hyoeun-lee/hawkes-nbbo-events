import sys
from pathlib import Path, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, ".")
import hawkes_taq.hawkes as hk
from recon import load_days, reconstruct, S
from scipy import stats
edges_gap = [0, 0.01, 0.1, 1.0, 10.0, np.inf]; labs = ["<10ms", "10-100ms", "0.1-1s", "1-10s", ">10s"]
def alternation(days):
    rows = []
    for day in days:
        g = np.diff(day.t); alt = day.types[1:] != day.types[:-1]
        rows.append(pd.DataFrame({"gap": g, "alt": alt}))
    e = pd.concat(rows); e["b"] = pd.cut(e.gap, edges_gap, labels=labs, right=False)
    tot = e.alt.mean(); by = e.groupby("b", observed=True).alt.agg(["mean", "size"])
    return tot, by
def day_balance(days):
    """correlation across days of (n_up - n_down)/n with the up-residual deficit is implicit; here: per-day share of up moves."""
    sh = np.array([(d.types == 0).mean() for d in days]); n = np.array([len(d.t) for d in days])
    z = (sh - 0.5) * 2 * np.sqrt(n)  # z of up/down imbalance per day
    return float(np.abs(z).max()), int((np.abs(z) > 2).sum()), float(np.std(sh))
k = pd.read_csv(S + "/delta/K4.csv", dtype={"month": str}); k = k[(k.definition == "E1m") & (k.K == 4) & (k.month == "201705")]
out = []
for sym in ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]:
    row = k[k.sym == sym].iloc[0]; days = load_days(sym); fit = reconstruct(days, row)
    tot, by = alternation(days); zmax, nz, sdsh = day_balance(days)
    sims = [hk.simulate_like(days, fit, np.random.default_rng([11, b])) for b in range(3)]
    stot = [alternation(s)[0] for s in sims]; sby = pd.concat([alternation(s)[1]["mean"] for s in sims], axis=1).mean(axis=1)
    sbal = [day_balance(s) for s in sims]
    line = f"{sym:5s} alternation share: real {tot:.3f} sim {np.mean(stot):.3f} | by gap real/sim: " + " ".join(f"{l} {by.loc[l,'mean']:.3f}/{sby.loc[l]:.3f} (n={by.loc[l,'size']})" for l in labs)
    line += f" | day up-share: sd real {sdsh:.3f} sim {np.mean([s[2] for s in sbal]):.3f}; days |z|>2 real {nz} sim {np.mean([s[1] for s in sbal]):.1f}"
    print(line); out.append(line)

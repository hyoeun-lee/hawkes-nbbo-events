import sys
from pathlib import Path, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, ".")
import hawkes_taq.hawkes as hk
from recon import load_days, reconstruct, S
lags = [0.01, 0.1, 1.0, 10.0, 100.0]
def response(days):
    """rate of same-direction and opposite-direction events in lag windows after an event, relative to the unconditional rate (events per second of that type)."""
    same = np.zeros(len(lags) - 1); opp = np.zeros(len(lags) - 1); n = 0; T = 0.0; N = 0
    for day in days:
        T += day.T; N += len(day.t)
        tt = [day.t[day.types == j] for j in (0, 1)]
        for i in (0, 1):
            src = tt[i]; n += len(src)
            for j in (0, 1):
                cnt = np.searchsorted(tt[j], src[:, None] + np.array(lags)[None, :], side="left")
                c = np.diff(cnt, axis=1).sum(axis=0)
                if i == j: same += c
                else: opp += c
    rate = (N / 2) / T                      # unconditional rate per type (per second)
    w = np.diff(lags)
    return same / n / w / rate, opp / n / w / rate   # ratio to the unconditional rate
if __name__ == "__main__":
  k = pd.read_csv(S + "/delta/K4.csv", dtype={"month": str}); k = k[(k.definition == "E1m") & (k.K == 4) & (k.month == "201705")]
  print("ratio of conditional to unconditional rate; windows 10-100ms, 0.1-1s, 1-10s, 10-100s")
  for sym in ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]:
      row = k[k.sym == sym].iloc[0]; days = load_days(sym); fit = reconstruct(days, row)
      rs, ro = response(days)
      sims = [response(hk.simulate_like(days, fit, np.random.default_rng([23, b]))) for b in range(3)]
      ss_ = np.mean([s[0] for s in sims], axis=0); so = np.mean([s[1] for s in sims], axis=0)
      print(f"{sym:5s} same real {np.round(rs,2)} sim {np.round(ss_,2)} | opposite real {np.round(ro,2)} sim {np.round(so,2)}")

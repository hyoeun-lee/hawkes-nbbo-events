import sys, time, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scipy import optimize
import hawkes_taq.hawkes as hk, hawkes_taq.simstudy as ss
import os
S = os.environ.get("HAWKES_SCRATCH", "data/scratch")  # folder with may/NBBO/<SYM>/*.csv.gz and delta/K4.csv
def parse(s): return np.array([float(x) for x in str(s).split(";")])
def load_days(sym):
    from pathlib import Path
    files = sorted(Path(S, "may/NBBO", sym).glob("*.csv.gz"))
    days = ss.days_from_nbbo([pd.read_csv(f) for f in files], "E1m")
    for d, f in zip(days, files): d.label = f.name.split(".")[0]
    return days
def reconstruct(days, row, bin_minutes=30):
    beta = 1.0 / parse(row.memory_s); alpha = np.vstack([parse(row.self_per_scale), parse(row.cross_per_scale)])
    edges = hk._bins(days[0].T, bin_minutes)
    design = hk._design(days, beta, "symmetric", edges, {}, baseline="shared", delta=float(row.delta))
    a = alpha.ravel(); off = design.X @ a; c_off = design.c_alpha @ a; P = len(design.c_mu)
    def f(mu):
        lam = mu[design.bidx] + off; w = 1.0 / lam
        return -(np.log(lam).sum() - design.c_mu @ mu - c_off), -(np.bincount(design.bidx, weights=w, minlength=P) - design.c_mu)
    x0 = np.maximum(np.bincount(design.bidx, minlength=P) / design.c_mu, 1e-6) * 0.5
    res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(1e-10, None)] * P)
    nb = len(edges) - 1
    fit = hk.HawkesFit(K=len(beta), kernel="symmetric", beta=beta, alpha=alpha, mu=res.x.reshape(len(days), nb), edges=edges,
                       loglik=-res.fun, n_events=sum(len(d.t) for d in days), n_params=P + 2 * len(beta) + len(beta), baseline="shared", delta=float(row.delta))
    return fit
if __name__ == "__main__":
    k = pd.read_csv(S + "/delta/K4.csv", dtype={"month": str}); k = k[(k.definition == "E1m") & (k.K == 4) & (k.month == "201705")]
    sym = sys.argv[1]; row = k[k.sym == sym].iloc[0]
    t0 = time.time(); days = load_days(sym); print(sym, "days", len(days), "events", sum(len(d.t) for d in days), "csv n_events", row.n_events, f"{time.time()-t0:.0f}s")
    fit = reconstruct(days, row); print("loglik", round(fit.loglik, 2), "csv", row.loglik, f"{time.time()-t0:.0f}s")
    ks = hk.ks_residuals(days, fit); print("KS up", round(ks["up"].statistic, 4), "csv", row.ks_up, "| down", round(ks["down"].statistic, 4), "csv", row.ks_down)

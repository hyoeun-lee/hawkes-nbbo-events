import sys
from pathlib import Path, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, ".")
import hawkes_taq.hawkes as hk
from recon import load_days, reconstruct, S
from scipy import stats
def per_event(days, fit):
    """rows: day index, type, time of day (s), residual."""
    out = []
    for d, day in enumerate(days):
        burn = getattr(fit, "burn", 0.0)
        f1 = hk.HawkesFit(K=fit.K, kernel=fit.kernel, beta=fit.beta, alpha=fit.alpha, mu=fit.mu[d:d+1], edges=fit.edges,
                          loglik=0.0, n_events=len(day.t), n_params=0, baseline=fit.baseline, delta=fit.delta, burn=burn)
        res = hk.residuals([day], f1)
        prev_type = np.concatenate([[-1], day.types[:-1]])
        for i in (0, 1):
            m = day.types == i
            t_i, p_i = day.t[m], prev_type[m]
            if burn > 0:   # residuals() drops events before burn and the first one after it
                idx = np.where(t_i >= burn)[0][1:]
                t_i, p_i = t_i[idx], p_i[idx]
            out.append(pd.DataFrame({"d": d, "type": i, "t": t_i, "r": res[i], "prev": p_i}))
    return pd.concat(out, ignore_index=True)
def diagnostics(ev, edges):
    rows = {}
    for i, name in ((0, "up"), (1, "down")):
        e = ev[ev.type == i]; r = e.r.to_numpy(); n = len(r)
        ks = stats.kstest(r, "expon"); xs = np.sort(r); F = np.arange(1, n + 1) / n; dev = F - (1 - np.exp(-xs)); j = np.argmax(np.abs(dev))
        q = {x: float((r <= x).mean() - (1 - np.exp(-x))) for x in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)}
        byday = e.groupby("d").r.agg(["mean", "size", lambda v: stats.kstest(v, "expon").statistic]); byday.columns = ["mean", "n", "ks"]
        z = (byday["mean"] - 1) * np.sqrt(byday.n)
        b = np.clip(np.searchsorted(edges, e.t, side="right") - 1, 0, len(edges) - 2)
        bybin = e.groupby(b).r.agg(["mean", "size", lambda v: stats.kstest(v, "expon").statistic]); bybin.columns = ["mean", "n", "ks"]
        ac = []
        for lag in (1, 2, 3):
            num = 0.0; den = 0.0; cnt = 0
            for _, g in e.groupby("d"):
                v = g.r.to_numpy() - 1.0
                if len(v) > lag + 5: num += np.sum(v[lag:] * v[:-lag]); den += np.sum(v * v); cnt += len(v) - lag
            ac.append((num / den, np.sqrt(cnt)))
        prev_same = e[e.prev == i].r.mean(); prev_other = e[e.prev == 1 - i].r.mean()
        rows[name] = dict(n=n, ks=ks.statistic, at=xs[j], dev=dev[j], q=q, day_ks_max=byday.ks.max(), day_ks_med=byday.ks.median(),
                          day_z_abs_gt2=int((z.abs() > 2).sum()), day_mean_min=byday["mean"].min(), day_mean_max=byday["mean"].max(),
                          bin_mean=bybin["mean"].round(3).tolist(), bin_ks=bybin.ks.round(3).tolist(),
                          ac=[(round(a, 4), round(a * s, 1)) for a, s in ac], prev_same=prev_same, prev_other=prev_other,
                          n_prev_same=int((e.prev == i).sum()), n_prev_other=int((e.prev == 1 - i).sum()))
    return rows
def show(tag, rows):
    for name, r in rows.items():
        q = " ".join(f"{k}:{v:+.3f}" for k, v in r["q"].items())
        print(f"{tag:14s} {name:4s} n={r['n']:6d} KS={r['ks']:.4f} at r={r['at']:.2f} dev={r['dev']:+.4f} | Femp-F: {q}")
        print(f"{'':14s}      days: KS med {r['day_ks_med']:.3f} max {r['day_ks_max']:.3f}; mean range {r['day_mean_min']:.3f}-{r['day_mean_max']:.3f}; |z|>2: {r['day_z_abs_gt2']}")
        print(f"{'':14s}      bins mean: {r['bin_mean']}")
        print(f"{'':14s}      bins KS:   {r['bin_ks']}")
        print(f"{'':14s}      autocorr (r, z): {r['ac']} | mean resid, prev same {r['prev_same']:.3f} (n={r['n_prev_same']}) vs prev other {r['prev_other']:.3f} (n={r['n_prev_other']})")
if __name__ == "__main__":
    k = pd.read_csv(S + "/delta/K4.csv", dtype={"month": str}); k = k[(k.definition == "E1m") & (k.K == 4) & (k.month == "201705")]
    syms = sys.argv[1:] or ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]
    for sym in syms:
        row = k[k.sym == sym].iloc[0]; days = load_days(sym); fit = reconstruct(days, row)
        ev = per_event(days, fit); ev.to_pickle(f"ev_{sym}.pkl")
        show(sym + " real", diagnostics(ev, fit.edges))
        for b in range(2):
            sim = hk.simulate_like(days, fit, np.random.default_rng([7, b]))
            show(f"{sym} sim{b}", diagnostics(per_event(sim, fit), fit.edges))
        print()

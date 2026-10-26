"""Score the by-hour kernels on their own windows (revision item A14 c).

Section 6.5 of the paper reports that the mean residual of the main (month-wide) fit in the first
half hour is below 1 (0.83-0.89 for MDLZ, T, TGT and ATVI in May 2017) and above 1 in the last
half hour, and attributes this to a kernel that varies over the day. This script tests the
attribution: each window of the day (10:00-11:00, 11:00-14:30, 14:30-15:30, as its own session with
a 60 s burn-in) is scored under three models,

  month    the month-wide kernel and the month-wide baselines (the original diagnostic, restricted
           to the window's events after the burn-in);
  month+b  the month-wide kernel with the baselines refitted on the window alone (so that a
           baseline level wrong for the window cannot be blamed);
  window   the kernel fitted on the window (fit_by_hour.py's fits),

and the mean residual of up and down moves in each half hour of the window is reported, with the
KS statistic of the window's residuals. Two months simulated from the window fit give the
reference for a correct model. If the window kernel removes the first-half-hour deficit, the open
misfit is time-varying excitation; if not, it is something else (a baseline trend within the bin).

    python scripts/checks/misfit/score_windows.py --nbbo DATA/nbbo_2017May --fits DATA/fits_e1m_delta_K4.csv \\
        --month 201705 --syms INTC ... --out score_windows_201705.csv
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hawkes_taq import hawkes as hk  # noqa: E402
import fit_panel as fp  # noqa: E402

BURN = 60.0


def parse(s):
    return np.array([float(x) for x in str(s).split(";")])


def windows(T):
    return [("10:00-11:00", 0.0, 3600.0), ("11:00-14:30", 3600.0, T - 3600.0), ("14:30-15:30", T - 3600.0, T)]


def piece(days, a, b):
    out = []
    for d in days:
        m = (d.t >= a) & (d.t < b)
        if m.sum() > 10:
            out.append(hk.Day(t=d.t[m] - a, types=d.types[m], T=b - a, label=d.label))
    return out


def baseline_refit(days, beta, alpha, delta, burn, bin_minutes=30):
    """Baselines by maximum likelihood with the kernel (beta, alpha) fixed; returns a HawkesFit."""
    edges = hk._bins(days[0].T, bin_minutes)
    design = hk._design(days, beta, "symmetric", edges, {}, baseline="shared", delta=delta, burn=burn)
    a = alpha.ravel(); off = design.X @ a; c_off = design.c_alpha @ a; P = len(design.c_mu)

    def f(mu):
        lam = mu[design.bidx] + off
        if np.any(lam <= 0):
            return np.inf, np.zeros_like(mu)
        w = 1.0 / lam
        return -(np.log(lam).sum() - design.c_mu @ mu - c_off), -(np.bincount(design.bidx, weights=w, minlength=P) - design.c_mu)

    x0 = np.maximum(np.bincount(design.bidx, minlength=P) / np.maximum(design.c_mu, 1e-9), 1e-6) * 0.5
    res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(1e-10, None)] * P)
    nb = len(edges) - 1
    return hk.HawkesFit(K=len(beta), kernel="symmetric", beta=beta, alpha=alpha, mu=res.x.reshape(len(days), nb), edges=edges,
                        loglik=-res.fun, n_events=sum(int(np.sum(d.t >= burn)) for d in days), n_params=P + 2 * len(beta),
                        baseline="shared", delta=delta, burn=burn)


def month_on_window(days_full, fit_month, lo, hi, burn):
    """The month-wide fit restricted to the window: same kernel, the month's baselines of the bins
    inside the window, scored from lo + burn with the window's own history only."""
    ds = piece(days_full, lo, hi)
    edges = hk._bins(ds[0].T, 30)
    b0 = int(round(lo / 1800.0))
    nb = len(edges) - 1
    mu = fit_month.mu[:, b0:b0 + nb]
    return ds, hk.HawkesFit(K=fit_month.K, kernel="symmetric", beta=fit_month.beta, alpha=fit_month.alpha, mu=mu, edges=edges,
                            loglik=0.0, n_events=0, n_params=0, baseline="shared", delta=fit_month.delta, burn=burn)


def per_event(days, fit):
    out = []
    for d, day in enumerate(days):
        f1 = hk.HawkesFit(K=fit.K, kernel=fit.kernel, beta=fit.beta, alpha=fit.alpha, mu=fit.mu[d:d + 1], edges=fit.edges,
                          loglik=0.0, n_events=len(day.t), n_params=0, baseline=fit.baseline, delta=fit.delta, burn=fit.burn)
        res = hk.residuals([day], f1)
        for i in (0, 1):
            t_i = day.t[day.types == i]
            if fit.burn > 0:
                idx = np.where(t_i >= fit.burn)[0][1:]
                t_i = t_i[idx]
            out.append(pd.DataFrame({"type": i, "t": t_i, "r": res[i]}))
    return pd.concat(out, ignore_index=True)


def summarize(ev, T):
    r = {}
    for i, name in ((0, "up"), (1, "down")):
        e = ev[ev.type == i]
        r[f"ks_{name}"] = stats.kstest(e.r, "expon").statistic
        r[f"first_half_{name}"] = e[e.t < 1800].r.mean()
        r[f"second_half_{name}"] = e[(e.t >= 1800) & (e.t < 3600)].r.mean() if T > 3600 else np.nan
        r[f"last_half_{name}"] = e[e.t >= T - 1800].r.mean()
        r[f"n_{name}"] = len(e)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--fits", required=True)
    p.add_argument("--month", default="201705")
    p.add_argument("--syms", nargs="+", default=["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--min-state", type=float, default=0.001)
    p.add_argument("--sims", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    fits = pd.read_csv(a.fits, dtype={"month": str})
    fits = fits[(fits.definition == "E1m") & (fits.K == a.K) & (fits.month == a.month)]
    rows = []
    for sym in a.syms:
        row = fits[fits.sym == sym].iloc[0]
        beta = 1.0 / parse(row.memory_s); alpha = np.vstack([parse(row.self_per_scale), parse(row.cross_per_scale)])
        days = fp.build_days(fp.month_files(a.nbbo, sym, a.month), "E1m", a.min_state)
        T = days[0].T
        fm = baseline_refit(days, beta, alpha, a.delta, 0.0)       # the month fit, reconstructed
        print(f"{sym}: month fit loglik {fm.loglik:.2f} (csv {row.loglik})", flush=True)
        for name, lo, hi in windows(T):
            ds, f_month = month_on_window(days, fm, lo, hi, BURN)
            f_monthb = baseline_refit(ds, beta, alpha, a.delta, BURN)
            f_win = hk.fit(ds, K=a.K, kernel="symmetric", bin_minutes=30, starts=[beta], baseline="shared", delta=a.delta, burn=BURN)
            for tag, f in (("month", f_month), ("month+b", f_monthb), ("window", f_win)):
                r = {"sym": sym, "window": name, "model": tag, "n1": float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * a.delta) - np.exp(-f.beta)))),
                     **summarize(per_event(ds, f), hi - lo)}
                rows.append(r)
                print(f"  {name} {tag:8s} n1 {r['n1']:.3f} | first half up {r['first_half_up']:.3f} down {r['first_half_down']:.3f} "
                      f"| last half up {r['last_half_up']:.3f} down {r['last_half_down']:.3f} | KS {r['ks_up']:.4f}/{r['ks_down']:.4f}", flush=True)
            for b in range(a.sims):
                sim = hk.simulate_like(ds, f_win, np.random.default_rng([11, b, len(sym)]))
                r = {"sym": sym, "window": name, "model": f"sim{b}", "n1": np.nan, **summarize(per_event(sim, f_win), hi - lo)}
                rows.append(r)
                print(f"  {name} sim{b}     first half up {r['first_half_up']:.3f} down {r['first_half_down']:.3f} | last half up {r['last_half_up']:.3f} "
                      f"down {r['last_half_down']:.3f} | KS {r['ks_up']:.4f}/{r['ks_down']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

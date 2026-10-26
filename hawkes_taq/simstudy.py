"""Simulation study: does cleaning the consolidated feed recover a known clean process?

One replicate for one stock and one experiment:

1. Truth. Clean price moves are simulated from the stock's fitted E1m model (May 2017, K = 4,
   fitted with the 1 ms dead time, so the kernel is estimated as if events could occur in the
   first millisecond), with that month's baselines per day and 30-minute bin. The clean process
   itself has no dead time: the dead time is a property of the consolidated feed.
2. Observation. The clean moves are turned into a synthetic consolidated NBBO with the stock's
   calibrated brief states (hawkes_taq.simulate_nbbo), and the real event pipeline builds E0,
   E1 and E1m from it.
3. Estimation. The same fitting code as for the real data, with and without the dead time.

Experiments (EXPERIMENTS below):
  base      calibrated brief states;
  share0    no brief states (artifact_scale 0): the feed shows the clean process directly;
  share50   half the calibrated brief states;
  long      brief-state durations tripled (a BAC-like tail beyond 1 ms);
  subms     calibrated brief states, and the truth has a genuine sub-millisecond component
            (memory 0.3 ms, excitation 0.10) added to its kernel.

Second batch (BATCH2; review of Oct 8, 2026):
  subms_clean  the sub-millisecond truth with no brief states: how much genuine fast excitation
               does merging erase when there is nothing to clean?
  powerlaw     the truth's kernel replaced by a power-law kernel with the same total mass,
               fitted to the truth's cumulative kernel mass and represented by 15 exponentials
               (0.2 ms to 200 s); estimated with K = 4 (misspecification).
  actscale     the truth's kernel with its baselines scaled by 0.5, 0.71, 1, 1.41 or 2 (by
               replicate): activity varies, the kernel does not. Does the estimator alone
               produce a relation between activity and estimated excitation?

The clean (true) process never has a dead time; the dead time enters only in estimation.

Every fit is written as one row with the truth alongside, so bias and RMSE can be computed
afterwards (scripts/summarize_simulation.py).
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import events as ev
from . import hawkes as hk
from .simulate_nbbo import BriefStateModel, synthetic_nbbo

EXPERIMENTS = {
    "base": dict(artifact_scale=1.0, duration_scale=1.0, subms=False),
    "share0": dict(artifact_scale=0.0, duration_scale=1.0, subms=False),
    "share50": dict(artifact_scale=0.5, duration_scale=1.0, subms=False),
    "long": dict(artifact_scale=1.0, duration_scale=3.0, subms=False),
    "subms": dict(artifact_scale=1.0, duration_scale=1.0, subms=True),
    # second batch
    "subms_clean": dict(artifact_scale=0.0, duration_scale=1.0, subms=True),
    "powerlaw": dict(artifact_scale=1.0, duration_scale=1.0, subms=False, truth="powerlaw"),
    "actscale": dict(artifact_scale=1.0, duration_scale=1.0, subms=False, scaled_baseline=True),
}
BATCH1 = ("base", "share0", "share50", "long", "subms")
BATCH2 = ("subms_clean", "powerlaw", "actscale")
ACT_SCALES = (0.5, 2 ** -0.5, 1.0, 2 ** 0.5, 2.0)
SUBMS_MEMORY, SUBMS_EXCITATION = 0.0003, 0.10
DELTA = 0.001


@dataclass
class Truth:
    sym: str
    beta: np.ndarray          # (K,)
    alpha: np.ndarray         # (2, K): rows self, cross
    mu: np.ndarray            # (n_days, n_bins)
    edges: np.ndarray

    def with_subms(self):
        """Add a genuine sub-millisecond component, split between self and cross like the
        fastest existing component."""
        a0 = self.alpha[:, 0]
        split = a0 / a0.sum() if a0.sum() > 0 else np.array([0.5, 0.5])
        beta = np.concatenate([[1.0 / SUBMS_MEMORY], self.beta])
        alpha = np.concatenate([(SUBMS_EXCITATION * split)[:, None], self.alpha], axis=1)
        return Truth(self.sym, beta, alpha, self.mu, self.edges)

    def with_baseline_scale(self, scale):
        """Same kernel, baselines multiplied by scale."""
        return Truth(self.sym, self.beta, self.alpha, self.mu * scale, self.edges)

    def powerlaw(self, n_exp=15, fast=2e-4, slow=200.0):
        """A power-law kernel with the same total mass, phi(t) = n theta c^theta (c + t)^(-1-theta),
        with (c, theta) fitted to this truth's cumulative kernel mass at 1 ms - 100 s (log scale), represented
        by n_exp exponentials with rates from 1/slow to 1/fast (Laplace mixture of the power law).
        Self and cross excitation are split in the truth's overall proportion."""
        from scipy import optimize, special
        n = self.branching_ratio
        H = np.logspace(-3, 2, 30)
        target = np.array([self.mass_within(h) for h in H])

        def pl_mass(params, h):
            c, th = np.exp(params)
            return n * (1.0 - (c / (c + h)) ** th)

        # relative (log) errors, so short horizons count as much as long ones
        res = optimize.minimize(lambda q: np.sum((np.log(pl_mass(q, H)) - np.log(target)) ** 2),
                                x0=np.log([0.01, 0.3]), method="Nelder-Mead")
        c, th = np.exp(res.x)
        # (c + t)^(-1-th) = (1 / Gamma(1+th)) * int x^th e^(-x c) e^(-x t) dx: discretize in log x
        x = np.logspace(np.log10(1.0 / slow), np.log10(1.0 / fast), n_exp)
        dlog = np.log(x[1] / x[0])
        w = x ** th * np.exp(-x * c) / special.gamma(1.0 + th) * x * dlog   # alpha_j * beta_j
        a = w / x
        a *= n / a.sum()
        self_share = self.alpha[0].sum() / self.alpha.sum()
        alpha = np.vstack([a * self_share, a * (1.0 - self_share)])
        order = np.argsort(x)[::-1]
        t = Truth(self.sym, x[order], alpha[:, order], self.mu, self.edges)
        t.powerlaw_params = (float(c), float(th))
        return t

    def mass_within(self, H, delta=0.0):
        """Kernel mass between delta and H, summed over components (self + cross)."""
        return float(np.sum(self.excitation * (np.exp(-self.beta * delta) - np.exp(-self.beta * H))))

    @property
    def excitation(self):
        return self.alpha.sum(axis=0)

    @property
    def branching_ratio(self):
        return float(self.alpha.sum())

    def save(self, path):
        np.savez(path, sym=self.sym, beta=self.beta, alpha=self.alpha, mu=self.mu, edges=self.edges)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(str(z["sym"]), z["beta"], z["alpha"], z["mu"], z["edges"])


def days_from_nbbo(frames, definition, min_state=0.001):
    """Event Days from a list of daily NBBO frames, exactly as for the real data."""
    out = []
    for i, nb in enumerate(frames):
        s = ev.build_states(nb, hk.SESSION_END)
        e = ev.events_e0(s) if definition == "E0" else ev.events_e1(
            s, min_state=min_state if definition == "E1m" else 0.0)
        e["t_fit"] = ev.spread_ties(e["t"])
        d = hk.day_from_events(e, label=str(i))
        if len(d.t):
            out.append(d)
    return out


def _gap_share(days, h=1e-3):
    g = np.concatenate([np.diff(d.t) for d in days if len(d.t) > 1])
    return float((g < h).mean())


def _starts(K, truth_beta, definition):
    """Decay-rate starts: the default grid plus the truth's ladder (with a sub-ms rate for raw
    events), so the search covers the region where the answer is expected."""
    starts = list(hk.full_starts(K)) if K <= 4 else list(hk._default_starts(K))
    tb = np.sort(truth_beta)[::-1]
    if len(tb) >= K:
        starts.append(tb[:K])
    if definition in ("E0", "E1"):
        starts.append(np.concatenate([[2000.0], tb[:K - 1]]))
    return starts


def _mass(f, H):
    """Fitted kernel mass between the fit's dead time and H (self + cross)."""
    exc = f.alpha_matrix()[0].sum(axis=0)
    return float(np.sum(exc * (np.exp(-f.beta * f.delta) - np.exp(-f.beta * H))))


def _row(f, days, definition, K, delta, pool, wall):
    ks = hk.ks_residuals(days, f)
    eff = f.effective_excitation()
    n = sum(len(d.t) for d in days)
    return {"definition": definition, "K": K, "delta": delta, "pool": pool,
            "n_days_fit": len(days), "n_events": n,
            "memory_s": "; ".join(f"{m:.5g}" for m in f.timescales()),
            "excitation": "; ".join(f"{a:.5f}" for a in f.alpha_matrix()[0].sum(axis=0)),
            "effective_excitation": "; ".join(f"{a:.5f}" for a in eff),
            "branching_ratio": float(f.branching_ratio), "effective_br": float(eff.sum()),
            "n1": _mass(f, 1.0), "n10": _mass(f, 10.0),
            "baseline_events": float(np.sum(f.mu * np.diff(f.edges)) * 2),
            "loglik": float(f.loglik), "bic": float(f.bic),
            "ks_up": float(ks["up"].statistic), "ks_down": float(ks["down"].statistic),
            "minutes": round(wall / 60, 2)}


def run_replicate(truth, model, experiment, rep, seed, fits="full", log=print):
    """Simulate one month and fit it. Returns a list of result rows (dicts)."""
    cfg = EXPERIMENTS[experiment]
    rng = np.random.default_rng(seed)
    tr = truth.with_subms() if cfg["subms"] else truth
    if cfg.get("truth") == "powerlaw":
        tr = tr.powerlaw()
    scale = ACT_SCALES[rep % len(ACT_SCALES)] if cfg.get("scaled_baseline") else 1.0
    if scale != 1.0:
        tr = tr.with_baseline_scale(scale)
    obs = model.scaled(cfg["artifact_scale"], cfg["duration_scale"])
    T = float(tr.edges[-1])

    clean, frames = [], []
    for d in range(tr.mu.shape[0]):
        day = hk.simulate_day(tr.mu[d], tr.edges, tr.alpha, tr.beta, rng, label=str(d))
        clean.append(day)
        frames.append(synthetic_nbbo(day.t, day.types, obs, rng, T=T))
    sets = {"E0": days_from_nbbo(frames, "E0"), "E1": days_from_nbbo(frames, "E1"),
            "E1m": days_from_nbbo(frames, "E1m")}

    n_clean = sum(len(d.t) for d in clean)
    common = {"sym": tr.sym, "experiment": experiment, "rep": rep, "seed": seed,
              "true_memory_s": "; ".join(f"{m:.5g}" for m in 1 / tr.beta),
              "true_excitation": "; ".join(f"{a:.5f}" for a in tr.excitation),
              "true_br": tr.branching_ratio, "baseline_scale": scale,
              "true_n1": tr.mass_within(1.0), "true_n10": tr.mass_within(10.0),
              "true_baseline_events": float(np.sum(tr.mu * np.diff(tr.edges)) * 2),
              "n_clean": n_clean, "clean_gap_lt1ms": _gap_share(clean),
              "n_e0": sum(len(d.t) for d in sets["E0"]), "n_e1": sum(len(d.t) for d in sets["E1"]),
              "n_e1m": sum(len(d.t) for d in sets["E1m"]),
              "e0_gap_lt1ms": _gap_share(sets["E0"]), "e1_gap_lt1ms": _gap_share(sets["E1"])}

    plan = [("E1m", 4, DELTA, "all"), ("E1m", 4, 0.0, "all"), ("E0", 4, 0.0, "all"),
            ("E1", 4, 0.0, "all")]
    if fits == "full" and experiment == "base":
        plan += [("E1m", 3, DELTA, "all"), ("E1m", 5, DELTA, "all"),
                 ("E1m", 4, DELTA, "days10"), ("E1m", 4, DELTA, "days5"),
                 ("E1m", 4, DELTA, "day0"), ("E1m", 4, DELTA, "day1"), ("E1m", 4, DELTA, "day2")]
    if experiment in ("subms", "subms_clean"):
        plan += [("E0", 5, 0.0, "all")]
    if experiment == "actscale":
        plan = [("E1m", 4, DELTA, "all"), ("E0", 4, 0.0, "all")]
    if fits == "truth":
        # the four core fits plus a fit of the simulated clean process itself, before the
        # observation layer, without the gate: its bias is the estimator's alone, so the
        # difference from the E1m fit is what the persistence filter and the gate add
        plan = plan[:4] + [("TRUE", 4, 0.0, "all")]

    rows = []
    for definition, K, delta, pool in plan:
        days = clean if definition == "TRUE" else sets[definition]
        if pool == "days10":
            days = days[:10]
        elif pool == "days5":
            days = days[:5]
        elif pool.startswith("day"):
            days = days[int(pool[3:]):int(pool[3:]) + 1]
        t0 = time.time()
        try:
            f = hk.fit(days, K=K, starts=_starts(K, tr.beta, definition), delta=delta)
        except Exception as e:  # keep the replicate going; record the failure
            rows.append({**common, "definition": definition, "K": K, "delta": delta, "pool": pool,
                         "error": repr(e)[:200]})
            continue
        rows.append({**common, **_row(f, days, definition, K, delta, pool, time.time() - t0)})
        log(f"{tr.sym} {experiment} rep {rep}: {definition} K={K} delta={delta} {pool} "
            f"br {rows[-1]['branching_ratio']:.3f} ({rows[-1]['minutes']} min)")
    return rows


def seed_for(sym, experiment, rep):
    """Reproducible seed per (stock, experiment, replicate), from a hash of all three.

    (An earlier version kept only the first 8 bytes of the key, so every replicate and experiment
    of a stock had the same seed; batch-1 results run before Oct 10, 2026 are affected.)"""
    key = json.dumps([sym, experiment, int(rep)]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)


def load_inputs(simdir, sym):
    simdir = Path(simdir)
    truth = Truth.load(simdir / f"truth_{sym}.npz")
    chains = pd.read_csv(simdir / f"chains_{sym}.csv.gz")
    return truth, BriefStateModel.from_chains(chains)

import math

import numpy as np
import pandas as pd
import pytest

from hawkes_taq import hawkes as hk


def brute_loglik(day, mu_bins, edges, A, beta):
    """O(N^2) log-likelihood straight from the definition. A is (2, 2, K)."""
    ll = 0.0
    for n, (tn, i) in enumerate(zip(day.t, day.types)):
        b = min(np.searchsorted(edges, tn, side="right") - 1, len(mu_bins) - 1)
        lam = mu_bins[b]
        for m in range(n):
            lam += sum(A[i, day.types[m], k] * beta[k] * math.exp(-beta[k] * (tn - day.t[m]))
                       for k in range(len(beta)))
        ll += math.log(lam)
    comp = 2 * np.sum(mu_bins * np.diff(edges))
    for tm, j in zip(day.t, day.types):
        for i in (0, 1):
            comp += sum(A[i, j, k] * (1 - math.exp(-beta[k] * (day.T - tm))) for k in range(len(beta)))
    return ll - comp


def design_loglik(days, mu, alpha, beta, kernel, edges):
    d = hk._design(days, np.asarray(beta), kernel, edges, {})
    lam = mu.ravel()[d.bidx] + d.X @ np.asarray(alpha).ravel()
    return np.log(lam).sum() - d.c_mu @ mu.ravel() - d.c_alpha @ np.asarray(alpha).ravel()


@pytest.fixture
def small_day():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 100, 40))
    return hk.Day(t=t, types=rng.integers(0, 2, 40), T=100.0)


@pytest.mark.parametrize("kernel", ["symmetric", "full"])
def test_design_matches_brute_force(small_day, kernel):
    edges = np.array([0.0, 50.0, 100.0])
    mu = np.array([[0.1, 0.2]])
    beta = [5.0, 0.2]
    if kernel == "symmetric":
        alpha = np.array([[0.3, 0.1], [0.05, 0.2]])
        A = np.array([[alpha[0], alpha[1]], [alpha[1], alpha[0]]])
    else:
        alpha = np.array([[[0.3, 0.1], [0.05, 0.2]], [[0.02, 0.04], [0.25, 0.15]]])
        A = alpha
    expect = brute_loglik(small_day, mu[0], edges, A, beta)
    assert design_loglik([small_day], mu, alpha, beta, kernel, edges) == pytest.approx(expect, rel=1e-9)


def test_residuals_sum_to_compensator(small_day):
    f = hk.HawkesFit(K=1, kernel="symmetric", beta=np.array([2.0]), alpha=np.array([[0.3], [0.1]]),
                     mu=np.array([[0.2]]), edges=np.array([0.0, 100.0]), loglik=0.0,
                     n_events=40, n_params=4)
    up, down = hk.residuals([small_day], f)
    assert len(up) == (small_day.types == 0).sum() and len(down) == (small_day.types == 1).sum()
    assert np.all(up > 0) and np.all(down > 0)
    # Lambda_up at the last up event, computed directly
    i_last = np.flatnonzero(small_day.types == 0)[-1]
    tn = small_day.t[i_last]
    direct = 0.2 * tn + sum(
        (0.3 if j == 0 else 0.1) * (1 - math.exp(-2.0 * (tn - tm)))
        for tm, j in zip(small_day.t[:i_last], small_day.types[:i_last]))
    assert up.sum() == pytest.approx(direct, rel=1e-9)


def test_day_from_events_drops_lock_types():
    ev = pd.DataFrame({"t_fit": [36001.0, 36002.0, 36003.0],
                       "type": ["up", "lock_on", "down"]})
    d = hk.day_from_events(ev)
    assert d.t.tolist() == [1.0, 3.0] and d.types.tolist() == [0, 1]


def simulated_days(n_days=8, seed=1):
    rng = np.random.default_rng(seed)
    edges = np.array([0.0, 2000.0, 4000.0])
    alpha = np.array([[0.25, 0.20], [0.10, 0.15]])  # rows: self, cross; columns: fast, slow
    beta = np.array([200.0, 0.05])
    days = [hk.simulate_day(np.array([0.03, 0.02]), edges, alpha, beta, rng, label=str(d))
            for d in range(n_days)]
    return days, alpha, beta


def test_fit_recovers_two_timescales():
    days, alpha, beta = simulated_days()
    f = hk.fit(days, K=2, bin_minutes=2000 / 60, starts=[[100.0, 0.1], [1000.0, 1.0]])
    assert f.beta[0] == pytest.approx(beta[0], rel=0.3)
    assert f.beta[1] == pytest.approx(beta[1], rel=0.6)
    assert f.branching_ratio == pytest.approx(alpha.sum(), abs=0.15)
    f1 = hk.fit(days, K=1, bin_minutes=2000 / 60, starts=[[200.0], [0.05]])
    assert f.bic < f1.bic


def test_residuals_exponential_under_true_model():
    days, alpha, beta = simulated_days(n_days=4, seed=2)
    f = hk.HawkesFit(K=2, kernel="symmetric", beta=beta, alpha=alpha,
                     mu=np.tile([0.03, 0.02], (len(days), 1)),
                     edges=np.array([0.0, 2000.0, 4000.0]), loglik=0.0, n_events=0, n_params=0)
    for r in hk.ks_residuals(days, f).values():
        assert r.pvalue > 0.01


def test_refit_baseline_matches_fit_on_same_days():
    days, _, _ = simulated_days(n_days=3, seed=3)
    f = hk.fit(days, K=1, bin_minutes=2000 / 60, starts=[[200.0]])
    assert hk.refit_baseline(days, f) == pytest.approx(f.loglik, abs=1e-3)


def test_fit_fixed_matches_fit_at_its_timescales():
    days, _, _ = simulated_days(n_days=4, seed=4)
    f = hk.fit(days, K=2, bin_minutes=2000 / 60, starts=[[100.0, 0.1]])
    g = hk.fit_fixed(days, f.beta, bin_minutes=2000 / 60)
    assert g.loglik == pytest.approx(f.loglik, abs=1e-3)
    assert np.allclose(g.alpha, f.alpha, atol=1e-3)
    assert np.allclose(g.mu, f.mu, rtol=1e-2, atol=1e-5)
    # per-day fits: the day log-likelihoods sum to at least the pooled one
    per_day = [hk.fit_fixed([d], f.beta, bin_minutes=2000 / 60) for d in days]
    assert sum(h.loglik for h in per_day) >= f.loglik - 1e-6
    assert all(h.mu.shape == (1, 2) and h.alpha.shape == (2, 2) for h in per_day)


def test_bootstrap_ks_calibrated_on_true_model():
    days, _, _ = simulated_days(n_days=2, seed=4)
    f = hk.fit(days, K=2, bin_minutes=2000 / 60, starts=[[200.0, 0.05]])
    res = hk.bootstrap_ks(days, f, n_boot=9, seed=1)
    for k in ("up", "down"):
        assert 0 < res[k]["p_boot"] <= 1
        assert res[k]["boot_median"] < res[k]["boot_95"] + 1e-12


def test_bootstrap_replicate_reproducible_and_order_free():
    days, _, _ = simulated_days(n_days=2, seed=5)
    f = hk.fit(days, K=1, bin_minutes=2000 / 60, starts=[[200.0]])
    f3, ks3 = hk.bootstrap_replicate(days, f, seed=7, b=3)
    hk.bootstrap_replicate(days, f, seed=7, b=0)          # another replicate in between
    f3b, ks3b = hk.bootstrap_replicate(days, f, seed=7, b=3)
    assert ks3 == ks3b and np.allclose(f3.beta, f3b.beta) and np.allclose(f3.alpha, f3b.alpha)
    _, ks4 = hk.bootstrap_replicate(days, f, seed=7, b=4)
    assert ks4 != ks3


def brute_loglik_sep(day, mu2, edges, A, beta):
    """O(N^2) log-likelihood with a separate baseline per type; mu2 is (n_bins, 2)."""
    ll = 0.0
    for n, (tn, i) in enumerate(zip(day.t, day.types)):
        b = min(np.searchsorted(edges, tn, side="right") - 1, len(mu2) - 1)
        lam = mu2[b, i]
        for m in range(n):
            lam += sum(A[i, day.types[m], k] * beta[k] * math.exp(-beta[k] * (tn - day.t[m]))
                       for k in range(len(beta)))
        ll += math.log(lam)
    comp = np.sum(mu2 * np.diff(edges)[:, None])
    for tm, j in zip(day.t, day.types):
        for i in (0, 1):
            comp += sum(A[i, j, k] * (1 - math.exp(-beta[k] * (day.T - tm))) for k in range(len(beta)))
    return ll - comp


@pytest.mark.parametrize("kernel", ["symmetric", "full"])
def test_separate_baseline_matches_brute_force(small_day, kernel):
    edges = np.array([0.0, 50.0, 100.0])
    mu2 = np.array([[0.1, 0.3], [0.2, 0.05]])          # (bins, types)
    beta = [5.0, 0.2]
    if kernel == "symmetric":
        alpha = np.array([[0.3, 0.1], [0.05, 0.2]])
        A = np.array([[alpha[0], alpha[1]], [alpha[1], alpha[0]]])
    else:
        alpha = np.array([[[0.3, 0.1], [0.05, 0.2]], [[0.02, 0.04], [0.25, 0.15]]])
        A = alpha
    d = hk._design([small_day], np.asarray(beta), kernel, edges, {}, baseline="separate")
    mu_flat = mu2.ravel()                               # index 2 * bin + type
    lam = mu_flat[d.bidx] + d.X @ alpha.ravel()
    ll = np.log(lam).sum() - d.c_mu @ mu_flat - d.c_alpha @ alpha.ravel()
    assert ll == pytest.approx(brute_loglik_sep(small_day, mu2, edges, A, beta), rel=1e-9)


def test_separate_baseline_fit_and_residuals():
    rng = np.random.default_rng(6)
    edges = np.array([0.0, 2000.0, 4000.0])
    mu2 = np.array([[0.04, 0.02], [0.02, 0.04]])
    alpha = np.array([[0.25, 0.20], [0.10, 0.15]])
    beta = np.array([200.0, 0.05])
    days = [hk.simulate_day(mu2, edges, alpha, beta, rng) for _ in range(6)]
    f = hk.fit(days, K=2, bin_minutes=2000 / 60, starts=[[200.0, 0.05]], baseline="separate")
    assert f.mu.shape == (6, 2, 2)
    est = f.mu.mean(axis=0)
    assert est[0, 0] > est[0, 1] and est[1, 1] > est[1, 0]   # recovers the asymmetry
    f_sh = hk.fit(days, K=2, bin_minutes=2000 / 60, starts=[[200.0, 0.05]])
    assert f.loglik > f_sh.loglik
    for r in hk.ks_residuals(days, f).values():
        assert r.pvalue > 0.01


def test_full_starts_cover_grid_and_ladders():
    s3 = hk.full_starts(3)
    assert len(s3) == 20 + 2 and all(len(b) == 3 for b in s3)
    s4 = hk.full_starts(4)
    assert len(s4) == 15 + 3 and all(np.all(b > 0) for b in s4)
    # a ladder start is passed as decay rates, so its memories come back in seconds
    assert 1 / s3[-1][0] == pytest.approx(0.018)


def test_fit_accepts_full_starts():
    days, _, _ = simulated_days(n_days=1, seed=5)
    f = hk.fit(days, K=1, bin_minutes=2000 / 60, starts=hk.full_starts(1))
    assert f.K == 1 and np.isfinite(f.loglik)


# ---------------------------------------------------------------- refractory period

def gapped_day(seed=0, n=40, T=100.0, delta=0.5):
    """Events at least delta apart, for testing the dead time."""
    rng = np.random.default_rng(seed)
    gaps = delta + rng.exponential(1.0, n)
    t = np.cumsum(gaps)
    t = t[t < T]
    return hk.Day(t=t, types=rng.integers(0, 2, len(t)), T=T)


def brute_loglik_dead(day, mu_bins, edges, A, beta, delta):
    """Log-likelihood with a dead time, compensator by numerical integration outside dead windows."""
    from scipy import integrate

    def lam(i, s):
        b = min(np.searchsorted(edges, s, side="right") - 1, len(mu_bins) - 1)
        v = mu_bins[b]
        for tm, j in zip(day.t, day.types):
            if tm < s:
                v += sum(A[i, j, k] * beta[k] * math.exp(-beta[k] * (s - tm)) for k in range(len(beta)))
        return v

    ll = sum(math.log(lam(i, tn)) for tn, i in zip(day.t, day.types))
    # live intervals: [0, t_1], [t_n + delta, t_{n+1}], ..., [t_N + delta, T]
    starts = np.concatenate([[0.0], day.t + delta])
    ends = np.concatenate([day.t, [day.T]])
    comp = 0.0
    for a, b in zip(starts, ends):
        if b <= a:
            continue
        pts = sorted(set([a, b] + [e for e in edges if a < e < b]))
        for lo, hi in zip(pts[:-1], pts[1:]):
            for i in (0, 1):
                comp += integrate.quad(lambda s: lam(i, s), lo, hi, limit=200, epsabs=1e-11)[0]
    return ll - comp


@pytest.mark.parametrize("kernel", ["symmetric", "full"])
def test_dead_time_likelihood_matches_numerical(kernel):
    day = gapped_day()
    edges = np.array([0.0, 50.0, 100.0])
    mu = np.array([[0.1, 0.2]])
    beta = [5.0, 0.2]
    delta = 0.5
    if kernel == "symmetric":
        alpha = np.array([[0.3, 0.1], [0.05, 0.2]])
        A = np.array([[alpha[0], alpha[1]], [alpha[1], alpha[0]]])
    else:
        alpha = np.array([[[0.3, 0.1], [0.05, 0.2]], [[0.02, 0.04], [0.25, 0.15]]])
        A = alpha
    d = hk._design([day], np.asarray(beta), kernel, edges, {}, delta=delta)
    lam = mu.ravel()[d.bidx] + d.X @ alpha.ravel()
    ll = np.log(lam).sum() - d.c_mu @ mu.ravel() - d.c_alpha @ alpha.ravel()
    assert ll == pytest.approx(brute_loglik_dead(day, mu[0], edges, A, beta, delta), rel=1e-7)


def test_dead_time_simulation_and_fit():
    rng = np.random.default_rng(7)
    edges = np.array([0.0, 2000.0, 4000.0])
    alpha = np.array([[0.25, 0.20], [0.10, 0.15]])
    beta = np.array([100.0, 0.05])
    delta = 0.003
    days = [hk.simulate_day(np.array([0.03, 0.02]), edges, alpha, beta, rng, delta=delta)
            for _ in range(6)]
    assert min(np.diff(d.t).min() for d in days) >= delta - 1e-12
    f = hk.fit(days, K=2, bin_minutes=2000 / 60, starts=[[100.0, 0.05]], delta=delta)
    assert f.delta == delta
    assert f.beta[0] == pytest.approx(beta[0], rel=0.4)
    assert f.branching_ratio == pytest.approx(alpha.sum(), abs=0.15)
    # residuals of the true model with the dead time are Exponential(1)
    true = hk.HawkesFit(K=2, kernel="symmetric", beta=beta, alpha=alpha,
                        mu=np.tile([0.03, 0.02], (len(days), 1)), edges=edges, loglik=0.0,
                        n_events=0, n_params=0, delta=delta)
    for r in hk.ks_residuals(days, true).values():
        assert r.pvalue > 0.01
    # and the model without the dead time fits worse
    f0 = hk.fit(days, K=2, bin_minutes=2000 / 60, starts=[[100.0, 0.05]])
    assert f.loglik > f0.loglik


def test_dead_time_rejects_close_events(small_day):
    with pytest.raises(ValueError):
        hk.fit([small_day], K=1, bin_minutes=None, starts=[[1.0]], delta=5.0)


def test_effective_excitation():
    f = hk.HawkesFit(K=2, kernel="symmetric", beta=np.array([100.0, 1.0]),
                     alpha=np.array([[0.2, 0.1], [0.1, 0.1]]), mu=np.zeros((1, 1)),
                     edges=np.array([0.0, 1.0]), loglik=0.0, n_events=0, n_params=0, delta=0.01)
    assert f.effective_excitation() == pytest.approx([0.3 * math.exp(-1.0), 0.2 * math.exp(-0.01)])


# ---------------------------------------------------------------- simulation study pieces

def test_synthetic_nbbo_roundtrip():
    """Clean moves through the synthetic feed and the real pipeline: E1m recovers them."""
    from hawkes_taq import simulate_nbbo as sn
    from hawkes_taq import simstudy as ss
    rng = np.random.default_rng(3)
    model = sn.BriefStateModel(p_direct=0.1, p_lock=0.8, p_widen=0.1, p_half=0.0,
                               flickers_per_move=0.05, p_flicker_lock=0.5,
                               lock_durations=np.array([0.0004, 0.0007]),
                               widen_durations=np.array([0.00003]),
                               flicker_durations=np.array([0.00005]))
    t = np.cumsum(rng.exponential(5.0, 300))
    t = t[t < 19000]
    y = rng.integers(0, 2, len(t))
    nb = sn.synthetic_nbbo(t, y, model, rng, T=19800.0)
    e1m = ss.days_from_nbbo([nb], "E1m")[0]
    e0 = ss.days_from_nbbo([nb], "E0")[0]
    assert len(e1m.t) == len(t)                        # every clean move recovered, nothing else
    assert np.allclose(e1m.t, t, atol=2e-6) and np.array_equal(e1m.types, y)
    assert len(e0.t) > 1.7 * len(t)                    # brief states roughly double raw events


def test_truth_with_subms():
    from hawkes_taq import simstudy as ss
    tr = ss.Truth("X", np.array([100.0, 1.0]), np.array([[0.1, 0.2], [0.05, 0.1]]),
                  np.zeros((1, 1)), np.array([0.0, 1.0]))
    t2 = tr.with_subms()
    assert len(t2.beta) == 3 and t2.beta[0] == pytest.approx(1 / ss.SUBMS_MEMORY)
    assert t2.branching_ratio == pytest.approx(tr.branching_ratio + ss.SUBMS_EXCITATION)


def test_truth_powerlaw_and_scaling():
    from hawkes_taq import simstudy as ss
    tr = ss.Truth("X", np.array([100.0, 1.0, 0.03]), np.array([[0.05, 0.08, 0.15], [0.03, 0.04, 0.1]]),
                  np.ones((2, 11)), np.linspace(0.0, 19800.0, 12))
    pl = tr.powerlaw()
    assert pl.branching_ratio == pytest.approx(tr.branching_ratio)
    assert len(pl.beta) == 15 and np.all(np.diff(pl.beta) < 0)
    c, th = pl.powerlaw_params
    assert c > 0 and th > 0
    m = [pl.mass_within(h) for h in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)]
    assert np.all(np.diff(m) > 0) and m[-1] <= pl.branching_ratio + 1e-12
    share = tr.alpha[0].sum() / tr.alpha.sum()
    assert pl.alpha[0].sum() / pl.alpha.sum() == pytest.approx(share)
    sc = tr.with_baseline_scale(2.0)
    assert np.allclose(sc.mu, 2.0 * tr.mu) and np.allclose(sc.alpha, tr.alpha)


def test_simulation_plans_keep_batch1():
    from hawkes_taq import simstudy as ss
    assert tuple(k for k in ss.EXPERIMENTS if k in ss.BATCH1) == ss.BATCH1
    assert list(ss.EXPERIMENTS)[:5] == list(ss.BATCH1)       # task order of the first batch
    assert set(ss.BATCH2) <= set(ss.EXPERIMENTS)


def test_seed_for_distinct():
    from hawkes_taq import simstudy as ss
    keys = [(s, e, r) for s in ("INTC", "MDLZ") for e in ("base", "share0", "subms") for r in range(20)]
    seeds = [ss.seed_for(*k) for k in keys]
    assert len(set(seeds)) == len(seeds)
    assert ss.seed_for("INTC", "base", 3) == ss.seed_for("INTC", "base", 3)


def brute_loglik_burn(day, mu_bins, edges, A, beta, burn):
    """Log-likelihood scored on [burn, T] with the full history: log-sum over events at or after
    burn, compensator (numerical) over [burn, T]."""
    from scipy import integrate

    def lam(i, s):
        b = min(np.searchsorted(edges, s, side="right") - 1, len(mu_bins) - 1)
        v = mu_bins[b]
        for tm, j in zip(day.t, day.types):
            if tm < s:
                v += sum(A[i, j, k] * beta[k] * math.exp(-beta[k] * (s - tm)) for k in range(len(beta)))
        return v

    ll = sum(math.log(lam(i, tn)) for tn, i in zip(day.t, day.types) if tn >= burn)
    pts = sorted(set([burn, day.T] + [e for e in edges if burn < e < day.T] + [tn for tn in day.t if tn > burn]))
    comp = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        for i in (0, 1):
            comp += integrate.quad(lambda s: lam(i, s), lo, hi, limit=200, epsabs=1e-11)[0]
    return ll - comp


@pytest.mark.parametrize("kernel", ["symmetric", "full"])
def test_burn_in_likelihood_matches_numerical(small_day, kernel):
    edges = np.array([0.0, 50.0, 100.0])
    mu = np.array([[0.1, 0.2]])
    beta = [5.0, 0.2]
    burn = 20.0
    if kernel == "symmetric":
        alpha = np.array([[0.3, 0.1], [0.05, 0.2]])
        A = np.array([[alpha[0], alpha[1]], [alpha[1], alpha[0]]])
    else:
        alpha = np.array([[[0.3, 0.1], [0.05, 0.2]], [[0.02, 0.04], [0.25, 0.15]]])
        A = alpha
    d = hk._design([small_day], np.asarray(beta), kernel, edges, {}, burn=burn)
    assert d.X.shape[0] == int(np.sum(small_day.t >= burn))
    lam = mu.ravel()[d.bidx] + d.X @ alpha.ravel()
    ll = np.log(lam).sum() - d.c_mu @ mu.ravel() - d.c_alpha @ alpha.ravel()
    assert ll == pytest.approx(brute_loglik_burn(small_day, mu[0], edges, A, beta, burn), rel=1e-7)
    # burn = 0 is the plain likelihood
    d0 = hk._design([small_day], np.asarray(beta), kernel, edges, {}, burn=0.0)
    ll0 = np.log(mu.ravel()[d0.bidx] + d0.X @ alpha.ravel()).sum() - d0.c_mu @ mu.ravel() - d0.c_alpha @ alpha.ravel()
    assert ll0 == pytest.approx(brute_loglik(small_day, mu[0], edges, A, beta), rel=1e-9)


def test_burn_in_fit_and_residuals():
    rng = np.random.default_rng(5)
    edges = np.array([0.0, 1800.0, 3600.0])
    days = [hk.simulate_day(np.array([0.05, 0.08]), edges, np.array([[0.3], [0.2]]), np.array([0.5]), rng)
            for _ in range(6)]
    f = hk.fit(days, K=1, bin_minutes=30, starts=[[0.5]], burn=120.0)
    assert f.burn == 120.0 and f.n_events == sum(int(np.sum(d.t >= 120.0)) for d in days)
    res = hk.residuals(days, f)
    assert sum(len(r) for r in res) == f.n_events - 2 * len(days)
    for r in res:
        assert abs(r.mean() - 1.0) < 0.1

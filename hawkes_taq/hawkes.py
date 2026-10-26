"""Two-dimensional (up/down) Hawkes model with a sum of K exponential kernels.

Intensity of type i (0 = up, 1 = down) on day d at time t (seconds from session start):

    lambda_i(t) = mu_{d, b(t)} + sum_k beta_k * sum_{m: t_m < t}
                      alpha_{i, j_m, k} * exp(-beta_k (t - t_m))

- mu_{d,b}: baseline, one level per day and time-of-day bin (shared by up and down).
- alpha_{i,j,k}: expected number of type-i events triggered by one type-j event through
  timescale k (the integral of that kernel component). With kernel="symmetric",
  alpha_{i,j,k} is alpha_self_k when i == j and alpha_cross_k otherwise.
- beta_k: decay rates, shared by all pairs, sorted in decreasing order (fastest first).

Days are independent: excitation does not carry over between sessions.

Optional refractory period (dead time) delta >= 0: after every event, of either type, both
intensities are zero for delta seconds,

    lambda_i(t) = 1{t - t_last >= delta} * [mu + excitation as above].

E1m merges every state shorter than 1 ms, so its events are never closer than 1 ms; delta =
0.001 makes the model consistent with that. The log-intensity terms are unchanged (events only
occur after the dead time) and the compensator loses the integral over each dead window, which
is linear in (mu, alpha) with a closed form, so the inner problem stays concave. alpha keeps its
meaning as the integral of the kernel; the excitation actually realized by an isolated event is
alpha_k * exp(-beta_k * delta) (see HawkesFit.effective_excitation).

Estimation profiles out the decay rates. For fixed beta the log-likelihood is concave in
(mu, alpha) >= 0, which is solved with L-BFGS-B; the outer search over log(beta) uses
Nelder-Mead from several starts.
"""

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

SESSION_START = 10 * 3600          # 10:00
SESSION_END = 15 * 3600 + 30 * 60  # 15:30
TYPES = {"up": 0, "down": 1}


# ---------------------------------------------------------------- data

@dataclass
class Day:
    """One session: event times (seconds from session start, sorted, no ties) and types."""
    t: np.ndarray
    types: np.ndarray
    T: float
    label: str = ""


def day_from_events(events, start=SESSION_START, end=SESSION_END, label=""):
    """Build a Day from an E0 or E1 table (columns t_fit, type in {'up', 'down'})."""
    ev = events[events["type"].isin(TYPES)].sort_values("t_fit")
    t = ev["t_fit"].to_numpy(dtype=float) - start
    return Day(t=t, types=ev["type"].map(TYPES).to_numpy(dtype=np.int64), T=float(end - start),
               label=label)


def load_days(event_dir, sym, definition="E1", dates=None):
    """Read saved events (pipeline output) for one stock into a list of Days."""
    from pathlib import Path
    files = sorted(Path(event_dir, definition, sym).glob("*.csv.gz"))
    if dates is not None:
        files = [f for f in files if f.name.split(".")[0] in set(dates)]
    return [day_from_events(pd.read_csv(f), label=f.name.split(".")[0]) for f in files]


# ---------------------------------------------------------------- features

def _decay_sums_py(t, types, beta, T):
    """For one decay rate: S[n, j] = sum_{m < n, type j} exp(-beta (t_n - t_m)), and the same
    sums at the session end T."""
    S = np.empty((len(t), 2))
    s0 = s1 = 0.0
    prev = 0.0
    exp = math.exp
    for n in range(len(t)):
        d = exp(-beta * (t[n] - prev))
        s0 *= d
        s1 *= d
        S[n, 0] = s0
        S[n, 1] = s1
        if types[n] == 0:
            s0 += 1.0
        else:
            s1 += 1.0
        prev = t[n]
    d = exp(-beta * (T - prev))
    return S, np.array([s0 * d, s1 * d])


try:  # compiled version of the same loop, much faster on long event series
    import numba

    _decay_sums_nb = numba.njit(cache=True)(_decay_sums_py)

    def _decay_sums(t, types, beta, T):
        return _decay_sums_nb(np.ascontiguousarray(t, dtype=np.float64),
                              np.ascontiguousarray(types, dtype=np.int64), float(beta), float(T))
except ImportError:
    _decay_sums = _decay_sums_py


@dataclass
class _Design:
    """Everything the inner (fixed-beta) problem needs, stacked over days."""
    X: np.ndarray          # (N, P_alpha): lambda_n = mu[bidx[n]] + X[n] @ alpha
    bidx: np.ndarray       # (N,) baseline index of each event
    c_alpha: np.ndarray    # (P_alpha,) compensator coefficients of alpha
    c_mu: np.ndarray       # (P_mu,) compensator coefficients of mu
    S: list = field(default_factory=list)  # per-day (decay sums at events, at end), for residuals


def _bins(T, bin_minutes):
    if bin_minutes is None:
        return np.array([0.0, T])
    edges = np.arange(0.0, T, 60.0 * bin_minutes)
    return np.append(edges, T)


def _dead_windows(day, delta):
    """Length of the dead window after each event: delta, cut at the next event and at T."""
    if delta <= 0 or len(day.t) == 0:
        return np.zeros(len(day.t))
    nxt = np.append(day.t[1:], day.T)
    return np.minimum(delta, nxt - day.t)


def _bin_overlap(edges, a, w):
    """(n, n_bins) overlap of each interval [a_n, a_n + w_n) with the baseline bins."""
    lo = np.clip(a[:, None], edges[None, :-1], edges[None, 1:])
    hi = np.clip((a + w)[:, None], edges[None, :-1], edges[None, 1:])
    return hi - lo


def check_dead_time(days, delta, tol=2e-6):
    """Raise if any two consecutive events are closer than delta (minus a tie tolerance)."""
    for day in days:
        if len(day.t) > 1 and np.min(np.diff(day.t)) < delta - tol:
            raise ValueError(f"day {day.label}: events closer than the dead time {delta} s; "
                             "use delta=0 or an event definition that merges short states")


def _design(days, beta, kernel, edges, cache, max_cache=None, baseline="shared", delta=0.0,
            burn=0.0):
    """burn > 0: events before `burn` seconds are history only (they excite later events but
    are not scored), and the compensator runs over [burn, T]."""
    K = len(beta)
    nb = len(edges) - 1
    X_parts, b_parts, S_all = [], [], []
    c_alpha = np.zeros(2 * K if kernel == "symmetric" else 4 * K)
    n_types = 2 if baseline == "separate" else 1
    c_mu = np.zeros(nb * len(days) * n_types)
    for d, day in enumerate(days):
        sums = []
        for b in beta:
            key = (d, round(float(b), 12))
            if key not in cache:
                cache[key] = _decay_sums(day.t, day.types, float(b), day.T)
            sums.append(cache[key])
        if max_cache is not None:
            # keep only the most recent entries; the search revisits recent decay rates only
            while len(cache) > max_cache:
                cache.pop(next(iter(cache)))
        S = np.stack([s[0] for s in sums], axis=2)      # (n, 2 [j], K)
        S_end = np.stack([s[1] for s in sums], axis=1)  # (2 [j], K)
        S_all.append((S, S_end))
        own = S[np.arange(len(day.t)), day.types, :] * beta        # (n, K)
        other = S[np.arange(len(day.t)), 1 - day.types, :] * beta  # (n, K)
        counts = np.bincount(day.types, minlength=2).astype(float)
        # integral over [0, T] of the kernel from each event j, summed: N_j - S_end[j]
        comp = counts[:, None] - S_end                              # (2 [j], K)
        scored = day.t >= burn
        if burn > 0:
            # remove the part of each early event's kernel that falls before the burn-in end
            pre = day.t < burn
            for j in (0, 1):
                tm = day.t[pre & (day.types == j)]
                comp[j] -= np.sum(1.0 - np.exp(-np.outer(burn - tm, beta)), axis=0)
        if kernel == "symmetric":
            X_parts.append(np.hstack([own, other])[scored])
            # each event of type j excites type j (self) and type 1-j (cross) equally
            c_alpha[:K] += comp.sum(axis=0)
            c_alpha[K:] += comp.sum(axis=0)
        else:
            # columns: alpha[i, j, k] flattened as (i, j, k)
            n = len(day.t)
            Xd = np.zeros((n, 4 * K))
            for i in (0, 1):
                rows = day.types == i
                for j in (0, 1):
                    cols = slice((2 * i + j) * K, (2 * i + j + 1) * K)
                    Xd[rows, cols] = S[rows, j, :] * beta
                    c_alpha[cols] += comp[j]
            X_parts.append(Xd[scored])
        if delta > 0 and len(day.t):
            # subtract the integral of the intensity over each dead window [t_n, t_n + w_n):
            # excitation just after event n is S[n] plus the event itself
            w = _dead_windows(day, delta) * scored                 # windows before the burn-in end are not scored
            post = S.copy()                                        # (n, 2 [j], K)
            post[np.arange(len(day.t)), day.types, :] += 1.0
            frac = 1.0 - np.exp(-np.outer(w, beta))               # (n, K)
            dead_j = np.einsum("njk,nk->jk", post, frac)          # (2 [j], K)
            if kernel == "symmetric":
                c_alpha[:K] -= dead_j.sum(axis=0)
                c_alpha[K:] -= dead_j.sum(axis=0)
            else:
                for i in (0, 1):
                    for j in (0, 1):
                        c_alpha[(2 * i + j) * K:(2 * i + j + 1) * K] -= dead_j[j]
            ov = _bin_overlap(edges, day.t, w).sum(axis=0)         # (n_bins,)
            if baseline == "separate":
                c_mu[2 * d * nb:2 * (d + 1) * nb] -= np.repeat(ov, 2)
            else:
                c_mu[d * nb:(d + 1) * nb] -= 2.0 * ov
        cell = d * nb + np.clip(np.searchsorted(edges, day.t, side="right") - 1, 0, nb - 1)
        live = _bin_overlap(edges, np.array([burn]), np.array([day.T - burn]))[0]  # bin lengths in [burn, T]
        if baseline == "separate":
            # one baseline per (day, bin, type), stored at index 2 * cell + type
            b_parts.append((2 * cell + day.types)[scored])
            c_mu[2 * d * nb:2 * (d + 1) * nb] += np.repeat(live, 2)
        else:
            b_parts.append(cell[scored])
            c_mu[d * nb:(d + 1) * nb] += 2.0 * live  # shared by both types
    return _Design(np.vstack(X_parts), np.concatenate(b_parts), c_alpha, c_mu, S_all)


# ---------------------------------------------------------------- inner problem

def _inner(design, x0=None):
    """Maximize sum log(mu[bidx] + X alpha) - c_mu.mu - c_alpha.alpha over mu, alpha >= 0."""
    P_mu, P_a = len(design.c_mu), design.X.shape[1]
    if x0 is None:
        n_per = np.bincount(design.bidx, minlength=P_mu)
        x0 = np.concatenate([np.maximum(n_per / design.c_mu, 1e-6) * 0.5, np.full(P_a, 0.05)])

    def f(x):
        mu, a = x[:P_mu], x[P_mu:]
        lam = mu[design.bidx] + design.X @ a
        if np.any(lam <= 0):
            return np.inf, np.zeros_like(x)
        w = 1.0 / lam
        ll = np.log(lam).sum() - design.c_mu @ mu - design.c_alpha @ a
        g_mu = np.bincount(design.bidx, weights=w, minlength=P_mu) - design.c_mu
        g_a = design.X.T @ w - design.c_alpha
        return -ll, -np.concatenate([g_mu, g_a])

    bounds = [(1e-10, None)] * P_mu + [(0.0, None)] * P_a
    res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    return -res.fun, res.x


# ---------------------------------------------------------------- fit

@dataclass
class HawkesFit:
    K: int
    kernel: str
    beta: np.ndarray
    alpha: np.ndarray        # symmetric: (2, K) rows self, cross; full: (2, 2, K) [i, j, k]
    mu: np.ndarray           # (n_days, n_bins); with baseline="separate", (n_days, n_bins, 2)
    edges: np.ndarray
    loglik: float
    n_events: int
    n_params: int
    baseline: str = "shared"
    delta: float = 0.0
    burn: float = 0.0

    def mu_by_type(self, d):
        """Baseline of day d as an (n_bins, 2) array: column 0 up, column 1 down."""
        m = self.mu[d]
        return np.stack([m, m], axis=1) if m.ndim == 1 else m

    @property
    def aic(self):
        return 2 * self.n_params - 2 * self.loglik

    @property
    def bic(self):
        return self.n_params * math.log(self.n_events) - 2 * self.loglik

    @property
    def branching_ratio(self):
        """Spectral radius of the 2x2 matrix of total excitation."""
        A = self.alpha_matrix().sum(axis=2)
        return float(max(abs(np.linalg.eigvals(A))))

    def alpha_matrix(self):
        """alpha as a (2, 2, K) array [i, j, k]."""
        if self.kernel == "symmetric":
            s, c = self.alpha
            return np.array([[s, c], [c, s]])
        return self.alpha

    def timescales(self):
        """Mean memory 1/beta_k in seconds."""
        return 1.0 / self.beta

    def effective_excitation(self):
        """Excitation per scale actually realized by an isolated event: (a_k + c_k) times the
        share of the kernel's mass after the dead time, exp(-beta_k delta). Equals the
        excitation per scale when delta = 0."""
        A = self.alpha_matrix()
        return A[0].sum(axis=0) * np.exp(-self.beta * self.delta)

    def summary(self):
        A = self.alpha_matrix()
        rows = [{"k": k, "beta": self.beta[k], "memory_s": 1 / self.beta[k],
                 "alpha_up_up": A[0, 0, k], "alpha_up_down": A[0, 1, k],
                 "alpha_down_up": A[1, 0, k], "alpha_down_down": A[1, 1, k]}
                for k in range(self.K)]
        return pd.DataFrame(rows)


_GRID = [1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2]   # decay rates: memories of 1 ms to 100 s

# Timescale ladders seen in the E1m fits (memories in seconds), used as extra starting points:
# the two triples that K = 3 switches between, and the four-scale ladder of the K = 4 fits.
_LADDERS = {3: [[0.008, 0.09, 3.0], [0.018, 1.0, 27.0]],
            4: [[0.012, 0.08, 1.7, 26.0], [0.016, 0.13, 2.7, 46.0], [0.007, 0.05, 0.9, 11.0]]}


def _default_starts(K):
    """Six decay-rate combinations spread through the grid (the original, thin set)."""
    combos = list(itertools.combinations(_GRID, K))
    idx = np.unique(np.linspace(0, len(combos) - 1, min(6, len(combos))).round().astype(int))
    return [np.array(combos[i]) for i in idx]


def full_starts(K):
    """Every grid combination plus the empirical ladders: 20 + 2 starts for K = 3, 15 + 3 for K = 4.

    The profile likelihood in the decay rates can have several local optima (for example MDLZ,
    March 2017, two optima 63 log-likelihood units apart), and the default six starts do not
    always reach the best one.
    """
    starts = [np.array(c) for c in itertools.combinations(_GRID, K)]
    starts += [1.0 / np.array(m) for m in _LADDERS.get(K, [])]
    return starts


def fit(days, K=1, kernel="symmetric", bin_minutes=30, starts=None, beta_bounds=(1e-4, 1e5),
        verbose=False, baseline="shared", delta=0.0, burn=0.0):
    """Fit the model to a list of Days by maximum likelihood (pooled over days).

    delta > 0 adds a refractory period of delta seconds after every event (module docstring).
    burn > 0 scores only the events from `burn` seconds after the session start; earlier events
    are history (they excite later ones), which removes the cold start at the window's open.
    """
    if delta > 0:
        check_dead_time(days, delta)
    if kernel not in ("symmetric", "full"):
        raise ValueError("kernel must be 'symmetric' or 'full'")
    if baseline not in ("shared", "separate"):
        raise ValueError("baseline must be 'shared' or 'separate'")
    T = days[0].T
    edges = _bins(T, bin_minutes)
    cache = {}
    lo, hi = np.log(beta_bounds)
    best = {"ll": -np.inf}

    def profile(u):
        if np.any(u < lo) or np.any(u > hi):
            return np.inf
        beta = np.sort(np.exp(u))[::-1]
        design = _design(days, beta, kernel, edges, cache, max_cache=4 * K * len(days),
                         baseline=baseline, delta=delta, burn=burn)
        ll, x = _inner(design)
        if ll > best["ll"]:
            best.update(ll=ll, beta=beta, x=x)
        return -ll

    for b0 in (starts if starts is not None else _default_starts(K)):
        res = optimize.minimize(profile, np.log(np.asarray(b0, dtype=float)), method="Nelder-Mead",
                                options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400 * K})
        if verbose:
            print(f"start {np.round(b0, 4)} -> beta {np.round(np.sort(np.exp(res.x))[::-1], 4)}, "
                  f"loglik {-res.fun:.3f}")

    beta, x = best["beta"], best["x"]
    nb = len(edges) - 1
    P_mu = nb * len(days) * (2 if baseline == "separate" else 1)
    mu, a = x[:P_mu], x[P_mu:]
    mu = mu.reshape(len(days), nb, 2) if baseline == "separate" else mu.reshape(len(days), nb)
    alpha = a.reshape(2, K) if kernel == "symmetric" else a.reshape(2, 2, K)
    n_events = sum(int(np.sum(d.t >= burn)) for d in days)
    n_params = P_mu + len(a) + K
    return HawkesFit(K, kernel, beta, alpha, mu, edges, best["ll"], n_events, n_params, baseline,
                     delta, burn)


def refit_baseline(days, fit_):
    """Log-likelihood on new days with the kernel fixed and only the baselines re-estimated.

    Used for out-of-sample comparison: a kernel fitted on some days is scored on others.
    """
    design = _design(days, fit_.beta, fit_.kernel, fit_.edges, {}, baseline=fit_.baseline,
                     delta=fit_.delta, burn=fit_.burn)
    a = fit_.alpha.ravel()
    off = design.X @ a
    c_off = design.c_alpha @ a
    P_mu = len(design.c_mu)

    def f(mu):
        lam = mu[design.bidx] + off
        w = 1.0 / lam
        ll = np.log(lam).sum() - design.c_mu @ mu - c_off
        return -ll, -(np.bincount(design.bidx, weights=w, minlength=P_mu) - design.c_mu)

    n_per = np.bincount(design.bidx, minlength=P_mu)
    x0 = np.maximum(n_per / design.c_mu, 1e-6) * 0.5
    res = optimize.minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(1e-10, None)] * P_mu)
    return -res.fun


def fit_fixed(days, beta, kernel="symmetric", bin_minutes=30, baseline="shared", delta=0.0,
              burn=0.0):
    """Fit with the decay rates held fixed: baselines and excitation weights by maximum
    likelihood, one call to the concave inner problem. Used for day-level fits with a month's
    timescales (the kernel's weights of one day on the month's components)."""
    if delta > 0:
        check_dead_time(days, delta)
    beta = np.sort(np.asarray(beta, dtype=float))[::-1]
    K = len(beta)
    T = days[0].T
    edges = _bins(T, bin_minutes)
    design = _design(days, beta, kernel, edges, {}, baseline=baseline, delta=delta, burn=burn)
    ll, x = _inner(design)
    nb = len(edges) - 1
    P_mu = nb * len(days) * (2 if baseline == "separate" else 1)
    mu, a = x[:P_mu], x[P_mu:]
    mu = mu.reshape(len(days), nb, 2) if baseline == "separate" else mu.reshape(len(days), nb)
    alpha = a.reshape(2, K) if kernel == "symmetric" else a.reshape(2, 2, K)
    n_events = sum(int(np.sum(d.t >= burn)) for d in days)
    return HawkesFit(K, kernel, beta, alpha, mu, edges, ll, n_events, P_mu + len(a), baseline,
                     delta, burn)


# ---------------------------------------------------------------- diagnostics

def residuals(days, fit_):
    """Time-rescaled residuals: compensator increments between consecutive events of each type.

    Returns a list of two arrays (up, down) pooled over days; under the model they are
    i.i.d. Exponential(1). With a burn-in, the increments of events before it and the first
    increment after it (which spans the burn-in end) are left out.
    """
    design = _design(days, fit_.beta, fit_.kernel, fit_.edges, {}, baseline=fit_.baseline,
                     delta=fit_.delta, burn=fit_.burn)
    A = fit_.alpha_matrix()            # (2, 2, K)
    out = [[], []]
    for d, (day, (S, _)) in enumerate(zip(days, design.S)):
        mu2 = fit_.mu_by_type(d)       # (n_bins, 2)
        b = np.clip(np.searchsorted(fit_.edges, day.t, side="right") - 1, 0, len(mu2) - 1)
        counts = np.stack([np.cumsum(day.types == j) - (day.types == j) for j in (0, 1)], axis=1)
        for i in (0, 1):
            mu = mu2[:, i]
            cum_mu = np.concatenate([[0.0], np.cumsum(mu * np.diff(fit_.edges))])
            base = cum_mu[b] + mu[b] * (day.t - fit_.edges[b])
            # Lambda_i(t_n) = int mu_i + sum_{j,k} alpha[i,j,k] (N_j(t_n-) - S[n, j, k])
            lam_int = base + np.einsum("jk,njk->n", A[i], counts[:, :, None] - S)
            if fit_.delta > 0:
                # remove the intensity over the dead windows of earlier events
                w = _dead_windows(day, fit_.delta)
                post = S.copy()
                post[np.arange(len(day.t)), day.types, :] += 1.0
                frac = 1.0 - np.exp(-np.outer(w, fit_.beta))
                dead = (_bin_overlap(fit_.edges, day.t, w) @ mu
                        + np.einsum("jk,njk,nk->n", A[i], post, frac))
                lam_int = lam_int - np.concatenate([[0.0], np.cumsum(dead)[:-1]])
            at = lam_int[day.types == i]
            inc = np.diff(np.concatenate([[0.0], at]))
            if fit_.burn > 0:
                idx = np.where(day.t[day.types == i] >= fit_.burn)[0]
                inc = inc[idx[1:]]
            out[i].append(inc)
    return [np.concatenate(r) for r in out]


def ks_residuals(days, fit_):
    """Kolmogorov-Smirnov statistics and p-values of the residuals against Exponential(1).

    The p-values assume known parameters, so they are only indicative; use a parametric
    bootstrap (simulate + refit) for calibrated tests.
    """
    res = residuals(days, fit_)
    return {name: stats.kstest(r, "expon") for name, r in zip(("up", "down"), res)}


# ---------------------------------------------------------------- simulation

def simulate_day(mu_bins, edges, alpha, beta, rng, kernel="symmetric", label="", delta=0.0):
    """Simulate one session by Ogata thinning.

    mu_bins: baseline per bin, shape (n_bins,) shared by both types or (n_bins, 2) per type.
    alpha as in HawkesFit.alpha. delta: refractory period after every event (seconds).
    """
    beta = np.asarray(beta, dtype=float)
    mu2 = np.asarray(mu_bins, dtype=float)
    if mu2.ndim == 1:
        mu2 = np.stack([mu2, mu2], axis=1)
    K = len(beta)
    A = (np.array([[alpha[0], alpha[1]], [alpha[1], alpha[0]]]) if kernel == "symmetric"
         else np.asarray(alpha))                     # (2, 2, K)
    T = edges[-1]
    exc = np.zeros((2, K))  # exc[i, k] = beta_k * sum_m alpha[i, j_m, k] exp(-beta_k (t - t_m))
    t, out_t, out_y = 0.0, [], []
    while True:
        b = min(np.searchsorted(edges, t, side="right") - 1, len(mu2) - 1)
        # intensity is non-increasing until the next bin edge, so its current value bounds it
        lam_bar = mu2[b].sum() + exc.sum()
        w = rng.exponential(1.0 / lam_bar)
        nxt = edges[b + 1]
        if t + w >= nxt:
            exc *= np.exp(-beta * (nxt - t))[None, :]
            t = nxt
            if t >= T:
                break
            continue
        exc *= np.exp(-beta * w)[None, :]
        t += w
        lam = mu2[b] + exc.sum(axis=1)
        u = rng.uniform(0.0, lam_bar)
        if u < lam.sum():
            i = 0 if u < lam[0] else 1
            out_t.append(t)
            out_y.append(i)
            exc += A[:, i, :] * beta[None, :]
            if delta > 0:
                # no events during the dead time: let the excitation decay through it
                jump = min(delta, T - t)
                exc *= np.exp(-beta * jump)[None, :]
                t += jump
                if t >= T:
                    break
    return Day(t=np.array(out_t), types=np.array(out_y, dtype=np.int64), T=float(T), label=label)


# ---------------------------------------------------------------- bootstrap

def simulate_like(days, fit_, rng):
    """Simulate one new set of days from a fitted model (same days, bins and baselines)."""
    return [simulate_day(fit_.mu[d], fit_.edges, fit_.alpha, fit_.beta, rng, kernel=fit_.kernel,
                         label=day.label, delta=fit_.delta) for d, day in enumerate(days)]


def bootstrap_replicate(days, fit_, seed, b):
    """One parametric-bootstrap replicate: simulate from the fit, refit (starting from the
    fitted decay rates) and compute the KS statistics. Replicate b has its own random stream
    (seed, b), so replicates can run in any order, in parallel, or be resumed.
    Returns (refit, {"up": KS statistic, "down": KS statistic})."""
    rng = np.random.default_rng([seed, b])
    bins = len(fit_.edges) - 1
    bin_minutes = None if bins == 1 else float(fit_.edges[1] - fit_.edges[0]) / 60.0
    sim = simulate_like(days, fit_, rng)
    f_b = fit(sim, K=fit_.K, kernel=fit_.kernel, bin_minutes=bin_minutes, starts=[fit_.beta],
              baseline=fit_.baseline, delta=fit_.delta)
    return f_b, {k: r.statistic for k, r in ks_residuals(sim, f_b).items()}


def bootstrap_pvalues(obs, boot):
    """Bootstrap p-values from observed KS statistics obs = {"up": x, "down": y} and replicate
    statistics boot = {"up": [...], "down": [...]}."""
    out = {}
    for k in ("up", "down"):
        bk = np.asarray(boot[k])
        out[k] = {"ks": obs[k], "p_boot": (1 + np.sum(bk >= obs[k])) / (1 + len(bk)),
                  "boot_median": float(np.median(bk)), "boot_95": float(np.quantile(bk, 0.95))}
    return out


def bootstrap_ks(days, fit_, n_boot=99, seed=0, verbose=False):
    """Parametric bootstrap of the residual KS statistics.

    Each replicate simulates data from the fitted model, refits it (starting from the fitted
    decay rates) and recomputes the KS statistics. The p-value of each observed statistic is
    the share of replicates with a statistic at least as large, so the estimation of the
    parameters is accounted for (Kling & Vetter, 2025; Cavaliere et al., 2023).
    """
    obs = {k: r.statistic for k, r in ks_residuals(days, fit_).items()}
    boot = {"up": [], "down": []}
    for b in range(n_boot):
        _, ks = bootstrap_replicate(days, fit_, seed, b)
        for k in boot:
            boot[k].append(ks[k])
        if verbose:
            print(f"replicate {b + 1}/{n_boot}", flush=True)
    return bootstrap_pvalues(obs, boot)

"""Synthetic consolidated NBBO from a clean price process (observation layer for the simulation).

A clean process gives the times and directions of real price moves (for example simulated from
a fitted E1m Hawkes model). This module turns them into a sequence of NBBO quotes (t, bid, ask)
with the brief states seen in consolidated data, so the real event pipeline
(events.build_states -> E0, E1, E1m) can be run on it and its output compared with the truth.

Each clean move at time t is drawn as one of:

- direct: both quotes move at t (one NBBO change);
- via lock: for an up move the bid rises to the old ask first, the market is locked for a
  duration L, and the ask rises at t (two NBBO changes, both moving the mid up half a tick);
  mirror image for a down move;
- via widening: for an up move the ask rises first (spread 2 ticks) for L, then the bid at t;
- half-tick move: only one quote moves (spread widens from 1 to 2 ticks, or narrows back).

Moves from a two-tick spread are half-tick moves that narrow it. From a one-tick spread the
path is drawn with the calibrated probabilities. Flickers, brief states that return to the
previous quotes, are added at a calibrated rate per clean move.

Brief-state durations are resampled from the calibrated empirical distribution (seconds). A
brief state never starts before the previous NBBO change: if the clean gap is shorter than the
drawn duration, the brief state starts just after the previous change, which is how close real
moves interact with the consolidation.

Kept simple on purpose (one brief state per move, no multi-state chains, no long locks kept as
states); differences from the data are checked by comparing simulated and real E0 statistics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BriefStateModel:
    """Calibrated observation-layer parameters for one stock-month."""
    p_direct: float
    p_lock: float
    p_widen: float
    p_half: float                 # share of moves from a one-tick spread that are half-tick
    flickers_per_move: float
    p_flicker_lock: float         # share of flickers that are locks (rest: widenings)
    lock_durations: np.ndarray    # seconds, resampled
    widen_durations: np.ndarray
    flicker_durations: np.ndarray

    def scaled(self, artifact_scale=1.0, duration_scale=1.0):
        """Variant with the brief-state probabilities scaled (direct takes the rest) and/or the
        durations stretched. artifact_scale=0 gives a feed with no brief states."""
        lock, widen = self.p_lock * artifact_scale, self.p_widen * artifact_scale
        return BriefStateModel(1.0 - lock - widen, lock, widen, self.p_half,
                               self.flickers_per_move * artifact_scale, self.p_flicker_lock,
                               self.lock_durations * duration_scale,
                               self.widen_durations * duration_scale,
                               self.flicker_durations * duration_scale)

    @classmethod
    def from_chains(cls, chains):
        """Build from a chain table written by scripts/calibrate_brief_states.py --durations."""
        mv = chains[chains.what == "move"]
        one = mv[mv.spread_before == 1]
        full = one[one.d_mid2.abs() == 2]
        n = len(full)
        fl = chains[chains.what == "flicker"]

        def durations(df, kind):
            d = df.loc[df.kind == kind, "chain_s"].to_numpy()
            return d if len(d) else np.array([0.0005])

        return cls(p_direct=(full.kind == "direct").sum() / n, p_lock=(full.kind == "lock").sum() / n,
                   p_widen=(full.kind.isin(["widen", "other"])).sum() / n,
                   p_half=(one.d_mid2.abs() == 1).mean(),
                   flickers_per_move=len(fl) / len(mv), p_flicker_lock=(fl.kind == "lock").mean()
                   if len(fl) else 1.0,
                   lock_durations=durations(full, "lock"), widen_durations=durations(full, "widen"),
                   flicker_durations=fl["chain_s"].to_numpy() if len(fl) else np.array([0.0005]))


def synthetic_nbbo(t, types, model, rng, start=36000.0, T=None, price_ticks=3500):
    """Quote sequence for one day.

    t, types: clean move times (seconds from session start, sorted) and directions (0 up,
    1 down). Returns a DataFrame with columns t (seconds after midnight), bid, ask (dollars),
    sorted by time, starting with the opening quote at the session start.
    """
    bid, ask = price_ticks, price_ticks + 1
    rows = [(0.0, bid, ask)]
    last = 0.0
    for tn, y in zip(t, types):
        up = y == 0
        gap = tn - last
        if ask - bid >= 2:                                   # narrow a two-tick spread
            if up:
                bid += 1
            else:
                ask -= 1
            rows.append((tn, bid, ask))
        elif rng.uniform() < model.p_half:                   # widen by half a tick
            if up:
                ask += 1
            else:
                bid -= 1
            rows.append((tn, bid, ask))
        else:
            u = rng.uniform()
            if u < model.p_direct:
                bid, ask = (bid + 1, ask + 1) if up else (bid - 1, ask - 1)
                rows.append((tn, bid, ask))
            else:
                via_lock = u < model.p_direct + model.p_lock
                pool = model.lock_durations if via_lock else model.widen_durations
                L = float(rng.choice(pool))
                t0 = tn - L if L < gap else last + 0.5 * gap  # never before the previous change
                if via_lock:   # the quote moving toward the other side goes first
                    mid = (bid + 1, ask) if up else (bid, ask - 1)
                else:          # the quote moving away goes first
                    mid = (bid, ask + 1) if up else (bid - 1, ask)
                rows.append((t0, *mid))
                bid, ask = (bid + 1, ask + 1) if up else (bid - 1, ask - 1)
                rows.append((tn, bid, ask))
        last = tn
    df = pd.DataFrame(rows, columns=["t", "bid", "ask"])
    df = _add_flickers(df, model, rng, T if T is not None else (t[-1] + 1.0 if len(t) else 1.0))
    df["t"] = df["t"] + start
    df["bid"] = df["bid"] * 0.01
    df["ask"] = df["ask"] * 0.01
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def _add_flickers(df, model, rng, T):
    """Insert brief excursions that return to the same quotes, inside long enough states."""
    n_fl = rng.poisson(model.flickers_per_move * max(len(df) - 1, 0))
    if n_fl == 0:
        return df
    t = df["t"].to_numpy()
    ends = np.append(t[1:], T)
    extra = []
    for _ in range(n_fl):
        k = int(rng.integers(len(df)))
        D = float(rng.choice(model.flicker_durations))
        span = ends[k] - t[k]
        if span <= 2 * D + 2e-6:
            continue
        a = t[k] + rng.uniform(1e-6, span - D - 1e-6)
        bid, ask = int(df.at[k, "bid"]), int(df.at[k, "ask"])
        if rng.uniform() < model.p_flicker_lock:            # one side touches the other and back
            mid = (bid + 1, ask) if rng.uniform() < 0.5 else (bid, ask - 1)
        else:                                               # one side retreats and comes back
            mid = (bid - 1, ask) if rng.uniform() < 0.5 else (bid, ask + 1)
        if mid[1] - mid[0] < 0:
            continue
        extra += [(a, *mid), (a + D, bid, ask)]
    if not extra:
        return df
    return pd.concat([df, pd.DataFrame(extra, columns=["t", "bid", "ask"])], ignore_index=True) \
             .sort_values("t", kind="stable").reset_index(drop=True)

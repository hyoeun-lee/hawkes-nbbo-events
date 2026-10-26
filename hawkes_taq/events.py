"""Event definitions E0, E1 and E2 built from a stock-day of NBBO price changes.

All prices are converted to integer ticks. The mid-price is kept in half-ticks (bid + ask in
ticks), so every mid change is an integer. A state is "valid" when the spread is at least one
tick; locked (spread 0) and crossed (spread < 0) states are "non-positive".

E0  Every NBBO price change that moves the mid-price. Type 'up' or 'down'.
E1  Mid-price moves after deleting transient non-positive states, i.e. locks and crosses that
    last less than `max_nonpos` seconds. A move that passed through such a state counts once,
    stamped at its exit (default) or entry. Non-positive states lasting longer are kept as
    ordinary states. Zero-duration states (rows sharing a timestamp, whose order is unknown in
    complete_nbbo) are merged the same way. With min_state > 0, every state shorter than
    min_state is merged too, whatever its spread. Type 'up' or 'down'.
E2  Locks as their own events: 'up'/'down' for moves between valid states, 'lock_on' for a move
    from a valid into a non-positive state, 'lock_off' for the reverse. Moves between two
    non-positive states are dropped (counted in the QC summary).
"""

import numpy as np
import pandas as pd

TICK = 0.01


def build_states(nbbo, end_time):
    """Turn NBBO price changes into a table of states with durations.

    nbbo has columns t, bid, ask (sorted by t). end_time is the end of the session in seconds
    after midnight and closes the last state. Returns columns t, bid, ask (in ticks), spread
    (ticks), mid2 (mid-price in half-ticks) and dur (seconds).
    """
    bid = np.rint(nbbo["bid"].to_numpy() / TICK).astype(np.int64)
    ask = np.rint(nbbo["ask"].to_numpy() / TICK).astype(np.int64)
    t = nbbo["t"].to_numpy(dtype=float)
    # Drop rows that repeat the previous state (can remain after rounding to ticks).
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = (bid[1:] != bid[:-1]) | (ask[1:] != ask[:-1])
    t, bid, ask = t[keep], bid[keep], ask[keep]
    # Round to the nanosecond so float noise doesn't move durations across thresholds.
    dur = np.round(np.diff(np.append(t, end_time)), 9)
    return pd.DataFrame({"t": t, "bid": bid, "ask": ask, "spread": ask - bid,
                         "mid2": bid + ask, "dur": dur})


def events_e0(states):
    """Every NBBO change that moves the mid-price."""
    d = np.diff(states["mid2"].to_numpy())
    idx = np.flatnonzero(d != 0) + 1
    return pd.DataFrame({"t": states["t"].to_numpy()[idx],
                         "type": np.where(d[idx - 1] > 0, "up", "down"),
                         "size": np.abs(d[idx - 1]),
                         "spread_after": states["spread"].to_numpy()[idx]})


def events_e1(states, max_nonpos=0.005, stamp="exit", min_state=0.0):
    """Mid-price moves with transient locks and crosses merged into the move around them.

    min_state > 0 also merges every state shorter than min_state seconds, whatever its
    spread (for example a two-tick widening reversed within microseconds).
    """
    if stamp not in ("exit", "entry"):
        raise ValueError("stamp must be 'exit' or 'entry'")
    s = states
    dur = s["dur"].to_numpy()
    # Zero-duration states come from rows sharing a timestamp, whose order is unknown.
    transient = ((s["spread"].to_numpy() <= 0) & (dur < max_nonpos)) | (dur == 0) | (dur < min_state)
    kept = np.flatnonzero(~transient)
    if len(kept) < 2:
        return pd.DataFrame(columns=["t", "type", "size", "via_merged"])
    prev, cur = kept[:-1], kept[1:]
    mid2, t = s["mid2"].to_numpy(), s["t"].to_numpy()
    d = mid2[cur] - mid2[prev]
    via = cur > prev + 1
    t_ev = t[cur] if stamp == "exit" else t[np.where(via, prev + 1, cur)]
    m = d != 0
    return pd.DataFrame({"t": t_ev[m], "type": np.where(d[m] > 0, "up", "down"),
                         "size": np.abs(d[m]), "via_merged": via[m]})


def events_e2(states):
    """Moves between valid states, plus lock entries and exits as separate event types."""
    s = states
    valid = s["spread"].to_numpy() > 0
    d = np.diff(s["mid2"].to_numpy())
    pv, cv = valid[:-1], valid[1:]
    typ = np.full(len(d), "", dtype=object)
    both = pv & cv
    typ[both & (d > 0)] = "up"
    typ[both & (d < 0)] = "down"
    typ[pv & ~cv] = "lock_on"
    typ[~pv & cv] = "lock_off"
    m = typ != ""
    t = s["t"].to_numpy()[1:]
    return pd.DataFrame({"t": t[m], "type": typ[m], "mid_change": d[m]})


def spread_ties(t, resolution=1e-6):
    """Spread identical times evenly within one timestamp resolution, keeping their order.

    A Hawkes likelihood assumes no two events share a time. Following Lee & Seo (2023), the
    k-th of n events sharing a time t is moved to t + k * resolution / n.
    """
    t = np.asarray(t, dtype=float)
    out = t.copy()
    i = 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] == t[i]:
            j += 1
        n = j - i + 1
        if n > 1:
            out[i:j + 1] = t[i] + np.arange(n) * resolution / n
        i = j + 1
    return out


def qc_summary(nbbo, states, e0, e1, e2, max_nonpos=0.005):
    """One row of counts for a stock-day, for checking the cleaning."""
    s = states
    nonpos = s["spread"].to_numpy() <= 0
    transient = nonpos & (s["dur"].to_numpy() < max_nonpos)
    valid = ~nonpos
    total = s["dur"].sum()
    d_mid = np.diff(s["mid2"].to_numpy())
    return {
        "n_rows": len(nbbo),
        "n_tied_rows": int(nbbo["t"].duplicated(keep=False).sum()),
        "n_states": len(s),
        "pct_time_1tick": 100 * s.loc[s["spread"] == 1, "dur"].sum() / total if total > 0 else np.nan,
        "n_nonpos": int(nonpos.sum()),
        "n_nonpos_transient": int(transient.sum()),
        "n_nonpos_long": int((nonpos & ~transient).sum()),
        "n_no_mid_change": int((d_mid == 0).sum()),
        "n_e0": len(e0),
        "n_e1": len(e1),
        "n_e1_via_merged": int(e1["via_merged"].sum()) if len(e1) else 0,
        "n_e2_up_down": int(e2["type"].isin(["up", "down"]).sum()),
        "n_e2_lock_on": int((e2["type"] == "lock_on").sum()),
        "n_e2_lock_off": int((e2["type"] == "lock_off").sum()),
        "n_nonpos_to_nonpos": int((~valid[:-1] & ~valid[1:]).sum()),
    }

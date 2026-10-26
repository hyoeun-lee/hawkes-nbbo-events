import numpy as np
import pandas as pd
import pytest

from hawkes_taq import (build_states, events_e0, events_e1, events_e2, qc_summary,
                        spread_ties)

END = 100.0


def nbbo(rows):
    """rows: (t, bid, ask) with prices in dollars."""
    return pd.DataFrame(rows, columns=["t", "bid", "ask"])


# One tick -> locked for 0.5 ms -> one tick, one tick higher.
LOCK_UP = nbbo([(10.0, 30.00, 30.01), (20.0, 30.01, 30.01), (20.0005, 30.01, 30.02)])
# One tick -> locked for 0.5 ms -> back to the same quotes.
LOCK_REVERT = nbbo([(10.0, 30.00, 30.01), (20.0, 30.01, 30.01), (20.0005, 30.00, 30.01)])
# Same as LOCK_UP but the lock lasts 10 ms.
LONG_LOCK_UP = nbbo([(10.0, 30.00, 30.01), (20.0, 30.01, 30.01), (20.010, 30.01, 30.02)])
# Both sides move up together, no lock.
DIRECT_UP = nbbo([(10.0, 30.00, 30.01), (20.0, 30.01, 30.02)])


def test_states_ticks_mid_and_durations():
    s = build_states(LOCK_UP, END)
    assert s["spread"].tolist() == [1, 0, 1]
    assert s["mid2"].tolist() == [6001, 6002, 6003]
    assert np.allclose(s["dur"], [10.0, 0.0005, END - 20.0005])


def test_states_drop_repeated_quotes():
    s = build_states(nbbo([(1.0, 30.00, 30.01), (2.0, 30.000001, 30.01), (3.0, 30.01, 30.02)]), END)
    assert len(s) == 2


def test_e0_counts_each_half_tick_move():
    e = events_e0(build_states(LOCK_UP, END))
    assert e["type"].tolist() == ["up", "up"]
    assert e["size"].tolist() == [1, 1]
    assert e["t"].tolist() == [20.0, 20.0005]


def test_e1_merges_transient_lock_into_one_move():
    e = events_e1(build_states(LOCK_UP, END))
    assert len(e) == 1
    row = e.iloc[0]
    assert (row["type"], row["size"], bool(row["via_merged"])) == ("up", 2, True)
    assert row["t"] == 20.0005


def test_e1_entry_stamp():
    e = events_e1(build_states(LOCK_UP, END), stamp="entry")
    assert e["t"].tolist() == [20.0]


def test_e1_reverting_lock_is_no_event():
    assert len(events_e1(build_states(LOCK_REVERT, END))) == 0
    assert len(events_e0(build_states(LOCK_REVERT, END))) == 2


def test_e1_keeps_long_lock_as_a_state():
    e = events_e1(build_states(LONG_LOCK_UP, END), max_nonpos=0.005)
    assert e["t"].tolist() == [20.0, 20.010]
    assert not e["via_merged"].any()


def test_e1_direct_move_not_via_lock():
    e = events_e1(build_states(DIRECT_UP, END))
    assert len(e) == 1 and not e["via_merged"].iloc[0] and e["size"].iloc[0] == 2


def test_e1_threshold_is_strict():
    s = build_states(nbbo([(10.0, 30.00, 30.01), (20.0, 30.01, 30.01), (20.005, 30.01, 30.02)]), END)
    assert len(events_e1(s, max_nonpos=0.005)) == 2


def test_e1_bad_stamp():
    with pytest.raises(ValueError):
        events_e1(build_states(LOCK_UP, END), stamp="middle")


def test_e2_lock_on_and_off():
    e = events_e2(build_states(LOCK_UP, END))
    assert e["type"].tolist() == ["lock_on", "lock_off"]
    assert e["mid_change"].tolist() == [1, 1]
    assert events_e2(build_states(DIRECT_UP, END))["type"].tolist() == ["up"]


def test_e2_drops_lock_to_cross():
    s = build_states(nbbo([(1.0, 30.00, 30.01), (2.0, 30.01, 30.01), (3.0, 30.02, 30.01),
                           (4.0, 30.02, 30.03)]), END)
    assert events_e2(s)["type"].tolist() == ["lock_on", "lock_off"]


def test_wider_spreads_move_mid_by_half_ticks():
    s = build_states(nbbo([(1.0, 30.00, 30.01), (2.0, 30.00, 30.02), (3.0, 30.01, 30.02)]), END)
    e = events_e1(s)
    assert e["type"].tolist() == ["up", "up"] and e["size"].tolist() == [1, 1]


def test_e1_merges_zero_duration_states_from_ties():
    s = build_states(nbbo([(1.0, 30.00, 30.02), (2.0, 30.01, 30.02), (2.0, 30.01, 30.03),
                           (3.0, 30.02, 30.03)]), END)
    e = events_e1(s)
    assert e["t"].tolist() == [2.0, 3.0]
    assert e["size"].tolist() == [2, 1]


def test_e1_min_state_merges_brief_widening():
    # one tick -> two ticks for 40 us -> one tick again, one tick lower: one move of size 2
    s = build_states(nbbo([(10.0, 30.00, 30.01), (20.0, 29.99, 30.01), (20.00004, 29.99, 30.00)]), END)
    assert len(events_e1(s)) == 2
    e = events_e1(s, min_state=0.001)
    assert e["type"].tolist() == ["down"] and e["size"].tolist() == [2]
    assert bool(e["via_merged"].iloc[0])


def test_spread_ties():
    out = spread_ties([1.0, 2.0, 2.0, 2.0, 3.0], resolution=1e-6)
    assert np.allclose(out, [1.0, 2.0, 2.0 + 1e-6 / 3, 2.0 + 2e-6 / 3, 3.0])
    assert np.all(np.diff(out) > 0)


def test_qc_summary_counts():
    s = build_states(LOCK_UP, END)
    q = qc_summary(LOCK_UP, s, events_e0(s), events_e1(s), events_e2(s))
    assert q["n_nonpos"] == 1 and q["n_nonpos_transient"] == 1 and q["n_nonpos_long"] == 0
    assert q["n_e0"] == 2 and q["n_e1"] == 1 and q["n_e1_via_merged"] == 1
    assert q["n_e2_lock_on"] == 1 and q["n_e2_lock_off"] == 1
    assert q["n_tied_rows"] == 0
    assert q["pct_time_1tick"] == pytest.approx(100 * (END - 0.0005 - 10.0) / (END - 10.0))

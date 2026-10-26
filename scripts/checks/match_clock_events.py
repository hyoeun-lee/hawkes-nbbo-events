"""Event-level matching of E1m (and E0) events between the SIP-time and exchange-time NBBO."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hawkes_taq import events as ev, hawkes as hk
S = sys.argv[1] if len(sys.argv) > 1 else "."  # folder with lf_raw/lfnbbo/{sip,ex}/SYM/*.csv.gz
TOLS = (0.001, 0.005, 0.02)

def build(f, definition):
    s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
    e = ev.events_e0(s) if definition == "E0" else ev.events_e1(s, min_state=0.001)
    return e["t"].to_numpy(), np.where(e["type"].to_numpy() == "up", 1, -1), e["size"].to_numpy()

def match(ta, ya, tb, yb, tol):
    """Greedy one-to-one matching in time order: each a-event to the nearest unmatched b-event of
    the same direction within tol. Returns matched pairs (ia, ib)."""
    pairs = []
    for sgn in (1, -1):
        ia = np.flatnonzero(ya == sgn); ib = np.flatnonzero(yb == sgn)
        a, b = ta[ia], tb[ib]
        cand = []
        j0 = 0
        for i, t in enumerate(a):
            lo = np.searchsorted(b, t - tol); hi = np.searchsorted(b, t + tol, side="right")
            for j in range(lo, hi):
                cand.append((abs(b[j] - t), i, j))
        cand.sort()
        used_a, used_b = set(), set()
        for d, i, j in cand:
            if i in used_a or j in used_b: continue
            used_a.add(i); used_b.add(j); pairs.append((ia[i], ib[j]))
    return pairs

rows = []
for sym in ["INTC", "BAC", "MDLZ", "TGT"]:
    for definition in ("E1m", "E0"):
        tot = {tol: [0, 0, 0, 0, []] for tol in TOLS}   # n_sip, n_ex, matched, same_size, offsets
        for f_sip in sorted(Path(f"{S}/lf_raw/lfnbbo/sip/{sym}").glob("2017*.csv.gz")):
            f_ex = Path(f"{S}/lf_raw/lfnbbo/ex/{sym}/{f_sip.name}")
            ta, ya, sa = build(f_sip, definition); tb, yb, sb = build(f_ex, definition)
            for tol in TOLS:
                p = match(ta, ya, tb, yb, tol)
                ia = np.array([x[0] for x in p], dtype=int); ib = np.array([x[1] for x in p], dtype=int)
                r = tot[tol]
                r[0] += len(ta); r[1] += len(tb); r[2] += len(p)
                r[3] += int(np.sum(sa[ia] == sb[ib])) if len(p) else 0
                r[4].append(tb[ib] - ta[ia])
        for tol in TOLS:
            n_a, n_b, m, same, off = tot[tol]; off = np.concatenate(off)
            rows.append(dict(sym=sym, definition=definition, tol_ms=tol * 1e3, n_sip=n_a, n_ex=n_b,
                             matched_share_sip=m / n_a, matched_share_ex=m / n_b, same_size=same / max(m, 1),
                             offset_median_ms=1e3 * float(np.median(off)),
                             offset_abs_p90_ms=1e3 * float(np.quantile(np.abs(off), 0.9))))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{S}/match_clock_events.csv", index=False)

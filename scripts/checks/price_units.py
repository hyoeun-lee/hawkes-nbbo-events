"""Raw and clean events compared in price units, May 2017 (review point C)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from fit_panel import month_files
from hawkes_taq import events as ev, hawkes as hk
S = sys.argv[1] if len(sys.argv) > 1 else "."  # folder with nbbo/2017Q2, delta/K3.csv, delta/K4.csv, lf_fit/fits_lf.csv
L = lambda s: np.array([float(x) for x in str(s).split(";")])
HS = (0.001, 0.01, 0.1, 1.0, 10.0)

def empirical(sym, definition):
    """Mean size (ticks) and mean signed mid change (ticks) over (t, t+H] after each event."""
    sizes, disp = [], {H: [] for H in HS}
    for f in month_files([f"{S}/nbbo/2017Q2"], sym, "201705"):
        s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
        e = ev.events_e0(s) if definition == "E0" else ev.events_e1(s, min_state=0.001)
        t_s, m_s = s["t"].to_numpy(), s["mid2"].to_numpy() / 2.0     # mid in ticks
        te = e["t"].to_numpy(); sg = np.where(e["type"].to_numpy() == "up", 1.0, -1.0)
        m0 = m_s[np.searchsorted(t_s, te, side="right") - 1]          # mid just after the event
        ok_end = te + max(HS) <= hk.SESSION_END
        for H in HS:
            m1 = m_s[np.searchsorted(t_s, te + H, side="right") - 1]
            disp[H].append((sg * (m1 - m0))[ok_end])
        sizes.append(e["size"].to_numpy() / 2.0)
    return float(np.mean(np.concatenate(sizes))), {H: float(np.mean(np.concatenate(v))) for H, v in disp.items()}

def cascade(mem, a, c, delta, rng, n=40000, Hmax=10.0):
    """Signed number of descendants within each H of one up event (sign +1), by simulation."""
    beta = 1.0 / mem
    out = {H: np.zeros(n) for H in HS}
    # queue of (cascade id, time, sign)
    ids = np.arange(n); tt = np.zeros(n); sg = np.ones(n)
    while len(ids):
        new_ids, new_t, new_s = [], [], []
        for k in range(len(beta)):
            for w, flip in ((a[k], 1.0), (c[k], -1.0)):
                cnt = rng.poisson(w, size=len(ids))
                if cnt.sum() == 0: continue
                rep = np.repeat(np.arange(len(ids)), cnt)
                d = rng.exponential(1.0 / beta[k], size=len(rep))
                keep = d >= delta                      # offspring inside the dead window are not realized
                rep, d = rep[keep], d[keep]
                t_new = tt[rep] + d
                keep = t_new <= Hmax
                rep, t_new = rep[keep], t_new[keep]
                new_ids.append(ids[rep]); new_t.append(t_new); new_s.append(sg[rep] * flip)
        if not new_ids: break
        ids = np.concatenate(new_ids); tt = np.concatenate(new_t); sg = np.concatenate(new_s)
        for H in HS:
            m = tt <= H
            np.add.at(out[H], ids[m], sg[m])
    return {H: float(v.mean()) for H, v in out.items()}

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    main4 = pd.read_csv(f"{S}/delta/K4.csv"); main3 = pd.read_csv(f"{S}/delta/K3.csv")
    lf = pd.read_csv(f"{S}/lf_fit/fits_lf.csv")
    rows = []
    for sym in ["INTC", "BAC", "MDLZ", "TGT"]:
        r1 = (main3 if sym == "BAC" else main4); r1 = r1[(r1.sym == sym) & (r1.month == 201705)].iloc[0]
        r0 = lf[(lf.sym == sym) & (lf.version == "complete") & (lf.definition == "E0")].iloc[0]
        for definition, r, delta in (("E0", r0, 0.0), ("E1m", r1, 0.001)):
            mem, a, c = L(r.memory_s), L(r.self_per_scale), L(r.cross_per_scale)
            size, emp = empirical(sym, definition)
            mod = cascade(mem, a, c, delta, rng)
            exc = a + c
            n = lambda H: float(np.sum(exc * (np.exp(-delta / mem) - np.exp(-H / mem))))
            row = dict(sym=sym, definition=definition, mean_size_ticks=size,
                       **{f"n_{H}": n(H) for H in HS},
                       **{f"model_disp_{H}": size * mod[H] for H in HS},
                       **{f"emp_disp_{H}": emp[H] for H in HS})
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(f"{S}/price_units.csv", index=False); print("ALLDONE")

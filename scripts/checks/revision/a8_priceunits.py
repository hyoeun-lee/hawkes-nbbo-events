"""A8: price units with the model side computed on simulated paths (same functional as the data),
standard errors on the data means (naive and day-clustered), and the isolated-cascade value for
comparison. May 2017, four stocks; raw = E0 K = 4 (May fits of lf_fit), clean = main fits."""
import os
import sys
from pathlib import Path, time, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, str(Path(__file__).resolve().parents[3]) + '/scripts'); sys.path.insert(0, str(Path(__file__).resolve().parents[3]) + '/scripts/checks'); sys.path.insert(0, str(Path(__file__).resolve().parents[3]) + '/scripts/checks/misfit')
from hawkes_taq import events as ev, hawkes as hk
import fit_panel as fp
from recon import reconstruct
from price_units import cascade
S = os.environ.get("HAWKES_DATA", "DATA")   # root of the NBBO, trade and intermediate files (see REPLICATION.md)
HS = (0.01, 0.1, 1.0, 10.0); R = 5
L = lambda s: np.array([float(x) for x in str(s).split(";")])

def empirical(sym, definition):
    sizes, per_day = [], {H: [] for H in HS}
    for f in fp.month_files([f"{S}/may/NBBO"], sym, "201705"):
        s = ev.build_states(pd.read_csv(f), hk.SESSION_END)
        e = ev.events_e0(s) if definition == "E0" else ev.events_e1(s, min_state=0.001)
        t_s, m_s = s["t"].to_numpy(), s["mid2"].to_numpy() / 2.0
        te = e["t"].to_numpy(); sg = np.where(e["type"].to_numpy() == "up", 1.0, -1.0)
        m0 = m_s[np.searchsorted(t_s, te, side="right") - 1]; ok = te + max(HS) <= hk.SESSION_END
        for H in HS:
            m1 = m_s[np.searchsorted(t_s, te + H, side="right") - 1]; per_day[H].append((sg * (m1 - m0))[ok])
        sizes.append(e["size"].to_numpy() / 2.0)
    out = {}
    for H in HS:
        allv = np.concatenate(per_day[H]); dm = np.array([v.mean() for v in per_day[H]])
        out[H] = (allv.mean(), allv.std() / np.sqrt(len(allv)), dm.std(ddof=1) / np.sqrt(len(dm)))
    return float(np.mean(np.concatenate(sizes))), out

def on_paths(days, fit, size, rng):
    acc = {H: [] for H in HS}
    for r in range(R):
        for d in hk.simulate_like(days, fit, rng):
            t = d.t; sg = np.where(d.types == 0, 1.0, -1.0); M = np.cumsum(sg * size)
            ok = t + max(HS) <= d.T
            for H in HS:
                j = np.searchsorted(t, t + H, side="right") - 1
                acc[H].append((sg * (M[j] - M))[ok])
    return {H: float(np.concatenate(v).mean()) for H, v in acc.items()}

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    k4 = pd.read_csv(f"{S}/delta/K4.csv", dtype={"month": str}); k3 = pd.read_csv(f"{S}/delta/K3.csv", dtype={"month": str}); lf = pd.read_csv(f"{S}/lf_fit/fits_lf.csv", dtype={"month": str})
    gaps = {"INTC": 0.451, "BAC": 0.414, "MDLZ": 0.501, "TGT": 0.504}
    rows = []
    for sym in ["INTC", "BAC", "MDLZ", "TGT"]:
        r1 = (k3 if sym == "BAC" else k4); r1 = r1[(r1.sym == sym) & (r1.month == "201705") & (r1.definition == "E1m")].iloc[0]
        r0 = lf[(lf.sym == sym) & (lf.version == "complete") & (lf.definition == "E0")].iloc[0]
        for definition, row in (("E0", r0), ("E1m", r1)):
            t0 = time.time()
            days = fp.build_days(fp.month_files([f"{S}/may/NBBO"], sym, "201705"), definition, 0.001)
            fit = reconstruct(days, row)
            size, emp = empirical(sym, definition)
            sim = on_paths(days, fit, size, rng)
            casc = cascade(L(row.memory_s), L(row.self_per_scale), L(row.cross_per_scale), float(row.delta), rng)
            out = dict(sym=sym, definition=definition, size=round(size, 3), loglik_recon=round(fit.loglik, 1), loglik_csv=row.loglik)
            for H in HS:
                m, se_n, se_d = emp[H]
                out.update({f"data_{H}": round(m, 4), f"se_naive_{H}": round(se_n, 4), f"se_day_{H}": round(se_d, 4), f"paths_{H}": round(sim[H], 4), f"cascade_{H}": round(size * casc[H], 4)})
            if definition == "E0":
                s = gaps[sym]; out["one_over_1ms"] = round(1 / (1 - s), 2); out["ratio_paths_data_1s"] = round(sim[1.0] / emp[1.0][0], 2); out["ratio_paths_data_100ms"] = round(sim[0.1] / emp[0.1][0], 2)
            out["minutes"] = round((time.time() - t0) / 60, 1); rows.append(out); print(out, flush=True)
            pd.DataFrame(rows).to_csv(f"{S}/a5/a8_priceunits.csv", index=False)
    print("ALLDONE")

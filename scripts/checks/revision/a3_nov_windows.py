"""A3 part 3: kernel by window of the day in November 2017, SIP time against exchange time,
for the four reconstruction stocks. E1m, K = 4, 1 ms dead time, 30-min bins, 60 s burn-in,
each window a session; started from the month's fit on the same version."""
import os
import sys
from pathlib import Path, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[3])); sys.path.insert(0, str(Path(__file__).resolve().parents[3]) + '/scripts')
from hawkes_taq import hawkes as hk
import fit_panel as fp
S = os.environ.get("HAWKES_DATA", "DATA")   # root of the NBBO, trade and intermediate files (see REPLICATION.md)
ROOTS = {"cqm_sip": f"{S}/lf_nov_in/lfnbbo/sip", "cqm_ex": f"{S}/lf_nov_in/lfnbbo/ex"}
fits = pd.read_csv(f"{S}/lf_nov/fits_lf_201711.csv", dtype={"month": str})
fits = fits[(fits.definition == "E1m") & (fits.K == 4)]
def windows(T): return [("10:00-11:00", 0.0, 3600.0), ("11:00-14:30", 3600.0, T - 3600.0), ("14:30-15:30", T - 3600.0, T)]
def piece(days, a, b):
    out = []
    for d in days:
        m = (d.t >= a) & (d.t < b)
        if m.sum() > 10: out.append(hk.Day(t=d.t[m] - a, types=d.types[m], T=b - a, label=d.label))
    return out
def mass(f, lo, H): return float(np.sum(f.alpha.sum(axis=0) * (np.exp(-f.beta * lo) - np.exp(-f.beta * H))))
def job(sym, version, name, lo, hi):
    t0 = time.time()
    days = fp.build_days(fp.month_files([ROOTS[version]], sym, "201711"), "E1m", 0.001)
    r0 = fits[(fits.sym == sym) & (fits.version == version)].iloc[0]
    beta0 = 1.0 / np.array([float(x) for x in str(r0.memory_s).split(";")])
    T = days[0].T; lo, hi = (lo, hi if hi > 0 else T) if name != "11:00-14:30" else (3600.0, T - 3600.0)
    if name == "14:30-15:30": lo, hi = T - 3600.0, T
    ds = piece(days, lo, hi)
    f = hk.fit(ds, K=4, kernel="symmetric", bin_minutes=30, starts=[beta0], baseline="shared", delta=0.001, burn=60.0)
    return dict(sym=sym, version=version, window=name, n_days=len(ds), n_events=f.n_events,
                events_per_hour=round(f.n_events / len(ds) / ((hi - lo) / 3600.0), 1),
                memory_s="; ".join(f"{m:.4g}" for m in f.timescales()), excitation="; ".join(f"{x:.4f}" for x in f.alpha.sum(axis=0)),
                br=round(float(f.branching_ratio), 4), n1=round(mass(f, 0.001, 1.0), 4), n10=round(mass(f, 0.001, 10.0), 4), minutes=round((time.time() - t0) / 60, 2))
if __name__ == "__main__":
    jobs = [(s, v, n, lo, hi) for s in ["INTC", "MDLZ", "BAC", "TGT"] for v in ROOTS for n, lo, hi in [("10:00-11:00", 0.0, 3600.0), ("11:00-14:30", 0, 0), ("14:30-15:30", 0, 0)]]
    out = f"{S}/a5/a3_nov_windows.csv"; rows = []
    with ProcessPoolExecutor(4) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for fu in as_completed(futs):
            try: r = fu.result()
            except Exception as e: print("FAILED", futs[fu], repr(e)[:200], flush=True); continue
            rows.append(r); pd.DataFrame(rows).to_csv(out, index=False); print(r, flush=True)

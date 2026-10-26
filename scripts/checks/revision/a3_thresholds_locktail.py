"""A3 parts A and B, and A4: activity gradients at h = 1 ms and 2 ms, the lock tail by month
from the calibration chain tables, and the across-month relation with a month trend, month
fixed effects and the lock median as a regressor. See README.md for the paths."""
import glob
import numpy as np
import pandas as pd

S = os.environ.get("HAWKES_DATA", "DATA")   # root of the NBBO, trade and intermediate files (see REPLICATION.md)
ORDER = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]
MONTHS = ["201611", "201612"] + [f"2017{m:02d}" for m in range(1, 13)]
vec = lambda s: np.array([float(x) for x in str(s).split(";")])


def mass(r, lo, H):
    m, e = vec(r.memory_s), vec(r.excitation_per_scale)
    return float(np.sum(e * (np.exp(-lo / m) - np.exp(-H / m))))


def panel(k4, k3, h):
    d = pd.concat([k4[k4.sym != "BAC"], k3[k3.sym == "BAC"]]).copy()
    d["n1"] = [mass(r, h, 1) for _, r in d.iterrows()]; d["n10"] = [mass(r, h, 10) for _, r in d.iterrows()]
    d["epd"] = d.n_events / d.n_days; d["h"] = h
    return d[["sym", "month", "n1", "n10", "epd", "branching_ratio", "h"]]


def demean(p, cols):
    for c in cols:
        p[c + "_d"] = p[c] - p.groupby("sym")[c].transform("mean")
    return p


def gradients(p, label):
    p = p.copy(); p["ln1"] = np.log(p.n1); p["ln10"] = np.log(p.n10); p["la"] = np.log(p.epd); demean(p, ["ln1", "ln10", "la"])
    print(f"{label}: within-stock corr(log n1, log events/day) {np.corrcoef(p.ln1_d, p.la_d)[0,1]:.2f}, n10 {np.corrcoef(p.ln10_d, p.la_d)[0,1]:.2f}; "
          f"elasticity n1 {np.polyfit(p.la_d, p.ln1_d, 1)[0]:.2f}, n10 {np.polyfit(p.la_d, p.ln10_d, 1)[0]:.2f}")


def main():
    rd = lambda f: pd.read_csv(f"{S}/delta/{f}", dtype={"month": str})
    p1 = panel(rd("K4.csv"), rd("K3.csv"), 0.001); p2 = panel(rd("h2_K4.csv"), rd("h2_K3.csv"), 0.002)
    gradients(p1, "h = 1 ms"); gradients(p2, "h = 2 ms")
    m = p1.merge(p2, on=["sym", "month"], suffixes=("_1", "_2")); m["ratio"] = m.n1_1 / m.n1_2
    m["la"] = np.log(m.epd_1); demean(m, ["la"])
    print(f"n1(1 ms)/n1(2 ms): median {m.ratio.median():.3f} [{m.ratio.quantile(.25):.3f}-{m.ratio.quantile(.75):.3f}]; "
          f"corr with activity {np.corrcoef(np.log(m.ratio), m.la_d)[0,1]:.2f}; slope per doubling {np.polyfit(m.la_d, np.log(m.ratio), 1)[0]*np.log(2):.3f}")
    rows = []
    for f in sorted(glob.glob(f"{S}/sim/cal/chains_*.csv.gz")):
        sym, month = f.split("chains_")[1].replace(".csv.gz", "").split("_")
        d = pd.read_csv(f); mv = d[d.what == "move"]; lk = mv[mv.kind == "lock"]; nd = d.day.nunique()
        rows.append(dict(sym=sym, month=month, moves_per_day=len(mv) / nd, lock_med_ms=1e3 * lk.chain_s.median(),
                         lock_p90_ms=1e3 * lk.chain_s.quantile(.9), lock_gt1=(lk.chain_s > 1e-3).mean(), lock_gt2=(lk.chain_s > 2e-3).mean(),
                         mi=MONTHS.index(month)))
    t = pd.DataFrame(rows); t["la"] = np.log(t.moves_per_day); demean(t, ["la", "lock_gt1", "lock_gt2", "lock_med_ms", "lock_p90_ms", "mi"])
    print(f"lock tail within stock: corr with log moves/day: share>1ms {np.corrcoef(t.lock_gt1_d, t.la_d)[0,1]:+.2f}, share>2ms {np.corrcoef(t.lock_gt2_d, t.la_d)[0,1]:+.2f}, "
          f"median {np.corrcoef(t.lock_med_ms_d, t.la_d)[0,1]:+.2f}; with month index: share>1ms {np.corrcoef(t.lock_gt1_d, t.mi_d)[0,1]:+.2f}, median {np.corrcoef(t.lock_med_ms_d, t.mi_d)[0,1]:+.2f}")
    print("lock median (ms) by month, median over stocks:", t.groupby("month").lock_med_ms.median().round(2).tolist())
    d = m.merge(t, on=["sym", "month"]); d["ln1"] = np.log(d.n1_1); d["t"] = d.mi; demean(d, ["ln1", "t", "lock_med_ms", "lock_gt1", "n1_1"])
    d["la_d"] = d["la_d_x"] if "la_d_x" in d else d["la_d"]
    b = np.linalg.lstsq(np.column_stack([d.la_d]), d.ln1_d, rcond=None)[0]; print(f"A4: elasticity {b[0]:.2f}")
    b = np.linalg.lstsq(np.column_stack([d.la_d, d.t_d]), d.ln1_d, rcond=None)[0]; print(f"    with month trend: {b[0]:.2f} (trend {b[1]:+.3f}/month)")
    d["ln1_dd"] = d.ln1_d - d.groupby("month").ln1_d.transform("mean"); d["la_dd"] = d.la_d - d.groupby("month").la_d.transform("mean")
    print(f"    with stock and month fixed effects: {np.polyfit(d.la_dd, d.ln1_dd, 1)[0]:.2f} (corr {np.corrcoef(d.la_dd, d.ln1_dd)[0,1]:.2f})")
    b = np.linalg.lstsq(np.column_stack([d.la_d, d.lock_med_ms_d]), d.ln1_d, rcond=None)[0]; print(f"    with the lock median: {b[0]:.2f} (lock median {b[1]:+.2f}/ms)")
    print(f"    corr(log n1, lock median) {np.corrcoef(d.ln1_d, d.lock_med_ms_d)[0,1]:+.2f}; levels corr(n1, log activity) {np.corrcoef(d.n1_1_d, d.la_d)[0,1]:.2f}")


if __name__ == "__main__":
    main()

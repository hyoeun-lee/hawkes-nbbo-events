"""Tables for results §24 from the three cluster jobs: kernel by hour (by_hour_SYM.csv),
outlier-day refits (outliers_SYM.csv) and the paired day-block bootstrap (blockboot/SYM_201705.csv).

    python scripts/summarize_panel_checks.py DATA_DIR [--fits DATA_DIR/fits_e1m_delta_K4.csv]
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]


def by_hour(d):
    d = d.copy()
    piv = d.pivot_table(index=["sym", "month"], columns="window", values=["n1", "n10", "branching_ratio", "events_per_hour"])
    first, mid, last = "10:00-11:00", "11:00-14:30", "14:30-15:30"
    out = pd.DataFrame({
        "n1 first/mid": piv["n1"][first] / piv["n1"][mid], "n1 last/mid": piv["n1"][last] / piv["n1"][mid],
        "n10 first/mid": piv["n10"][first] / piv["n10"][mid], "n10 last/mid": piv["n10"][last] / piv["n10"][mid],
        "br first/mid": piv["branching_ratio"][first] / piv["branching_ratio"][mid],
        "activity first/mid": piv["events_per_hour"][first] / piv["events_per_hour"][mid],
        "activity last/mid": piv["events_per_hour"][last] / piv["events_per_hour"][mid]})
    print("\n### Kernel by hour: stock-months", len(out))
    g = out.groupby(level="sym").agg(["median", "min", "max"]).reindex([s for s in ORDER if s in out.index.get_level_values(0)])
    print(g.round(2).to_string())
    print("\nAll stock-months: median n1 first/mid %.2f (IQR %.2f-%.2f); share > 1: %.2f; n10 first/mid median %.2f; last/mid n1 median %.2f"
          % (out["n1 first/mid"].median(), out["n1 first/mid"].quantile(.25), out["n1 first/mid"].quantile(.75),
             (out["n1 first/mid"] > 1).mean(), out["n10 first/mid"].median(), out["n1 last/mid"].median()))
    # within-day activity-excitation relation: slope of log n1 on log activity across the three windows, pooled
    x = np.log(d.events_per_hour); y = np.log(d.n1)
    xd = x - d.groupby(["sym", "month"]).events_per_hour.transform(lambda v: np.log(v).mean())
    yd = y - d.groupby(["sym", "month"]).n1.transform(lambda v: np.log(v).mean())
    print("within stock-month elasticity of n1 to activity across windows: %.2f (corr %.2f)" % (np.polyfit(xd, yd, 1)[0], np.corrcoef(xd, yd)[0, 1]))
    return out


def outliers(d):
    print("\n### Outlier days: stock-months", len(d), "flagged", int((d.n_flagged > 0).sum()))
    f = d[d.n_flagged > 0].copy()
    if f.empty:
        return
    f["dn1"] = f.refit_n1 / f.main_n1 - 1; f["dn10"] = f.refit_n10 / f.main_n10 - 1; f["dbr"] = f.refit_br / f.main_br - 1
    f["max/median"] = f.max_e0_per_day / f.median_e0_per_day
    cols = ["sym", "month", "flagged", "max/median", "main_n1", "refit_n1", "dn1", "main_n10", "refit_n10", "dn10", "main_br", "refit_br", "dbr"]
    print(f[cols].round(3).to_string(index=False))
    print("change in n1 without the flagged days: median %+.1f%%, range %+.1f%% to %+.1f%%" % (100 * f.dn1.median(), 100 * f.dn1.min(), 100 * f.dn1.max()))


def blockboot(d, fits=None):
    print("\n### Paired day-block bootstrap, May 2017")
    rows = []
    for sym, g in d.groupby("sym"):
        r = {"sym": sym, "reps": len(g)}
        for c in ("clean_n1", "clean_n10", "clean_br", "raw_n1", "raw_n10", "raw_br", "ratio_n1", "ratio_n10"):
            r[c + " sd"] = g[c].std(); r[c + " p2.5"] = g[c].quantile(.025); r[c + " p97.5"] = g[c].quantile(.975); r[c + " med"] = g[c].median()
        rows.append(r)
    t = pd.DataFrame(rows).set_index("sym").reindex([s for s in ORDER if s in set(d.sym)])
    show = t[[c for c in t.columns if c.endswith(("reps", " sd"))]]
    print("standard deviations:"); print(show.round(4).to_string())
    print("\nratio raw/clean n(1 s): median [2.5%, 97.5%]")
    for sym, r in t.iterrows():
        print(f"  {sym:5s} {r['ratio_n1 med']:.2f} [{r['ratio_n1 p2.5']:.2f}, {r['ratio_n1 p97.5']:.2f}]   n(10 s): {r['ratio_n10 med']:.2f} [{r['ratio_n10 p2.5']:.2f}, {r['ratio_n10 p97.5']:.2f}]   clean n1 sd {r['clean_n1 sd']:.4f}  raw n1 sd {r['raw_n1 sd']:.4f}")
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--fits", default=None)
    a = p.parse_args()
    bh = [pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{a.data}/by_hour_*.csv"))]
    if bh:
        by_hour(pd.concat(bh))
    ol = [pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{a.data}/outliers_*.csv"))]
    if ol:
        outliers(pd.concat(ol))
    bb = [pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{a.data}/blockboot/*.csv"))]
    if bb:
        blockboot(pd.concat(bb), a.fits)


if __name__ == "__main__":
    main()

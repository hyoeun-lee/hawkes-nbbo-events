"""Day-level excitation with the month's timescales held fixed (revision item A2; Section 6.4.4).
Reads results/by_day_fixed_timescales.csv (scripts/fit_by_day.py) and prints the calendar-class ratios,
the count-rule overlap and the day-level elasticities; writes the annotated day table.

    python scripts/checks/revision/a2_day_level.py results/by_day_fixed_timescales.csv results/panel/by_day_annotated.csv
"""
import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
d = pd.read_csv(sys.argv[1], dtype={"month": str, "date": str})
d["calendar"] = d.calendar.fillna("")
d["cls"] = d.calendar.str.split(";")
d["is_cal"] = d.calendar != ""
d["is_mkt"] = d.cls.apply(lambda c: any(x in ("election", "fomc", "index") for x in c))
d["is_earn"] = d.cls.apply(lambda c: "earnings" in c)
d["is_rep"] = d.cls.apply(lambda c: "earnings_report" in c)
d["ordinary"] = ~d.is_cal

# day counts check against the main fits
nd = d.groupby(["sym", "month"]).size()
print("days per stock-month: min", nd.min(), "max", nd.max(), "total", len(d))

# 1. overlap of the count rule with the calendar
print("\n1. Count-flagged days (84) by calendar class")
cf = d[d.count_flag]
print("  on calendar:", int(cf.is_cal.sum()), "| market-wide:", int(cf.is_mkt.sum()),
      "| earnings reaction:", int(cf.is_earn.sum()), "| report day:", int(cf.is_rep.sum()),
      "| none:", int((~cf.is_cal).sum()))
print("  count-flagged dates (not on calendar):")
print(cf[~cf.is_cal].groupby("date").sym.apply(lambda s: " ".join(s)).to_string())
print("  calendar days that are count-flagged, by class:")
for name, m in [("election", d.cls.apply(lambda c: "election" in c)), ("fomc", d.cls.apply(lambda c: "fomc" in c)),
                ("index", d.cls.apply(lambda c: "index" in c)), ("earnings", d.is_earn), ("earnings_report", d.is_rep)]:
    g = d[m]
    print(f"    {name:16s} {int(g.count_flag.sum()):3d} of {len(g):3d}")

# 2. day values relative to the stock-month's ordinary days (median of ordinary days)
for col in ["n1", "n10", "total", "n_e0", "mu_mean"]:
    ref = d[d.ordinary].groupby(["sym", "month"])[col].median().rename("ref")
    d = d.join(ref, on=["sym", "month"])
    d[f"r_{col}"] = d[col] / d["ref"]
    d = d.drop(columns="ref")

print("\n2. Day value / median of the stock-month's ordinary days: median [IQR], share above 1, n")
groups = {"ordinary": d.ordinary, "election": d.cls.apply(lambda c: "election" in c),
          "fomc": d.cls.apply(lambda c: "fomc" in c), "index": d.cls.apply(lambda c: "index" in c),
          "earnings (reaction)": d.is_earn, "earnings (report day, after close)": d.is_rep,
          "any calendar": d.is_cal, "count-flagged": d.count_flag,
          "count-flagged, not on calendar": d.count_flag & ~d.is_cal}
rows = []
for name, m in groups.items():
    g = d[m]
    r = {"group": name, "n": len(g)}
    for col in ["n1", "n10", "total", "n_e0", "mu_mean"]:
        x = g[f"r_{col}"]
        r[col] = f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]"
        r[f"{col}>1"] = f"{(x > 1).mean():.2f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n2b. By stock: earnings reaction days, n1 ratio (each day)")
print(d[d.is_earn].sort_values(["sym", "date"])[["sym", "date", "n_e0", "r_n_e0", "n1", "r_n1", "r_n10", "r_mu_mean", "count_flag"]].round(2).to_string(index=False))

print("\n2c. By stock: market-wide days, median n1 ratio and share above 1")
g = d[d.is_mkt].groupby("sym")
print(pd.DataFrame({"n": g.size(), "n1 ratio": g.r_n1.median().round(2), "share>1": g.r_n1.apply(lambda x: (x > 1).mean()).round(2),
                    "n_e0 ratio": g.r_n_e0.median().round(2)}).to_string())

# 3. sign tests: share of calendar days above the ordinary median, binomial p
from scipy import stats
print("\n3. Sign test, n1 above the month's ordinary median")
for name in ["election", "fomc", "index", "earnings (reaction)", "earnings (report day, after close)", "any calendar"]:
    x = d[groups[name]].r_n1
    k, n = int((x > 1).sum()), len(x)
    print(f"  {name:36s} {k}/{n}  p = {stats.binomtest(k, n, 0.5).pvalue:.3g}")

# 4. within stock-month relations (log, demeaned by stock-month)
print("\n4. Within stock-month elasticities (OLS on demeaned logs), all days")
def within(y, x, mask=None):
    e = d if mask is None else d[mask]
    ly = np.log(e[y]); lx = np.log(e[x])
    gy = ly - ly.groupby([e.sym, e.month]).transform("mean")
    gx = lx - lx.groupby([e.sym, e.month]).transform("mean")
    b = (gx * gy).sum() / (gx * gx).sum()
    r = np.corrcoef(gx, gy)[0, 1]
    return b, r
for y, x in [("n1", "n_e0"), ("n1", "mu_mean"), ("n10", "n_e0"), ("n10", "mu_mean"), ("n1", "n_events")]:
    b, r = within(y, x)
    b2, r2 = within(y, x, d.ordinary)
    print(f"  log {y} on log {x}: elasticity {b:.2f} (corr {r:.2f}); ordinary days only {b2:.2f} ({r2:.2f})")
print("  by stock, n1 on mu_mean:")
for s, e in d.groupby("sym"):
    ly = np.log(e.n1); lx = np.log(e.mu_mean)
    gy = ly - ly.groupby(e.month).transform("mean"); gx = lx - lx.groupby(e.month).transform("mean")
    print(f"    {s:5s} {((gx * gy).sum() / (gx * gx).sum()):.2f} (corr {np.corrcoef(gx, gy)[0, 1]:.2f})")

# 5. month n1 without calendar days (event-weighted mean of day fits), against the count rule
print("\n5. Month n(1 s): event-weighted mean of day values, all days vs without flagged days")
def wmean(g, col, mask):
    w = g.n_events[mask]
    return (w * g[col][mask]).sum() / w.sum()
rows = []
for (s, m), g in d.groupby(["sym", "month"]):
    r = {"sym": s, "month": m, "all": wmean(g, "n1", np.ones(len(g), bool)), "month_fit": g.month_n1.iloc[0],
         "n_cal": int(g.is_cal.sum()), "n_count": int(g.count_flag.sum())}
    r["no_cal"] = wmean(g, "n1", ~g.is_cal) if g.is_cal.any() else np.nan
    r["no_mkt"] = wmean(g, "n1", ~g.is_mkt) if g.is_mkt.any() else np.nan
    r["no_earn"] = wmean(g, "n1", ~(g.is_earn | g.is_rep)) if (g.is_earn | g.is_rep).any() else np.nan
    r["no_count"] = wmean(g, "n1", ~g.count_flag) if g.count_flag.any() else np.nan
    r["all10"] = wmean(g, "n10", np.ones(len(g), bool))
    r["no_cal10"] = wmean(g, "n10", ~g.is_cal) if g.is_cal.any() else np.nan
    r["no_count10"] = wmean(g, "n10", ~g.count_flag) if g.count_flag.any() else np.nan
    rows.append(r)
M = pd.DataFrame(rows)
print("  weighted mean of day n1 vs month fit: median abs rel diff", (abs(M["all"] / M.month_fit - 1)).median().round(3),
      "max", (abs(M["all"] / M.month_fit - 1)).max().round(3))
for k in ["no_cal", "no_mkt", "no_earn", "no_count"]:
    ch = (M[k] / M["all"] - 1).dropna()
    print(f"  n1 change without {k[3:]:6s}: {len(ch)} stock-months, median {100*ch.median():+.1f}%, IQR [{100*ch.quantile(.25):+.1f}, {100*ch.quantile(.75):+.1f}], range [{100*ch.min():+.1f}, {100*ch.max():+.1f}], share negative {(ch < 0).mean():.2f}")
for k in ["no_cal10", "no_count10"]:
    ch = (M[k] / M["all10"] - 1).dropna()
    print(f"  n10 change without {k[3:-2]:6s}: {len(ch)} stock-months, median {100*ch.median():+.1f}%, range [{100*ch.min():+.1f}, {100*ch.max():+.1f}]")
print("  largest drops without calendar days:")
M["ch_cal"] = M.no_cal / M["all"] - 1
print(M.sort_values("ch_cal")[["sym", "month", "n_cal", "n_count", "all", "no_cal", "no_count", "ch_cal"]].head(8).round(3).to_string(index=False))

# 6. dispersion of day n1 within stock-month (ordinary days): IQR / median
x = d[d.ordinary].groupby(["sym", "month"]).n1.agg(lambda v: (v.quantile(.75) - v.quantile(.25)) / v.median())
print("\n6. Ordinary days: IQR/median of day n1 within stock-month, median over stock-months", x.median().round(2), "by stock", x.groupby("sym").median().round(2).to_dict())
d.to_csv(sys.argv[2], index=False)

"""Summaries of the cluster runs of October 16 (baseline resolution for the panel, the November
bootstrap, the full-grid raw check, the share0 decomposition, the brief-state panel, BAC by hour at
2 ms, the outlier refits), from the files under results/ (results doc section 35). Writes
baseline_bins_panel.csv, boot_201711_summary.csv, brief_states_panel.csv and byhour_ratios.csv.

    python scripts/checks/revision/cluster5_summaries.py --results results
"""
import argparse
import glob
import sys
import numpy as np
import pandas as pd

_p = argparse.ArgumentParser(); _p.add_argument("--results", default="results"); _a = _p.parse_args()
R = _a.results
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)


def arr(s):
    return np.array([float(x) for x in str(s).split(";")])


def mass_row(r, lo, H, col="excitation_per_scale"):
    m, e = arr(r.memory_s), arr(getattr(r, col))
    return float(np.sum(e * (np.exp(-lo / m) - np.exp(-H / m))))


def mass_boot(row, K, lo, H, col="excitation"):
    m = np.array([row[f"memory_{k}"] for k in range(1, K + 1)])
    e = np.array([row[f"{col}_{k}"] for k in range(1, K + 1)])
    return float(np.sum(e * (np.exp(-lo / m) - np.exp(-H / m))))


main4 = pd.read_csv(f"{R}/fits/fits_e1m_delta_K4.csv", dtype={"month": str})
main3 = pd.read_csv(f"{R}/fits/fits_e1m_h2_K3.csv", dtype={"month": str})


def main_row(sym, month):
    F = main3 if sym == "BAC" else main4
    return F[(F.sym == sym) & (F.month == month)].iloc[0]


# ------------------------------------------------------------------ A16 baseline bins
print("=" * 100); print("A16 baseline resolution, eight stocks, May and November 2017")
bb = pd.concat([pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{R}/panel/baseline_bins/*.csv"))])
rows = []
for (sym, month), g in bb.groupby(["sym", "month"]):
    m = main_row(sym, month)
    h = 0.002 if sym == "BAC" else 0.001
    cells = {30.0: m}
    for r in g.itertuples():
        cells[float(r.bin_minutes)] = r
    for b in [5.0, 10.0, 30.0, 60.0, 110.0, 330.0]:
        if b not in cells:
            continue
        r = cells[b]
        mem = arr(r.memory_s)
        rows.append({"sym": sym, "month": month, "bins": b, "aic": r.aic, "bic": r.bic, "fastest_ms": mem.min() * 1e3,
                     "slowest_s": mem.max(), "total": float(arr(r.effective_excitation).sum()),
                     "n1": mass_row(r, h, 1.0), "n10": mass_row(r, h, 10.0), "loglik": r.loglik})
B = pd.DataFrame(rows)
piv = B.pivot_table(index=["sym", "month"], columns="bins", values=["n1", "n10", "total", "slowest_s", "fastest_ms"])
print(piv["n1"].round(3).to_string()); print(piv["n10"].round(3).to_string()); print(piv["total"].round(2).to_string())
print(piv["slowest_s"].round(0).to_string()); print(piv["fastest_ms"].round(1).to_string())
# changes against 30 min
summ = []
for (sym, month), g in B.groupby(["sym", "month"]):
    g = g.set_index("bins")
    r30 = g.loc[30.0]
    d = {"sym": sym, "month": month}
    for b in [5.0, 10.0, 60.0, 110.0, 330.0]:
        if b in g.index:
            d[f"dn1_{int(b)}"] = g.loc[b].n1 - r30.n1
            d[f"dn10_{int(b)}"] = g.loc[b].n10 - r30.n10
            d[f"rn10_{int(b)}"] = g.loc[b].n10 / r30.n10 - 1
            d[f"fast_ratio_{int(b)}"] = g.loc[b].fastest_ms / r30.fastest_ms
    d["aic_best"] = g.aic.idxmin(); d["bic_best"] = g.bic.idxmin()
    d["aic_best_3"] = g.loc[[30.0, 60.0, 330.0]].aic.idxmin(); d["bic_best_3"] = g.loc[[30.0, 60.0, 330.0]].bic.idxmin()
    d["total_30"] = r30.total; d["total_330"] = g.loc[330.0].total
    summ.append(d)
Sm = pd.DataFrame(summ)
print(Sm.round(3).to_string())
up = ["dn1_60", "dn1_110", "dn1_330"]
print("max |dn1| from 30 min upward:", Sm[up].abs().max().max().round(4), "| incl. 5 and 10 min:", Sm[["dn1_5", "dn1_10"] + up].abs().max().max().round(4))
print("max |dn10| from 30 min upward:", Sm[["dn10_60", "dn10_110", "dn10_330"]].abs().max().max().round(4))
print("n10 at 5 min relative:", Sm.rn10_5.describe().round(3).to_dict()); print("n10 at 10 min relative:", Sm.rn10_10.describe().round(3).to_dict())
print("AIC best over six:", Sm.aic_best.value_counts().to_dict(), "| BIC best:", Sm.bic_best.value_counts().to_dict())
print("AIC best over 30/60/day:", Sm.aic_best_3.value_counts().to_dict(), "| BIC:", Sm.bic_best_3.value_counts().to_dict())
print("total at daily baseline:", Sm.total_330.describe().round(2).to_dict(), "at 30 min:", Sm.total_30.describe().round(2).to_dict())
print("ladder rearrangements (fastest memory ratio outside 0.7-1.4):")
for b in [5, 10, 60, 110, 330]:
    x = Sm[(Sm[f"fast_ratio_{b}"] < 0.7) | (Sm[f"fast_ratio_{b}"] > 1.4)]
    for r in x.itertuples():
        print(f"  {r.sym} {r.month} B{b}: fastest ratio {getattr(r, f'fast_ratio_{b}'):.2f}, dn1 {getattr(r, f'dn1_{b}'):+.3f}")
# excluding rearranged cells
ok = Sm.copy()
for b in [5, 10, 60, 110, 330]:
    bad = (ok[f"fast_ratio_{b}"] < 0.7) | (ok[f"fast_ratio_{b}"] > 1.4)
    ok.loc[bad, [f"dn1_{b}", f"dn10_{b}"]] = np.nan
print("excluding rearranged cells: max |dn1| upward", ok[up].abs().max().max().round(4), "all", ok[["dn1_5", "dn1_10"] + up].abs().max().max().round(4),
      "| max |dn10| upward", ok[["dn10_60", "dn10_110", "dn10_330"]].abs().max().max().round(4))

print("\n-- boot_bins INTC May 5 and 10 min (30 reps): bias of total and masses")
for b in [5, 10]:
    p = f"{R}/bootstrap/bins_201705/INTC_201705_B{b}"
    fit = pd.read_csv(f"{p}/fit.csv").iloc[0]; rep = pd.read_csv(f"{p}/replicates.csv")
    K = int(fit.K); h = fit.delta
    est = {"total": fit.effective_br, "n1": mass_boot(fit, K, h, 1.0, "effective_excitation"), "n10": mass_boot(fit, K, h, 10.0, "effective_excitation")}
    bm = {"total": rep.effective_br.mean(), "n1": rep.apply(lambda r: mass_boot(r, K, h, 1.0, "effective_excitation"), axis=1).mean(),
          "n10": rep.apply(lambda r: mass_boot(r, K, h, 10.0, "effective_excitation"), axis=1).mean()}
    sd = {"total": rep.effective_br.std(), "n1": rep.apply(lambda r: mass_boot(r, K, h, 1.0, "effective_excitation"), axis=1).std()}
    print(f"  B{b}: total {est['total']:.3f} bias {bm['total'] - est['total']:+.3f} (sd {sd['total']:.3f}, MC se {sd['total'] / np.sqrt(len(rep)):.3f}); "
          f"n1 {est['n1']:.3f} bias {bm['n1'] - est['n1']:+.4f}; n10 {est['n10']:.3f} bias {bm['n10'] - est['n10']:+.4f}; "
          f"slowest {max(fit[f'memory_{k}'] for k in range(1, K + 1)):.1f}s, reps {len(rep)}")
    print("   check vs tab: n1 from dead time with excitation (not effective):", round(mass_boot(fit, K, h, 1.0, "excitation"), 4),
          "n10", round(mass_boot(fit, K, h, 10.0, "excitation"), 4), "aic", round(2 * (K + 2 * K + 22 * int(330 / b)) - 2 * fit.loglik), "bic", round(fit.bic))

# ------------------------------------------------------------------ A14 November bootstrap
print("\n" + "=" * 100); print("A14 November 2017 parametric bootstrap, 99 replicates")
rows = []
for sym in ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]:
    p = f"{R}/bootstrap/parametric_201711/{sym}_201711"
    fit = pd.read_csv(f"{p}/fit.csv").iloc[0]; rep = pd.read_csv(f"{p}/replicates.csv")
    K = int(fit.K); h = fit.delta
    m = main_row(sym, "201711")
    n1 = mass_boot(fit, K, h, 1.0); n10 = mass_boot(fit, K, h, 10.0)
    r1 = rep.apply(lambda r: mass_boot(r, K, h, 1.0), axis=1); r10 = rep.apply(lambda r: mass_boot(r, K, h, 10.0), axis=1)
    slow = max(fit[f"memory_{k}"] for k in range(1, K + 1)); kslow = max(range(1, K + 1), key=lambda k: fit[f"memory_{k}"])
    rows.append({"sym": sym, "K": K, "h": h, "n_events": fit.n_events, "loglik_fit": fit.loglik, "loglik_main": m.loglik,
                 "ks_up": fit.ks_up, "ks_down": fit.ks_down, "null95_up": rep.ks_up.quantile(.95), "null95_down": rep.ks_down.quantile(.95),
                 "p_up": (1 + (rep.ks_up >= fit.ks_up).sum()) / (len(rep) + 1), "p_down": (1 + (rep.ks_down >= fit.ks_down).sum()) / (len(rep) + 1),
                 "eff_br": fit.effective_br, "bias_br": rep.effective_br.mean() - fit.effective_br, "se_br": rep.effective_br.std(),
                 "n1": n1, "bias_n1": r1.mean() - n1, "se_n1": r1.std(), "n10": n10, "bias_n10": r10.mean() - n10, "se_n10": r10.std(),
                 "slowest_s": slow, "slow_ratio": rep[f"memory_{kslow}"].mean() / slow, "n_reps": len(rep)})
N = pd.DataFrame(rows)
print(N.round(4).to_string(index=False))
print("rejections at 5%: up", int((N.p_up <= 0.05).sum()), "down", int((N.p_down <= 0.05).sum()), "stocks with at least one", int(((N.p_up <= 0.05) | (N.p_down <= 0.05)).sum()))
print("KS up range", N.ks_up.min().round(4), N.ks_up.max().round(4), "null 95th up", N.null95_up.min().round(4), N.null95_up.max().round(4))
print("KS down range", N.ks_down.min().round(4), N.ks_down.max().round(4), "null 95th down", N.null95_down.min().round(4), N.null95_down.max().round(4))
print("se n1 range", N.se_n1.min().round(4), N.se_n1.max().round(4), "| bias n1 range", N.bias_n1.min().round(4), N.bias_n1.max().round(4))
print("bias br range", N.bias_br.min().round(3), N.bias_br.max().round(3), "| se br", N.se_br.min().round(3), N.se_br.max().round(3), "| slow ratio", N.slow_ratio.min().round(2), N.slow_ratio.max().round(2))

# ------------------------------------------------------------------ A13a full-grid raw check
print("\n" + "=" * 100); print("A13a raw K = 4 May: full+ladder against default+ladder")
fg = pd.concat([pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{R}/fits_e0_k4_fullgrid_201705.csv"))])
pan = pd.read_csv(f"{R}/fits_e0_k4_panel.csv", dtype={"month": str})
pan = pan[(pan.month == "201705") & (pan.K == 4)]
for r in fg.itertuples():
    q = pan[pan.sym == r.sym].iloc[0]
    mf, ef = arr(r.memory_s), arr(r.excitation_per_scale); mq, eq = arr(q.memory_s), arr(q.excitation_per_scale)
    print(f"  {r.sym}: loglik full {r.loglik:.2f} default {q.loglik:.2f} diff {r.loglik - q.loglik:+.2f} | n1 {mass_row(r, 0, 1):.4f} / {mass_row(q, 0, 1):.4f} "
          f"| n10 {mass_row(r, 0, 10):.4f} / {mass_row(q, 0, 10):.4f} | subms {ef[mf < 1e-3].sum():.4f} / {eq[mq < 1e-3].sum():.4f} | starts {q.starts}")

# ------------------------------------------------------------------ A13b share0 decomposition
print("\n" + "=" * 100); print("A13b share0 with the TRUE fit: bias decomposition (median over 25 reps)")
sh = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/sim_share0_truth.csv"))])
sh["key"] = sh.definition + " d=" + sh.delta.astype(str)
sh["rel1"] = sh.n1 / sh.true_n1 - 1; sh["rel10"] = sh.n10 / sh.true_n10 - 1; sh["relbr"] = sh.branching_ratio / sh.true_br - 1
sh["merged_share"] = 1 - sh.n_e1m / sh.n_clean
print(sh.groupby(["sym", "key"])[["rel1", "rel10", "relbr"]].median().round(4).unstack("key").to_string())
print(sh.groupby("sym")[["true_n1", "clean_gap_lt1ms", "merged_share", "n_clean"]].median().round(4).to_string())
print("reps per stock:", sh.groupby("sym").rep.nunique().to_dict(), "rows", len(sh))

# ------------------------------------------------------------------ A20 brief states panel
print("\n" + "=" * 100); print("A20 brief-state panel")
cs = pd.concat([pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{R}/panel/brief_states/*/calibration_summary.csv"))])
print(cs.shape, cs.groupby("sym").size().to_dict())
cols = ["share_direct", "share_via_lock", "share_via_widen", "flickers_per_move", "flicker_share_lock", "median_lock_chain_ms", "p90_lock_chain_ms", "median_widen_chain_ms", "median_flicker_ms", "share_time_1tick_excl_nonpos"]
print(cs.groupby("sym")[cols].agg(["median", "min", "max"]).round(3).T.to_string())
old = pd.read_csv(f"{R}/simulation/calibration/calibration_summary.csv", dtype={"month": str})
mm = cs.merge(old, on=["sym", "month"], suffixes=("", "_old"))
print("May comparison with the earlier calibration (days, share_via_lock, median lock):")
print(mm[mm.month == "201705"][["sym", "days", "days_old", "share_via_lock", "share_via_lock_old", "median_lock_chain_ms", "median_lock_chain_ms_old", "flickers_per_move", "flickers_per_move_old"]].round(3).to_string(index=False))
print("monthly median lock chain (ms) across the eight stocks, and range:")
g = cs.groupby("month").median_lock_chain_ms
print(pd.DataFrame({"median": g.median(), "min": g.min(), "max": g.max()}).round(2).T.to_string())
cs.to_csv(f"{R}/brief_states_panel.csv", index=False)

# ------------------------------------------------------------------ by hour with BAC at 2 ms
print("\n" + "=" * 100); print("By hour: BAC at h = 2 ms, K = 3")
bh = pd.concat([pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{R}/panel/by_hour/by_hour_*.csv"))])
bac_old = bh[bh.sym == "BAC"].copy()
bh = pd.concat([bh[bh.sym != "BAC"], pd.read_csv(f"{R}/by_hour_BAC_h2.csv", dtype={"month": str})])
W = {"10:00-11:00": "first", "11:00-14:30": "mid", "14:30-15:30": "last"}
bh["w"] = bh.window.map(W)
P = bh.pivot_table(index=["sym", "month"], columns="w", values=["n1", "n10", "events_per_hour"])
R = pd.DataFrame({"r1_first": P.n1["first"] / P.n1["mid"], "r10_first": P.n10["first"] / P.n10["mid"], "r1_last": P.n1["last"] / P.n1["mid"],
                  "act_first": P.events_per_hour["first"] / P.events_per_hour["mid"]}).reset_index()
tab = R.groupby("sym").agg(r1_med=("r1_first", "median"), r1_min=("r1_first", "min"), r1_max=("r1_first", "max"), months_gt1=("r1_first", lambda x: int((x > 1).sum())),
                           n=("r1_first", "size"), r10_med=("r10_first", "median"), last_med=("r1_last", "median"), act_med=("act_first", "median"))
print(tab.round(2).to_string())
print("pooled: first>mid in", int((R.r1_first > 1).sum()), "of", len(R), "| median", R.r1_first.median().round(2), "IQR", R.r1_first.quantile(.25).round(2), R.r1_first.quantile(.75).round(2),
      "| n10 median", R.r10_first.median().round(2), "| last median", R.r1_last.median().round(2))
bh["fast_ms"] = bh.memory_s.apply(lambda s: arr(s).min() * 1e3); bh["slow_s"] = bh.memory_s.apply(lambda s: arr(s).max())
print("median fastest memory by window (ms):", bh.groupby("w").fast_ms.median().round(1).to_dict(), "slowest (s):", bh.groupby("w").slow_s.median().round(0).to_dict())
print("  excluding BAC:", bh[bh.sym != "BAC"].groupby("w").fast_ms.median().round(1).to_dict())
# within-day elasticity: log n1 on log events per hour, demeaned by stock-month
ly = np.log(bh.n1); lx = np.log(bh.events_per_hour)
gy = ly - ly.groupby([bh.sym, bh.month]).transform("mean"); gx = lx - lx.groupby([bh.sym, bh.month]).transform("mean")
print("within-day elasticity", round((gx * gy).sum() / (gx * gx).sum(), 2), "corr", round(np.corrcoef(gx, gy)[0, 1], 2))
print("BAC old row (1 ms, K=4) vs new (2 ms, K=3):")
for name, d in [("old", bac_old), ("new", bh[bh.sym == "BAC"])]:
    d = d.copy(); d["w"] = d.window.map(W)
    p = d.pivot_table(index="month", columns="w", values=["n1", "n10", "events_per_hour"])
    r1 = p.n1["first"] / p.n1["mid"]; r10 = p.n10["first"] / p.n10["mid"]; rl = p.n1["last"] / p.n1["mid"]
    print(f"  {name}: first/mid n1 median {r1.median():.2f} ({r1.min():.2f}-{r1.max():.2f}), months>1 {int((r1 > 1).sum())}/{len(r1)}, n10 {r10.median():.2f}, last {rl.median():.2f}, last>1 {int((rl > 1).sum())}")
    print("   by month:", r1.round(2).to_dict())

# ------------------------------------------------------------------ outliers cross-check
print("\n" + "=" * 100); print("Outlier refits (count rule) against the day-level weighted means")
ol = pd.concat([pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(f"{R}/panel/outliers/outliers_*.csv"))])
ol = ol[ol.n_flagged > 0]
ch = ol.refit_n1 / ol.main_n1 - 1
print("refit: ", len(ol), "stock-months, median", round(100 * ch.median(), 1), "% range", round(100 * ch.min(), 1), round(100 * ch.max(), 1), "share negative", round((ch < 0).mean(), 2))
bd = pd.read_csv(f"{R}/by_day_fixed_timescales.csv", dtype={"month": str, "date": str})
rows = []
for (s, m), g in bd.groupby(["sym", "month"]):
    if g.count_flag.any():
        w = g.n_events
        rows.append({"sym": s, "month": m, "wm_change": (w[~g.count_flag] * g.n1[~g.count_flag]).sum() / w[~g.count_flag].sum() / ((w * g.n1).sum() / w.sum()) - 1})
wm = pd.DataFrame(rows).merge(ol[["sym", "month", "main_n1", "refit_n1"]], on=["sym", "month"])
wm["refit_change"] = wm.refit_n1 / wm.main_n1 - 1
print("paired (same stock-months):", len(wm), "median refit", round(100 * wm.refit_change.median(), 1), "median weighted-mean", round(100 * wm.wm_change.median(), 1), "corr", round(wm[["refit_change", "wm_change"]].corr().iloc[0, 1], 2))
B.to_csv(f"{R}/baseline_bins_panel.csv", index=False); N.to_csv(f"{R}/boot_201711_summary.csv", index=False); R.to_csv(f"{R}/panel/byhour_ratios.csv", index=False)

"""Rebuild every numerical table of the paper from the files in results/.

Each function prints the rows of one table in the paper's row order, with the paper's rounding,
so that the output can be compared line by line with docs/paper/paper.tex. No TAQ data are read:
everything comes from the fitted parameters, bootstrap replicates, simulation rows and panel
checks saved under results/ (see results/README.md for the file manifest).

    python scripts/paper_tables.py                 # all tables
    python scripts/paper_tables.py main masses     # selected tables, by label (tab:<name>)

Figures are made by scripts/make_paper_figures.py and the legends page by scripts/make_figure_legends.py.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
ORDER = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]
SIM = ["INTC", "MDLZ", "BAC", "TGT"]


# ----------------------------------------------------------------------------- helpers
def vec(s):
    return np.array([float(x) for x in str(s).split(";")])


def load_fits(path, **kw):
    d = pd.read_csv(path, dtype={"month": str}, **kw)
    d["mem"] = d.memory_s.map(vec)
    d["exc"] = d.excitation_per_scale.map(vec)
    if "delta" not in d:
        d["delta"] = 0.0
    return d


def mass(mem, exc, lo, H):
    return float(np.sum(exc * (np.exp(-lo / mem) - np.exp(-H / mem))))


def add_masses(d):
    d = d.copy()
    d["n1"] = [mass(m, e, dl, 1.0) for m, e, dl in zip(d.mem, d.exc, d.delta)]
    d["n10"] = [mass(m, e, dl, 10.0) for m, e, dl in zip(d.mem, d.exc, d.delta)]
    d["n1ms"] = [mass(m, e, 0.0, 0.001) for m, e in zip(d.mem, d.exc)]
    d["n1_from5ms"] = [mass(m, e, 0.005, 1.0) for m, e in zip(d.mem, d.exc)]
    d["n10_from5ms"] = [mass(m, e, 0.005, 10.0) for m, e in zip(d.mem, d.exc)]
    d["eff_br"] = [float(np.sum(e * np.exp(-dl / m))) for m, e, dl in zip(d.mem, d.exc, d.delta)]
    d["fast_mem"] = d.mem.map(np.min)
    d["slow_mem"] = d.mem.map(np.max)
    d["epd"] = d.n_events / d.n_days
    return d


def main_panel():
    """The main clean panel: E1m, K = 4, 1 ms, except BAC at K = 3 and 2 ms."""
    k4 = load_fits(R / "fits/fits_e1m_delta_K4.csv")
    k3 = load_fits(R / "fits/fits_e1m_h2_K3.csv")
    d = pd.concat([k4[k4.sym != "BAC"], k3[k3.sym == "BAC"]]).sort_values(["sym", "month"]).reset_index(drop=True)
    return add_masses(d)


def raw_panel():
    return add_masses(load_fits(R / "fits_e0_k4_panel.csv"))


def q(s, nd=2, sep="--"):
    return f"{s.median():.{nd}f} [{s.quantile(.25):.{nd}f}{sep}{s.quantile(.75):.{nd}f}]"


def fmt_mem(m):
    return f"{m * 1e3:.1f} ms" if m < 0.1 else (f"{m:.2f} s" if m < 10 else f"{m:.0f} s")


def fmt_mem_iqr(s):
    """Median [IQR] of a memory, in the paper's style: ms below 0.5 s (one decimal under 20 ms), seconds above
    (two significant figures under 10 s, none above)."""
    lo, med, hi = s.quantile(.25), s.median(), s.quantile(.75)
    if med < 0.5:
        f = lambda x: f"{x * 1e3:.1f}" if x < 0.02 else f"{x * 1e3:.0f}"
        return f"{f(med)} ms [{f(lo)}--{f(hi)}]"
    f = lambda x: f"{x:.2g}" if x < 10 else f"{x:.0f}"
    return f"{f(med)} s [{f(lo)}--{f(hi)}]"


def head(label):
    print(f"\n===== tab:{label}")


# ----------------------------------------------------------------------------- tables
def tab_main():
    head("main")
    d = main_panel()
    for s in ORDER:
        g = d[d.sym == s]
        K = int(g.K.iloc[0])
        mems = pd.DataFrame(g.mem.tolist())   # columns fastest..slowest (fits store memories ascending)
        excs = pd.DataFrame([vec(x) for x in g.effective_excitation])
        cells = [fmt_mem_iqr(mems[k]) for k in range(K)]
        if K == 3:
            cells = cells[:2] + ["---"] + cells[2:]
        exc = " $\\cdot$ ".join(f"{excs[k].median():.2f}" for k in range(K))
        print(f"{s}{' ($K=3$)' if K == 3 else ''} & " + " & ".join(cells) + f" & {exc} \\\\")


def tab_masses():
    head("masses")
    d = main_panel()
    sd = pd.read_csv(R / "blockboot_panel_sd.csv", dtype={"month": str})
    for s in ORDER:
        g = d[d.sym == s]; b = sd[sd.sym == s]
        print(f"{s} & {q(g.n1)} ({b.sd_n1.median():.3f}) & {q(g.n10)} ({b.sd_n10.median():.3f}) & "
              f"{g.eff_br.median():.2f} [{g.eff_br.min():.2f}--{g.eff_br.max():.2f}] \\\\")


def tab_horizon():
    head("horizon")
    raw = raw_panel().set_index(["sym", "month"]); clean = main_panel().set_index(["sym", "month"])
    e1 = add_masses(load_fits(R / "fits/fits_K3_nodelta_panel.csv"))
    e1 = e1[e1.definition == "E1"].set_index(["sym", "month"])
    m = raw.join(clean[["n1", "n10", "n1_from5ms", "n10_from5ms", "branching_ratio", "eff_br", "delta"]], rsuffix="_c")
    m["raw_from_h_1"] = [mass(a, b, h, 1.0) for a, b, h in zip(m.mem, m.exc, m.delta_c)]
    m["raw_from_h_10"] = [mass(a, b, h, 10.0) for a, b, h in zip(m.mem, m.exc, m.delta_c)]
    print(f"Clean (\\Em), from $h$ & {q(m.n1_c)} & {q(m.n10_c)} & {q(m.branching_ratio_c)} \\\\")
    print(f"Raw (\\Eo), from 0 & {q(m.n1)} & {q(m.n10)} & {q(m.branching_ratio)} \\\\")
    print(f"\\quad of which in the first millisecond & {q(m.n1ms)} & & \\\\")
    print(f"Locks merged only (\\Ei), from 0 & {q(e1.n1)} & {q(e1.n10)} & \\\\")
    print(f"Raw $\\div$ clean, raw from 0 & {q(m.n1 / m.n1_c, 1)} & {q(m.n10 / m.n10_c, 1)} & {q(m.branching_ratio / m.branching_ratio_c)} \\\\")
    print(f"Raw $\\div$ clean, both from $h$ & {q(m.raw_from_h_1 / m.n1_c)} & {q(m.raw_from_h_10 / m.n10_c)} & \\\\")
    print(f"Raw $\\div$ clean, both from 5 ms & {q(m.n1_from5ms / m.n1_from5ms_c)} & {q(m.n10_from5ms / m.n10_from5ms_c)} & \\\\")


def tab_blockboot():
    head("blockboot")
    raw = raw_panel(); clean = main_panel()
    bl = pd.read_csv(R / "blockboot_may_blocks.csv"); bl = bl[bl.block == 5].set_index("sym")
    for s in ORDER:
        rep = pd.read_csv(R / f"bootstrap/parametric_201705/{s}_201705/replicates.csv")
        fit = pd.read_csv(R / f"bootstrap/parametric_201705/{s}_201705/fit.csv").iloc[0]
        K = int(fit.K); h = float(fit.delta)
        n1 = rep.apply(lambda r: sum(r[f"excitation_{k}"] * (np.exp(-h / r[f"memory_{k}"]) - np.exp(-1.0 / r[f"memory_{k}"])) for k in range(1, K + 1)), axis=1)
        r = raw[(raw.sym == s) & (raw.month == "201705")].iloc[0]; c = clean[(clean.sym == s) & (clean.month == "201705")].iloc[0]
        b = bl.loc[s]
        print(f"{s} & {n1.std():.3f} & {b.sd_clean_n1:.3f} & {b.sd_clean_n10:.3f} & {r.n1 / c.n1:.1f}; {b.ratio1_med:.1f} [{b.ratio1_lo:.1f}, {b.ratio1_hi:.1f}] "
              f"& {b.ratio10_med:.1f} [{b.ratio10_lo:.1f}, {b.ratio10_hi:.1f}] \\\\")


def tab_threshold():
    head("threshold")
    cur = pd.read_csv(R / "fits/threshold_curve/curve.csv", dtype={"month": str})
    main = main_panel().set_index(["sym", "month"])
    k4 = add_masses(load_fits(R / "fits/fits_e1m_delta_K4.csv")).set_index(["sym", "month"])   # 1 ms reference at K = 4 for every stock
    rows = {}
    for h in (0.00025, 0.0005, 0.002, 0.005):
        f = add_masses(load_fits(R / f"fits/threshold_curve/fits_threshold_{h}.csv")).set_index(["sym", "month"])
        f = f.join(k4[["n_events", "mem", "eff_br", "n1_from5ms", "n10_from5ms"]], rsuffix="_1")
        mr = pd.DataFrame({k: [np.sort(a)[k] / np.sort(b)[k] for a, b in zip(f.mem, f.mem_1)] for k in range(4)})
        rows[h] = dict(ev=(f.n_events / f.n_events_1).median(), m=[mr[k].median() for k in range(4)], d=(f.eff_br - f.eff_br_1).median(),
                       r1=(f.n1_from5ms / f.n1_from5ms_1).median(), r10=(f.n10_from5ms / f.n10_from5ms_1).median(), n=len(f))
    for h, lab in ((0.00025, "0.25 ms"), (0.0005, "0.5 ms"), (0.001, "1 ms"), (0.002, "2 ms"), (0.005, "5 ms")):
        if h == 0.001:
            print("1 ms & 1 & 1 & 1 & 1 & 1 & 0 & 1 & 1 \\\\"); continue
        r = rows[h]
        dd = f"{r['d']:+.3f}" if abs(r['d']) >= 0.0005 else "0.000"
        print(f"{lab} & {r['ev']:.2f} & " + " & ".join(f"{x:.2f}" for x in r["m"]) + f" & ${dd}$ & {r['r1']:.2f} & {r['r10']:.2f} \\\\   % {r['n']} stock-months")


def tab_baseline():
    head("baseline")
    bb = pd.read_csv(R / "baseline_bins_panel.csv", dtype={"month": str}); bb = bb[(bb.sym == "INTC") & (bb.month == "201705")].set_index("bins")
    bias = {}
    for b, lab in ((5, "B5"), (10, "B10"), (60, "B60"), (110, "B110"), (330, "B330")):
        p = R / f"bootstrap/bins_201705/INTC_201705_{lab}"
        fit = pd.read_csv(p / "fit.csv").iloc[0]; rep = pd.read_csv(p / "replicates.csv")
        bias[b] = rep.effective_br.mean() - fit.effective_br
    s30 = pd.read_csv(R / "bootstrap/parametric_201705_summary.csv").set_index("sym").loc["INTC"]
    bias[30] = s30.eBR_bias
    for b, lab in ((5, "5 min"), (10, "10 min"), (30, "30 min (main)"), (60, "60 min"), (110, "110 min"), (330, "one per day")):
        r = bb.loc[float(b)]
        print(f"{lab} & {r.aic:,.0f} & {r.bic:,.0f} & {r.slowest_s:.0f} s & {r.total:.2f} & ${bias[b]:+.3f}$ & {r.n1:.3f} & {r.n10:.2f} \\\\")


def tab_byhour():
    head("byhour")
    bh = pd.concat([pd.read_csv(f, dtype={"month": str}) for f in sorted(glob.glob(str(R / "panel/by_hour/by_hour_*.csv")))])
    bh = pd.concat([bh[bh.sym != "BAC"], pd.read_csv(R / "by_hour_BAC_h2.csv", dtype={"month": str})])
    W = {"10:00-11:00": "first", "11:00-14:30": "mid", "14:30-15:30": "last"}
    bh["w"] = bh.window.map(W)
    P = bh.pivot_table(index=["sym", "month"], columns="w", values=["n1", "n10", "events_per_hour"])
    Rt = pd.DataFrame({"r1": P.n1["first"] / P.n1["mid"], "r10": P.n10["first"] / P.n10["mid"], "rl": P.n1["last"] / P.n1["mid"],
                       "act": P.events_per_hour["first"] / P.events_per_hour["mid"]}).reset_index()
    for s in ORDER:
        g = Rt[Rt.sym == s]
        print(f"{s} & {g.r1.median():.2f} ({g.r1.min():.2f}--{g.r1.max():.2f}) & {int((g.r1 > 1).sum())} / {len(g)} & {g.r10.median():.2f} & {g.rl.median():.2f} & {g.act.median():.2f} \\\\")


def tab_regime():
    head("regime")
    t = pd.read_csv(R / "panel/tick_regime.csv", dtype={"month": str}).set_index(["sym", "month"])
    raw = raw_panel().set_index(["sym", "month"]); clean = main_panel().set_index(["sym", "month"])
    # time at one tick excluding locked and crossed states, and the regime rule of Section 6.1
    t1 = 100 * t.pct_time_1tick / (100 - t.pct_time_locked)
    t = t.assign(pct_time_1tick=t1, regime=np.where(t1 >= 99, "large", np.where(t1 >= 90, "intermediate", "small")))[["regime", "pct_time_1tick", "changes_per_day", "pct_changes_lock"]]
    t = t.join(raw[["n1", "n1ms"]].rename(columns={"n1": "raw_n1", "n1ms": "raw_n1ms"})).join(clean[["n1", "fast_mem", "slow_mem"]])
    t["ratio"] = t.raw_n1 / t.n1
    cols = {}
    for reg in ("large", "intermediate", "small"):
        g = t[t.regime == reg]
        cols[reg] = (len(g), g.pct_time_1tick.median(), g.changes_per_day.median(), g.pct_changes_lock.median(), g.raw_n1ms.median(), g.raw_n1.median(), g.n1.median(), g.ratio.median(), g.fast_mem.median(), g.slow_mem.median())
    print("regimes (n): " + ", ".join(f"{k} {v[0]}" for k, v in cols.items()))
    rows = [("Time at one tick", lambda v: f"{v[1]:.2f}\\%" if v[1] > 99.9 else f"{v[1]:.1f}\\%"), ("NBBO changes a day", lambda v: f"{v[2]:,.0f}"),
            ("Changes that lock or cross", lambda v: f"{v[3]:.0f}\\%"), ("Raw kernel mass below $1\\ms$", lambda v: f"{v[4]:.2f}"),
            ("Raw $n(1\\sx)$", lambda v: f"{v[5]:.2f}"), ("Clean $n(1\\sx)$", lambda v: f"{v[6]:.2f}"), ("Raw $\\div$ clean $n(1\\sx)$", lambda v: f"{v[7]:.1f}"),
            ("Clean fastest / slowest memory", lambda v: f"{v[8] * 1e3:.1f} ms / {v[9]:.0f} s")]
    for name, f in rows:
        print(f"{name} & " + " & ".join(f(cols[r]) for r in ("large", "intermediate", "small")) + " \\\\")


def tab_fullygated():
    head("fullygated")
    d = pd.read_csv(R / "fully_gated_201705.csv").set_index("sym")
    for s in SIM:
        r = d.loc[s]
        print(f"{s} & {r.fastest_loss_pct:.2g}\\% & {r.eff_br:.4f} & {r.fully_gated:.4f} \\\\")


def tab_priceunits():
    head("priceunits")
    d = pd.read_csv(R / "a8_priceunits_k4.csv").set_index(["sym", "definition"])
    for s in SIM:
        for dfn, lab in (("E0", "\\Eo"), ("E1m", "\\Em")):
            r = d.loc[(s, dfn)]
            cells = " & ".join(f"${r[f'data_{H}']:+.3f}$ ({r[f'se_day_{H}']:.3f}) & ${r[f'paths_{H}']:+.3f}$" for H in ("0.1", "1.0", "10.0"))
            print(f"{s if dfn == 'E0' else ''} & {lab} & {cells} \\\\")


def tab_ibm():
    head("ibm")
    d = pd.read_csv(R / "fits/lee2023_replication_fits.csv"); d = d[(d["mode"] == "lee") & (d.K == 2)]
    for s in ("IBM", "AAPL"):
        for dfn, lab in (("E0", "\\Eo"), ("E1m", "\\Em")):
            g = d[(d.sym == s) & (d.definition == dfn)]
            p = lambda c, nd, f=1: f"{g[c].median() * f:.{nd}f} [{g[c].quantile(.1) * f:.{nd}f}--{g[c].quantile(.9) * f:.{nd}f}]"
            mm = p('mem1_ms', 2 if g.mem1_ms.median() < 5 else 1).replace(" [", " ms [")
            print(f"{s}, {lab} & {mm} & {g.share1.median() * 100:.0f}\\% [{g.share1.quantile(.1) * 100:.0f}--{g.share1.quantile(.9) * 100:.0f}] & {p('branching', 2)} \\\\   % {len(g)} days")


def tab_events():
    head("events")
    gs = pd.read_csv(R / "panel/gapshares_monthly.csv", dtype={"month": str}); gs = gs[gs.month == "201705"].set_index("sym")
    tr = pd.read_csv(R / "panel/tick_regime_merged.csv", dtype={"month": str}); tr = tr[tr.month == "201705"].set_index("sym")
    k3 = pd.read_csv(R / "fits/fits_K3_nodelta_panel.csv", dtype={"month": str}); k3 = k3[k3.month == "201705"]
    ev = {d: k3[k3.definition == d].set_index("sym").n_events for d in ("E0", "E1", "E1m")}
    main = load_fits(R / "fits/fits_e1m_delta_K4.csv"); main = main[main.month == "201705"].set_index("sym")   # E1m at 1 ms for every stock
    row = lambda name, f: print(f"{name} & " + " & ".join(f(s) for s in ORDER) + " \\\\")
    row("\\% time at 1 tick", lambda s: f"{tr.loc[s].pct_time_1tick:.1f}")
    row("\\% states locked or crossed", lambda s: f"{gs.loc[s].pct_states_nonpos:.1f}")
    row("\\Eo{} events", lambda s: f"{ev['E0'][s]:,}")
    row("\\Eo: \\% gaps $< 1\\ms$", lambda s: f"{gs.loc[s].gap_lt1ms_E0 * 100:.1f}")
    row("\\Ei{} events", lambda s: f"{ev['E1'][s]:,}")
    row("\\Ei: \\% gaps $< 1\\ms$", lambda s: f"{gs.loc[s].gap_lt1ms_E1 * 100:.1f}")
    row("\\Em{} events", lambda s: f"{main.loc[s].n_events:,}")
    row("\\Em: median gap (s)", lambda s: f"{gs.loc[s].median_gap_E1m:.1f}")


def tab_sim():
    head("simrecovery")
    d = pd.read_csv(R / "simulation/batch1_replicates.csv"); base = d[d.experiment == "base"]
    for c in ("memory_s", "excitation", "true_memory_s", "true_excitation"):
        base[c + "_v"] = base[c].map(vec)
    base["eff_br"] = [float(np.sum(e * np.exp(-dl / m))) for m, e, dl in zip(base.memory_s_v, base.excitation_v, base.delta)]
    cl = base[(base.definition == "E1m") & (base.K == 4) & (base.delta == 0.001) & (base.pool == "all")]
    for s in SIM:
        g = cl[cl.sym == s]
        t1, t10, tb = g.true_n1.iloc[0], g.true_n10.iloc[0], g.true_br.iloc[0]
        print(f"{s} & {t1:.3f} & {g.n1.median():.3f} ({g.n1.std():.3f}), ${100 * (g.n1.median() / t1 - 1):+.0f}\\%$ & {t10:.3f} & {g.n10.median():.3f} ({g.n10.std():.3f}), "
              f"${100 * (g.n10.median() / t10 - 1):+.0f}\\%$ & {tb:.3f} & {g.branching_ratio.median():.3f}, ${100 * (g.branching_ratio.median() / tb - 1):+.0f}\\%$ \\\\   % {len(g)} reps; total = branching ratio (eq:br)")
    head("simraw")
    raw = base[(base.definition == "E0") & (base.K == 4) & (base.pool == "all")].set_index(["sym", "rep"])
    cl = cl.set_index(["sym", "rep"])
    j = raw.join(cl[["n1", "n10", "branching_ratio", "n_events", "baseline_events"]], rsuffix="_c")
    j["n1ms"] = [mass(m, e, 0.0, 0.001) for m, e in zip(j.memory_s_v, j.excitation_v)]
    j["ident"] = j.branching_ratio - (j.e0_gap_lt1ms + j.branching_ratio_c * j.n_events_c / j.n_events)
    cw = pd.read_csv(R / "common_window_k4.csv", dtype={"month": str}); cw = cw[cw.month == "201705"].set_index("sym")
    rawp = raw_panel(); rawp = rawp[rawp.month == "201705"].set_index("sym")
    cleanp = main_panel(); cleanp = cleanp[cleanp.month == "201705"].set_index("sym")
    gs = pd.read_csv(R / "panel/gapshares_monthly.csv", dtype={"month": str}); gs = gs[gs.month == "201705"].set_index("sym")
    for s in SIM:
        g = j.loc[s]
        r, c = rawp.loc[s], cleanp.loc[s]
        ident_d = r.branching_ratio - (gs.loc[s].gap_lt1ms_E0 + c.branching_ratio * c.n_events / r.n_events)
        base_d = r.n_events * (1 - r.branching_ratio) / (c.n_events * (1 - c.branching_ratio))   # baseline (immigrant) events N(1 - n), raw over clean
        print(f"{s} & {g.n1.median() / g.n1_c.median():.1f} / {cw.loc[s].ratio_0_1:.1f} & {g.n10.median() / g.n10_c.median():.2f} / {cw.loc[s].ratio_0_10:.2f} & {g.n1ms.median():.2f} / {r.n1ms:.2f} "
              f"& ${g.ident.median():+.2f}$ / ${ident_d:+.2f}$ & {g.baseline_events.median() / g.baseline_events_c.median():.2f} / {base_d:.2f} \\\\   % simulation: ratios of medians over replicates; relation with the clean total (eq:br)")


TABLES = {"main": tab_main, "masses": tab_masses, "horizon": tab_horizon, "blockboot": tab_blockboot, "threshold": tab_threshold,
          "baseline": tab_baseline, "byhour": tab_byhour, "regime": tab_regime, "fullygated": tab_fullygated, "priceunits": tab_priceunits,
          "ibm": tab_ibm, "events": tab_events, "sim": tab_sim}

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    for name in (sys.argv[1:] or TABLES):
        TABLES[name]()

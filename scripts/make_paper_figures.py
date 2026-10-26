"""Figures for Section 6 of the paper.

fig_kernel   the fitted kernel on raw (E0, K = 4) and clean (E1m, main specification) events, INTC and
             TGT, May 2017, drawn as t * phi(t) on a log time axis so that the area under a curve
             between two times is the kernel mass there;
fig_months   kernel mass within one second against clean events per day, one panel per stock, the
             112 stock-months of the main panel, with the within-stock slope of log n(1 s) on log
             events per day.

    python scripts/make_paper_figures.py --fits-e1m DATA/fits_e1m_delta_K4.csv --fits-e0 results/fits_e0_k4_panel.csv \\
        --panel-sd results/blockboot_panel_sd.csv --by-day results/by_day_fixed_timescales.csv --out docs/paper/figures
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "dejavusans", "font.size": 9, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
STOCKS = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]


def parse(s):
    return np.array([float(x) for x in str(s).split(";")])


def kernel(row, t):
    """phi(t) = sum_k (a_k + c_k) / m_k exp(-t / m_k), zero below the refractory period."""
    m, w = parse(row.memory_s), parse(row.excitation_per_scale)
    phi = np.sum(w[:, None] / m[:, None] * np.exp(-t[None, :] / m[:, None]), axis=0)
    phi[t < float(row.delta)] = 0.0
    return phi


def save(fig, out, name):
    for ext in ("pdf", "eps", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def fig_priceunits(fine, out, syms=("INTC", "BAC", "MDLZ", "TGT")):
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.3), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), syms):
        for d, col, name in (("E0", ORANGE, "raw"), ("E1m", BLUE, "clean")):
            g = fine[(fine.sym == s) & (fine.definition == d)].sort_values("H")
            ax.fill_between(g.H, g.data - 2 * g.se_day, g.data + 2 * g.se_day, color=col, alpha=0.2, lw=0)
            ax.plot(g.H, g.data, color=col, lw=1.4, label=f"{name}: observed")
            ax.plot(g.H, g.model, color=col, lw=1.4, ls=(0, (4, 2)), label=f"{name}: model")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.1, 1, 10])
        ax.set_xticklabels(["10 ms", "0.1 s", "1 s", "10 s"], fontsize=8)
        ax.minorticks_off()
        ax.set_title(s, fontsize=9, loc="left")
        ax.grid(axis="y", lw=0.4, color="#ececec", zorder=0)
    for ax in axes[1]:
        ax.set_xlabel("Horizon $H$ (log scale)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mid change after an up move (ticks)", fontsize=8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=7.5, loc="lower center", ncol=4, handlelength=2.2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, out, "fig_priceunits")


def fig_kernel(e0, e1m, out, month="201705", syms=("INTC", "TGT")):
    t = np.logspace(-5, 2.3, 2000)
    fig, axes = plt.subplots(1, len(syms), figsize=(6.4, 2.9), sharey=True)
    for ax, sym in zip(axes, syms):
        r = e0[(e0.sym == sym) & (e0.month == month)].iloc[0]
        c = e1m[(e1m.sym == sym) & (e1m.month == month)].iloc[0]
        for row, col, label in ((r, ORANGE, "raw events"), (c, BLUE, "clean events")):
            y = t * kernel(row, t)
            ax.fill_between(t, 0, y, color=col, alpha=0.25, lw=0)
            ax.plot(t, y, color=col, lw=1.4, label=label)
        ax.axvline(1e-3, color=GREY, lw=0.6, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 200)
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        ax.set_xticklabels(["0.1 ms", "1 ms", "10 ms", "0.1 s", "1 s", "10 s", "100 s"], fontsize=7.5)
        ax.minorticks_off()
        ax.set_title(sym, fontsize=9, loc="left")
        ax.set_xlabel("Time since the event, $t$ (log scale)", fontsize=8.5)
        ax.grid(axis="y", lw=0.4, color="#e5e5e5", zorder=0)
    axes[0].set_ylabel("$t\\,\\varphi(t)$  (kernel mass per unit of $\\log t$)", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    save(fig, out, "fig_kernel")


def fig_months(pan, out):
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), STOCKS):
        g = pan[pan.sym == s]
        col = ORANGE if s in ("TGT", "ATVI") else BLUE
        ax.scatter(pan.events_per_day, pan.n1, s=6, color="#d9d9d9", lw=0, zorder=1)
        ax.scatter(g.events_per_day, g.n1, s=16, color=col, lw=0, zorder=3)
        b, a = np.polyfit(g.logN, np.log(g.n1), 1)
        xs = np.array([g.logN.min(), g.logN.max()])
        ax.plot(np.exp(xs), np.exp(a + b * xs), color=col, lw=1.0, zorder=2)
        ax.set_title(f"{s}   slope {b:.2f}", fontsize=8.5, loc="left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(380, 11000)
        ax.set_ylim(0.05, 0.6)
        ax.set_xticks([500, 2000, 8000])
        ax.set_xticklabels(["500", "2,000", "8,000"], fontsize=7.5)
        ax.set_yticks([0.05, 0.1, 0.2, 0.4])
        ax.set_yticklabels(["0.05", "0.1", "0.2", "0.4"], fontsize=7.5)
        ax.minorticks_off()
        ax.grid(lw=0.4, color="#ececec", zorder=0)
        print(f"{s}: slope {b:.3f}, {len(g)} months")
    fig.supxlabel("Clean events per day", fontsize=9, y=0.02)
    fig.supylabel("$n(1\\,\\mathrm{s})$", fontsize=9, x=0.015)
    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    save(fig, out, "fig_months")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fits-e1m", required=True, help="main-panel clean fits (E1m, K = 4, refractory period)")
    p.add_argument("--fits-e0", required=True, help="raw panel fits (E0, K = 4)")
    p.add_argument("--panel-sd", required=True, help="results/blockboot_panel_sd.csv (n(1 s) of the main panel, BAC at 2 ms)")
    p.add_argument("--by-day", required=True, help="results/by_day_fixed_timescales.csv (clean events per day)")
    p.add_argument("--priceunits-fine", default=None, help="results/a8_priceunits_fine.csv (price-units figure)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.priceunits_fine:
        fig_priceunits(pd.read_csv(a.priceunits_fine), out)
    e1m = pd.read_csv(a.fits_e1m, dtype={"month": str})
    e1m = e1m[(e1m.definition == "E1m") & (e1m.K == 4)]
    e0 = pd.read_csv(a.fits_e0, dtype={"month": str})
    e0 = e0[(e0.definition == "E0") & (e0.K == 4)]
    fig_kernel(e0, e1m, out)
    pan = pd.read_csv(a.panel_sd, dtype={"month": str})
    day = pd.read_csv(a.by_day, dtype={"month": str})
    ev = day.groupby(["sym", "month"]).n_events.mean().rename("events_per_day").reset_index()
    pan = pan.merge(ev, on=["sym", "month"])
    assert len(pan) == 112, len(pan)
    pan["logN"] = np.log(pan.events_per_day)
    fig_months(pan, out)


if __name__ == "__main__":
    main()

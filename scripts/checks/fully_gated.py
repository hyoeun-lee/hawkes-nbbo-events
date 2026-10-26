"""Appendix B table: kernel summaries against the fully gated quantities.

For each event at t_m the kernel is switched off inside the dead windows [t_n, t_n + delta) of all later
events, so the realized excitation of the event is below the kernel summary n(infinity) =
sum_k w_k exp(-beta_k delta), w_k = a_k + c_k. Since the clean series has no two events closer than
delta, the later windows are disjoint and start at or after t_m + delta, and the mass lost by
component k is w_k (1 - exp(-beta_k delta)) sum_{n > m} exp(-beta_k (t_n - t_m)), a backward decayed
count. The script averages the loss over the events of the month and prints, for each stock, the loss
of the fastest component as a fraction of its effective weight, the effective branching ratio and
its fully gated counterpart.

    python scripts/checks/fully_gated.py --nbbo DATA/nbbo_2017May --fits DATA/fits_e1m_delta_K4.csv \\
        --fits-bac DATA/fits_e1m_h2_K3.csv --month 201705 --syms INTC BAC MDLZ TGT
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fit_panel as fp  # noqa: E402


def parse(s):
    return np.array([float(x) for x in str(s).split(";")])


def gated_loss(days, mem, w, delta):
    """Effective weights and the mass each component loses to later dead windows, per event."""
    beta = 1.0 / mem
    eff = w * np.exp(-beta * delta)
    loss = np.zeros(len(mem))
    n = 0
    for d in days:
        t = d.t
        for k, b in enumerate(beta):
            acc = 0.0
            total = 0.0
            for m in range(len(t) - 2, -1, -1):
                acc = np.exp(-b * (t[m + 1] - t[m])) * (acc + 1.0)
                total += acc
            loss[k] += w[k] * (1.0 - np.exp(-b * delta)) * total
        n += len(t)
    return eff, loss / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nbbo", nargs="+", required=True)
    p.add_argument("--fits", required=True, help="E1m fits, K = 4, refractory period 1 ms")
    p.add_argument("--fits-bac", default=None, help="E1m fits for BAC, K = 3, 2 ms (its main specification)")
    p.add_argument("--month", default="201705")
    p.add_argument("--syms", nargs="+", default=["INTC", "BAC", "MDLZ", "TGT"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    fits = pd.read_csv(a.fits, dtype={"month": str})
    bac = pd.read_csv(a.fits_bac, dtype={"month": str}) if a.fits_bac else None
    rows = []
    for sym in a.syms:
        src = bac if (sym == "BAC" and bac is not None) else fits
        r = src[(src.sym == sym) & (src.month == a.month) & (src.definition == "E1m")].iloc[0]
        h = float(r.delta)
        days = fp.build_days(fp.month_files(a.nbbo, sym, a.month), "E1m", h)
        eff, loss = gated_loss(days, parse(r.memory_s), parse(r.excitation_per_scale), h)
        row = dict(sym=sym, K=int(r.K), delta=h, fastest_loss_pct=100 * loss[0] / eff[0], eff_br=eff.sum(),
                   fully_gated=eff.sum() - loss.sum(), loss_total=loss.sum())
        rows.append(row)
        print(f"{sym} K={row['K']} delta={h}: fastest loss {row['fastest_loss_pct']:.2f}%  eff BR {row['eff_br']:.4f}  "
              f"fully gated {row['fully_gated']:.4f}", flush=True)
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

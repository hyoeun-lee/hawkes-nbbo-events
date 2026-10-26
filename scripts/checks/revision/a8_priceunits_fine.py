"""A8 on a fine horizon grid, for the price-units figure of the paper.

Same functional as a8_priceunits_k4.py (observed signed mid change after an up move, day-clustered
standard error, and the same statistic on R = 5 months simulated from the fitted model), evaluated on
30 log-spaced horizons from 10 ms to 10 s. The values at 0.1, 1 and 10 s agree with the table of
a8_priceunits_k4.py: the data exactly, the model within Monte Carlo noise of the five simulated months.

    python scripts/checks/revision/a8_priceunits_fine.py --out results/a8_priceunits_fine.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a8_priceunits_k4 as a8  # noqa: E402
from a8_priceunits_k4 import S, fp, reconstruct  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-horizons", type=int, default=30)
    a = p.parse_args()
    HS = tuple(float(x) for x in np.logspace(-2, 1, a.n_horizons))
    a8.HS = HS
    rng = np.random.default_rng(7)
    k4 = pd.read_csv(f"{S}/delta/K4.csv", dtype={"month": str})
    k3 = pd.read_csv(f"{S}/delta/K3.csv", dtype={"month": str})
    rk4 = pd.read_csv(f"{S}/cluster4/fits_e0_k4_panel.csv", dtype={"month": str})
    rows = []
    for sym in ["INTC", "BAC", "MDLZ", "TGT"]:
        r1 = (k3 if sym == "BAC" else k4)
        r1 = r1[(r1.sym == sym) & (r1.month == "201705") & (r1.definition == "E1m")].iloc[0]
        r0 = rk4[(rk4.sym == sym) & (rk4.month == "201705")].iloc[0]
        for definition, row in (("E0", r0), ("E1m", r1)):
            t0 = time.time()
            days = fp.build_days(fp.month_files([f"{S}/may/NBBO"], sym, "201705"), definition, 0.001)
            fit = reconstruct(days, row)
            size, emp = a8.empirical(sym, definition)
            sim = a8.on_paths(days, fit, size, rng)
            for H in HS:
                m, se_n, se_d = emp[H]
                rows.append(dict(sym=sym, definition=definition, H=H, data=m, se_naive=se_n, se_day=se_d, model=sim[H]))
            print(sym, definition, f"{(time.time() - t0) / 60:.1f} min", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

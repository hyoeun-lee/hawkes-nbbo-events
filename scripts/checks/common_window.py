"""Revision item A5: raw and clean kernel mass on common windows, and the horizon-wise identity.

Clean: the main E1m fits (K = 4; BAC from the h = 2 ms, K = 3 panel). Raw: a panel of E0 fits.
For each stock-month: raw mass from 0, from h and from 5 ms; clean mass from h and from 5 ms;
and the ratio of the raw mass on (h, H] to the dilution prediction (N_m / N_0) x clean n(H).

    python scripts/checks/common_window.py --clean DATA/fits_e1m_delta_K4.csv \\
        --clean-bac DATA/fits_e1m_h2_K3.csv --raw results/fits_e0_K3_panel.csv --out a5.csv
"""
import argparse
import numpy as np
import pandas as pd

ORDER = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]


def vec(s):
    return np.array([float(x) for x in str(s).split(";")])


def mass(r, lo, H):
    m, e = vec(r.memory_s), vec(r.excitation_per_scale)
    return float(np.sum(e * (np.exp(-lo / m) - np.exp(-H / m))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--clean-bac", required=True, help="E1m fits at h = 2 ms (K = 3) for BAC")
    p.add_argument("--raw", required=True)
    p.add_argument("--K-clean", type=int, default=4)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    k4 = pd.read_csv(a.clean, dtype={"month": str}); k4 = k4[(k4.definition == "E1m") & (k4.K == a.K_clean)]
    h2 = pd.read_csv(a.clean_bac, dtype={"month": str}); h2 = h2[(h2.definition == "E1m") & (h2.K == 3)]
    clean = pd.concat([k4[k4.sym != "BAC"], h2[h2.sym == "BAC"]]).set_index(["sym", "month"])
    raw = pd.read_csv(a.raw, dtype={"month": str}); raw = raw[raw.definition == "E0"].set_index(["sym", "month"])
    rows = []
    for key, c in clean.iterrows():
        if key not in raw.index:
            continue
        r = raw.loc[key]; h = 0.002 if key[0] == "BAC" else 0.001
        d = dict(sym=key[0], month=key[1], Nm=c.n_events, N0=r.n_events, ratioN=c.n_events / r.n_events, h=h,
                 clean_h_1=mass(c, h, 1), clean_h_10=mass(c, h, 10), clean_5_1=mass(c, 0.005, 1), clean_5_10=mass(c, 0.005, 10),
                 raw_0_1=mass(r, 0, 1), raw_0_10=mass(r, 0, 10), raw_sub_h=mass(r, 0, h),
                 raw_h_1=mass(r, h, 1), raw_h_10=mass(r, h, 10), raw_5_1=mass(r, 0.005, 1), raw_5_10=mass(r, 0.005, 10))
        rows.append(d)
    t = pd.DataFrame(rows)
    t["ratio_0_1"] = t.raw_0_1 / t.clean_h_1; t["ratio_h_1"] = t.raw_h_1 / t.clean_h_1; t["ratio_5_1"] = t.raw_5_1 / t.clean_5_1
    t["ratio_0_10"] = t.raw_0_10 / t.clean_h_10; t["ratio_h_10"] = t.raw_h_10 / t.clean_h_10; t["ratio_5_10"] = t.raw_5_10 / t.clean_5_10
    t["identity_h_1"] = t.raw_h_1 / (t.ratioN * t.clean_h_1); t["identity_h_10"] = t.raw_h_10 / (t.ratioN * t.clean_h_10)
    t["identity_5_1"] = t.raw_5_1 / (t.ratioN * t.clean_5_1); t["identity_5_10"] = t.raw_5_10 / (t.ratioN * t.clean_5_10)

    def q(col):
        return f"{t[col].median():.2f} [{t[col].quantile(.25):.2f}-{t[col].quantile(.75):.2f}] ({t[col].min():.2f}-{t[col].max():.2f})"
    print("raw:", a.raw, "| stock-months:", len(t))
    print("\nRaw / clean, median [IQR] (range):")
    for name, col in [("n(1 s): raw from 0, clean from h", "ratio_0_1"), ("        both from h", "ratio_h_1"), ("        both from 5 ms", "ratio_5_1"),
                      ("n(10 s): raw from 0, clean from h", "ratio_0_10"), ("         both from h", "ratio_h_10"), ("         both from 5 ms", "ratio_5_10")]:
        print(f"  {name:36s} {q(col)}")
    print("\nRaw mass on the window divided by (N_m/N_0) x clean mass on the same window (1 = exact dilution):")
    for name, col in [("(h, 1 s]", "identity_h_1"), ("(h, 10 s]", "identity_h_10"), ("(5 ms, 1 s]", "identity_5_1"), ("(5 ms, 10 s]", "identity_5_10")]:
        print(f"  {name:14s} {q(col)}")
    g = t.groupby("sym").median(numeric_only=True).reindex(ORDER)
    print("\nBy stock (medians):")
    print(g[["clean_h_1", "raw_0_1", "raw_h_1", "raw_5_1", "clean_5_1", "ratioN", "identity_h_1", "identity_5_1"]].round(3).to_string())
    if a.out:
        t.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

"""A11: the accounting identity under the Lee (2023) specification (IBM and AAPL, January 2017):
K = 2 or 3, constant daily baseline, one day at a time ('lee') and pooled over the month ('ours').
Reads the replication fits (ibm_fits.csv) and the monthly share of raw gaps under 1 ms (ibm_stats.csv)."""
import pandas as pd

S = os.environ.get("HAWKES_DATA", "DATA")   # root of the NBBO, trade and intermediate files (see REPLICATION.md)


def mass_beyond(r, suf, lo_ms=2.0):
    """Mass of the components with memory above lo_ms (the raw fast component is 0.3-1.1 ms)."""
    return sum(r[f"exc{k}_{suf}"] for k in (1, 2, 3) if pd.notna(r.get(f"mem{k}_ms_{suf}")) and r[f"mem{k}_ms_{suf}"] > lo_ms)


def main():
    d = pd.read_csv(f"{S}/ibm/ibm_fits.csv"); st = pd.read_csv(f"{S}/ibm/ibm_stats.csv").set_index("sym")
    cols = ["n_events", "branching", "mem1_ms", "exc1", "mem2_ms", "exc2", "mem3_ms", "exc3"]
    for sym in ["IBM", "AAPL"]:
        s = st.loc[sym, "gap_lt1ms_E0"] / 100
        for K in (2, 3):
            for mode in ("lee", "ours"):
                sel = lambda definition: d[(d.sym == sym) & (d.definition == definition) & (d.K == K) & (d["mode"] == mode)].set_index("day")[cols]
                m = sel("E0").join(sel("E1m"), lsuffix="_0", rsuffix="_m")
                m["ratioN"] = m.n_events_m / m.n_events_0; m["pred"] = s + m.branching_m * m.ratioN; m["err"] = m.branching_0 - m.pred
                m["raw_beyond"] = [mass_beyond(r, "0") for _, r in m.iterrows()]; m["dilution"] = m.branching_m * m.ratioN
                q = m.median(numeric_only=True)
                print(f"{sym} K={K} {mode:5s} ({len(m)} fits): N_m/N_0 {q.ratioN:.3f}, s {s:.3f}, raw BR {q.branching_0:.3f}, clean BR {q.branching_m:.3f}, "
                      f"predicted {q.pred:.3f}, error {q.err:+.3f} [{m.err.min():+.2f}, {m.err.max():+.2f}]; raw mass beyond 2 ms {q.raw_beyond:.3f} vs diluted clean {q.dilution:.3f}")


if __name__ == "__main__":
    main()

import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/scripts/checks')
from hawkes_taq import events as ev
from hawkes_taq.hawkes import SESSION_END
from trades_match import classify, matched, TICK
S = sys.argv[1] if len(sys.argv) > 1 else "."  # folder with may/NBBO/<SYM>/ and trades_in/trades/<SYM>/
def day(nbbo, trades):
    st = ev.build_states(nbbo, SESSION_END); side, gone = classify(st)
    bid, ask, mid2, dur, sp = (st[c].to_numpy() for c in ("bid", "ask", "mid2", "dur", "spread"))
    brief = (dur < 1e-3) & (sp > 0) & (side != "") & (side != "both")
    idx = np.flatnonzero(brief); idx = idx[idx < len(st) - 1]
    # resolution: next state restores the old price on the moved side (refill) / the other side follows (two-step move) / other
    nb, na = bid[idx + 1], ask[idx + 1]
    refill = np.where(side[idx] == "ask", na == gone[idx], nb == gone[idx])
    follow = np.where(side[idx] == "ask", (nb > bid[idx]) & ~refill, (na < ask[idx]) & ~refill)
    other = ~refill & ~follow
    tr_t = trades.t.to_numpy(); tr_p = np.rint(trades.price.to_numpy() / TICK).astype(np.int64)
    roundlot = ~trades.cond.fillna("").str.contains("I").to_numpy() & (trades["size"].to_numpy() >= 100)
    m_all = matched(st.t.to_numpy()[idx], gone[idx], tr_t, tr_p, -0.003, 0.0005)
    m_rl = matched(st.t.to_numpy()[idx], gone[idx], tr_t[roundlot], tr_p[roundlot], -0.003, 0.0005)
    return dict(n=len(idx), refill=refill.sum(), follow=follow.sum(), other=other.sum(), m_all=m_all.sum(), m_rl=m_rl.sum(),
                m_refill=m_all[refill].sum(), m_follow=m_all[follow].sum(), n_rl_trades=int(roundlot.sum()), n_trades=len(tr_t))
for sym in ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]:
    tot = {}
    for f in sorted(Path(S, "may/NBBO", sym).glob("*.csv.gz")):
        g = Path(S, "trades_in/trades", sym, f.name)
        if g.exists():
            for k, v in day(pd.read_csv(f), pd.read_csv(g)).items(): tot[k] = tot.get(k, 0) + int(v)
    n = tot["n"]
    print(f"{sym:5s} brief pos-spread n={n:6d} | resolution: refill {tot['refill']/n:.2f} follow {tot['follow']/n:.2f} other {tot['other']/n:.2f}"
          f" | matched all trades {tot['m_all']/n:.3f}, round lots only {tot['m_rl']/n:.3f} (round-lot share of trades {tot['n_rl_trades']/tot['n_trades']:.2f})"
          f" | matched within refill {tot['m_refill']/max(tot['refill'],1):.3f}, within follow {tot['m_follow']/max(tot['follow'],1):.3f}", flush=True)

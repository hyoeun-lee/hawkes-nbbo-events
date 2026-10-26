"""A10 (exact immigrant accounting) and A9 (discriminant: follower counts and sizes), May 2017, from the states."""
import os
import sys
from pathlib import Path, glob, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from hawkes_taq import events as ev
from hawkes_taq.hawkes import SESSION_END
S = os.environ.get("HAWKES_DATA", "DATA")   # root of the NBBO, trade and intermediate files (see REPLICATION.md)
ORDER = ["INTC", "MU", "MDLZ", "BAC", "WFC", "T", "TGT", "ATVI"]
H, HL, D = 0.001, 0.005, 0.001
rows = []
for sym in ORDER:
    acc = dict(N0=0, Nm=0, follower=0, removed=0, rm_nf=0, kept_f=0, rt=0, longbrief=0, longbrief_nonpos=0, other=0,
               f0=0, f1=0, f2=0, f3=0, size2_all=0, size2_f=0, cont_f=0, n_f_size=0)
    for f in sorted(glob.glob(f"{S}/may/NBBO/{sym}/*.csv.gz")):
        st = ev.build_states(pd.read_csv(f), SESSION_END)
        t, sp, dur, mid = (st[c].to_numpy() for c in ("t", "spread", "dur", "mid2"))
        n = len(st)
        transient = ((sp <= 0) & (dur < HL)) | (dur == 0) | (dur < H)
        kept = ~transient
        # raw events: states i >= 1 where the mid changed
        raw_idx = np.flatnonzero(np.diff(mid) != 0) + 1
        tr = t[raw_idx]; dm = mid[raw_idx] - mid[raw_idx - 1]
        gap_prev = np.diff(tr, prepend=-np.inf)
        follower = gap_prev < H
        # survival: a kept state whose mid differs from the previous kept state's mid carries a clean event
        kidx = np.flatnonzero(kept); prev_kept_mid = np.full(n, np.nan)
        prev_kept_mid[kidx[1:]] = mid[kidx[:-1]]
        survives = kept[raw_idx] & (mid[raw_idx] != prev_kept_mid[raw_idx])
        removed = ~survives
        acc["N0"] += len(raw_idx); acc["Nm"] += int(survives.sum()); acc["follower"] += int(follower.sum()); acc["removed"] += int(removed.sum())
        rm_nf = removed & ~follower; kept_f = survives & follower
        acc["rm_nf"] += int(rm_nf.sum()); acc["kept_f"] += int(kept_f.sum())
        # categories of removed non-followers: entry into a brief state that (a) is a round trip (mid returns before the next survivor),
        # (b) lasts longer than H before the next raw event (long lock/cross merged by h_L), (c) other
        nxt_gap = np.diff(tr, append=np.inf)
        state_dur = dur[raw_idx]
        nsurv = np.cumsum(survives)                       # survivors up to and including i
        next_surv_mid_before = np.full(len(raw_idx), np.nan)  # mid just before the next survivor after i
        surv_pos = np.flatnonzero(survives)
        for i in np.flatnonzero(rm_nf):
            if nxt_gap[i] >= H:
                acc["longbrief"] += 1
                acc["longbrief_nonpos"] += int(sp[raw_idx[i]] <= 0)
            else:
                k = np.searchsorted(surv_pos, i, side="right")
                if k < len(surv_pos):
                    j = surv_pos[k]
                    if mid[raw_idx[j] - 1] == mid[raw_idx[i] - 1]: acc["rt"] += 1
                    else: acc["other"] += 1
                else: acc["other"] += 1
        # A9: follower counts within H after each raw event
        cnt = np.searchsorted(tr, tr + H, side="right") - np.arange(len(tr)) - 1
        acc["f0"] += int((cnt == 0).sum()); acc["f1"] += int((cnt == 1).sum()); acc["f2"] += int((cnt == 2).sum()); acc["f3"] += int((cnt >= 3).sum())
        size2 = np.abs(dm) >= 2
        acc["size2_all"] += int(size2.sum()); acc["size2_f"] += int((size2 & follower).sum()); acc["n_f_size"] += int(follower.sum())
        same = np.sign(dm[1:]) == np.sign(dm[:-1]); acc["cont_f"] += int((same & follower[1:]).sum())
    a = acc; N0 = a["N0"]
    s = a["follower"] / N0
    pred_excess = a["removed"] - a["follower"]
    P1 = 1 - a["f0"] / N0; lam = -np.log(1 - P1)
    pois2 = 1 - np.exp(-lam) * (1 + lam)
    rows.append(dict(sym=sym, N0=N0, Nm=a["Nm"], s=round(s, 3), removed=a["removed"], followers=a["follower"],
                     pred_excess=pred_excess, pred_excess_per_N0=round(pred_excess / N0, 4),
                     rm_nonfollower=a["rm_nf"], kept_follower=a["kept_f"], longbrief=a["longbrief"], longbrief_nonpos=a["longbrief_nonpos"], roundtrip=a["rt"], entry_of_pair=a["other"],
                     P_ge1=round(P1, 3), P_ge2=round((a["f2"] + a["f3"]) / N0, 4), P_ge2_poisson=round(pois2, 4), P_ge3=round(a["f3"] / N0, 4),
                     share_size2_all=round(a["size2_all"] / N0, 3), share_size2_followers=round(a["size2_f"] / max(a["n_f_size"], 1), 3),
                     share_cont_followers=round(a["cont_f"] / max(a["n_f_size"], 1), 3)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"{S}/a5/a9a10_may.csv", index=False); print("ALLDONE")

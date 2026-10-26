"""Simulation study driver (see hawkes_taq/simstudy.py for the design).

Two steps, both run where the saved NBBO files are (the campus cluster):

1. truth: for each stock, fit E1m (K = 4, 1 ms dead time, full start grid) to one month and
   calibrate its brief states. Writes truth_<SYM>.npz and chains_<SYM>.csv.gz to --simdir.

       python scripts/simulate_study.py truth --nbbo DATA/nbbo_2017Q2 --month 201705 \\
           --syms INTC MDLZ BAC TGT --simdir DATA/sim

2. run: simulate and fit replicates. Rows are appended to --out as each replicate finishes,
   so a run can be resumed; with --task the replicate is chosen from a job-array index.

       python scripts/simulate_study.py run --simdir DATA/sim --sym INTC --experiment base \\
           --reps 0 1 2 --out DATA/sim/results_INTC_base.csv

       python scripts/simulate_study.py run --simdir DATA/sim --task $SLURM_ARRAY_TASK_ID \\
           --plan DATA/sim/plan.csv --out-dir DATA/sim/results

   `plan` writes the task table (one row per stock, experiment and replicate):

       python scripts/simulate_study.py plan --syms INTC MDLZ BAC TGT --reps-base 50 \\
           --reps-other 25 --out DATA/sim/plan.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hawkes_taq import hawkes as hk  # noqa: E402
from hawkes_taq import simstudy as ss  # noqa: E402


def cmd_truth(args):
    from calibrate_brief_states import stock_month
    from fit_panel import build_days, month_files
    out = Path(args.simdir)
    out.mkdir(parents=True, exist_ok=True)
    for sym in args.syms:
        files = month_files(args.nbbo, sym, args.month)
        chains, extra = stock_month(files)
        chains.to_csv(out / f"chains_{sym}.csv.gz", index=False)
        days = build_days(files, "E1m", 0.001)
        f = hk.fit(days, K=4, starts=hk.full_starts(4), delta=ss.DELTA)
        ss.Truth(sym, f.beta, f.alpha, f.mu, f.edges).save(out / f"truth_{sym}.npz")
        print(f"{sym}: {len(days)} days, {f.n_events} events, memory {f.timescales().round(4)}, "
              f"excitation {f.alpha.sum(axis=0).round(4)}, branching ratio {f.branching_ratio:.3f}",
              flush=True)


def cmd_plan(args):
    rows = []
    exps = ss.BATCH2 if args.batch == 2 else ss.BATCH1
    for sym in args.syms:
        for exp in exps:
            if args.batch == 2:
                n = args.reps_act if exp == "actscale" else args.reps_other
            else:
                n = args.reps_base if exp == "base" else args.reps_other
            rows += [{"sym": sym, "experiment": exp, "rep": r} for r in range(n)]
    pd.DataFrame(rows).to_csv(args.out, index_label="task")
    print(f"{len(rows)} tasks")


def cmd_run(args):
    if args.task is not None:
        plan = pd.read_csv(args.plan, index_col="task")
        jobs = [tuple(plan.loc[args.task, ["sym", "experiment", "rep"]])]
        out = Path(args.out_dir) / f"task_{args.task:05d}.csv"
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    else:
        jobs = [(args.sym, args.experiment, r) for r in args.reps]
        out = Path(args.out)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev["sym"], prev["experiment"], prev["rep"]))
    for sym, exp, rep in jobs:
        if (sym, exp, int(rep)) in done:
            continue
        truth, model = ss.load_inputs(args.simdir, sym)
        rows = ss.run_replicate(truth, model, exp, int(rep), ss.seed_for(sym, exp, int(rep)),
                                fits=args.fits, log=lambda m: print(m, flush=True))
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--nbbo", nargs="+", required=True)
    t.add_argument("--month", default="201705")
    t.add_argument("--syms", nargs="+", default=["INTC", "MDLZ", "BAC", "TGT"])
    t.add_argument("--simdir", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--syms", nargs="+", default=["INTC", "MDLZ", "BAC", "TGT"])
    pl.add_argument("--reps-base", type=int, default=50)
    pl.add_argument("--reps-other", type=int, default=25)
    pl.add_argument("--batch", type=int, choices=[1, 2], default=1,
                    help="1: base, share0, share50, long, subms; 2: subms_clean, powerlaw, actscale")
    pl.add_argument("--reps-act", type=int, default=50,
                    help="batch 2: replicates of actscale (cycled over the five baseline scales)")
    pl.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--simdir", required=True)
    r.add_argument("--sym")
    r.add_argument("--experiment", choices=list(ss.EXPERIMENTS), default="base")
    r.add_argument("--reps", nargs="+", type=int, default=[0])
    r.add_argument("--out")
    r.add_argument("--task", type=int)
    r.add_argument("--plan")
    r.add_argument("--out-dir")
    r.add_argument("--fits", choices=["full", "core", "truth"], default="full",
                   help="core: only the four main fits per replicate; truth: the four main fits "
                        "plus a fit of the simulated clean process itself, ungated (definition TRUE)")
    args = p.parse_args()
    {"truth": cmd_truth, "plan": cmd_plan, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()

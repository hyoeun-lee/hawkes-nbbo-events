"""Build E0/E1/E2 event series from WRDS TAQ for the sample stocks.

Example (on WRDS, from the repository root):
    python scripts/build_events.py --start 2017-05-01 --end 2017-05-31
    python scripts/build_events.py --syms INTC --start 2017-05-17 --end 2017-05-17
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hawkes_taq.pipeline import SAMPLE_STOCKS, run, weekdays  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--syms", nargs="+", default=SAMPLE_STOCKS)
    p.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    p.add_argument("--out", default="data/events")
    p.add_argument("--max-nonpos", type=float, default=0.005,
                   help="locks/crosses shorter than this many seconds are merged in E1")
    p.add_argument("--stamp", choices=["exit", "entry"], default="exit")
    p.add_argument("--min-state", type=float, default=0.0,
                   help="in E1, also merge any state shorter than this many seconds")
    p.add_argument("--wrds-username", default=None)
    args = p.parse_args()

    import wrds
    db = wrds.Connection(wrds_username=args.wrds_username)
    qc = run(db, args.syms, weekdays(args.start, args.end), args.out,
             max_nonpos=args.max_nonpos, stamp=args.stamp, min_state=args.min_state)
    print(qc.to_string())


if __name__ == "__main__":
    main()

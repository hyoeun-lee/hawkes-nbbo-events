"""Export the replication package: copy the public subset of this repository into a target directory
(the working tree of the public repository), deleting what is there first.

Included: the package, tests, scripts, sql, results, REPLICATION.md (as README.md), requirements.txt,
CITATION.cff, LICENSE and .zenodo.json. Excluded: docs/ (manuscript, notes, reviews), the development
README, caches and anything not on the list.

    python scripts/export_replication.py /path/to/public-repo [--dry-run]
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["hawkes_taq", "tests", "scripts", "sql", "results", "requirements.txt", "CITATION.cff", "LICENSE", ".zenodo.json"]
RENAME = {"REPLICATION.md": "README.md"}
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".log"}
KEEP_IN_TARGET = {".git", "LICENSE", "LICENSE.md", "CITATION.cff", ".zenodo.json"}


def copy_tree(src, dst, dry):
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in EXCLUDE_DIRS for part in rel.parts) or p.suffix in EXCLUDE_SUFFIXES:
            continue
        if p.is_dir():
            continue
        out = dst / rel
        if dry:
            print(" ", out.relative_to(dst.parent))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    target = Path(a.target).resolve()
    if target == ROOT or ROOT in target.parents:
        sys.exit("target must be outside the development repository")
    target.mkdir(parents=True, exist_ok=True)
    if not a.dry_run:
        for child in target.iterdir():
            if child.name in KEEP_IN_TARGET:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for name in INCLUDE:
        src = ROOT / name
        if src.is_dir():
            copy_tree(src, target / name, a.dry_run)
        else:
            print(" ", name) if a.dry_run else shutil.copy2(src, target / name)
    for src_name, dst_name in RENAME.items():
        print(" ", dst_name) if a.dry_run else shutil.copy2(ROOT / src_name, target / dst_name)
    n = sum(1 for p in target.rglob("*") if p.is_file() and ".git" not in p.parts) if not a.dry_run else None
    print(f"exported to {target}" + (f": {n} files" if n is not None else " (dry run)"))


if __name__ == "__main__":
    main()

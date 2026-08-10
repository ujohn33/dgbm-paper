#!/usr/bin/env python3
"""Fetch the UCI benchmark datasets and verify them against the published copies.

    python reproduce/fetch_uci_data.py            # download what is missing, then verify
    python reproduce/fetch_uci_data.py --verify   # verify only, download nothing

Why this exists
---------------
The UCI loaders in ``uci/UCI_*_single_run_HP.py`` get their data three ways:

1. Four files ship inside the vendored ``ngboost`` submodule, so a
   ``--recurse-submodules`` clone already has them.
2. Five are read straight from ``archive.ics.uci.edu`` over the network at run
   time. This script caches them locally so a rerun does not depend on UCI
   staying reachable.
3. ``YearPredictionMSD.txt`` (448 MB) is gitignored inside that submodule, so a
   fresh clone does NOT have it and UCI dataset index 9 fails with
   ``FileNotFoundError`` until this script downloads it.

The SHA-256 values below are of the exact bytes behind the published results, so
a mismatch means the upstream copy changed after publication.

Uses only the standard library: no curl, wget, unzip or sha256sum required.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
import shutil
import sys
import urllib.request
import zipfile

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"

# (destination, url or None if vendored, sha256, member name inside a zip)
DATASETS = [
    ("ngboost/data/uci/kin8nm.csv", None,
     "7b9bf0301ac936d88122557a151e1ba8f1ebc278fcf46d9f3c6d462debdbc8ad", None),
    ("ngboost/data/uci/naval-propulsion.txt", None,
     "de0ea69da1efaab8b9655ffed828547d10dd68c1fb8c6e0163e6a988def393a6", None),
    ("ngboost/data/uci/power-plant.xlsx", None,
     "ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651", None),
    ("ngboost/data/uci/protein.csv", None,
     "4277cfcb4e91a181746cbc654f001b57951c9e6a80f4f795fdb5c807e0848f40", None),
    ("reproduce/uci_cache/housing.data", f"{UCI}/housing/housing.data",
     "baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a", None),
    ("reproduce/uci_cache/Concrete_Data.xls", f"{UCI}/concrete/compressive/Concrete_Data.xls",
     "710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4", None),
    ("reproduce/uci_cache/ENB2012_data.xlsx", f"{UCI}/00242/ENB2012_data.xlsx",
     "0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4", None),
    ("reproduce/uci_cache/winequality-red.csv", f"{UCI}/wine-quality/winequality-red.csv",
     "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e", None),
    ("reproduce/uci_cache/yacht_hydrodynamics.data", f"{UCI}/00243/yacht_hydrodynamics.data",
     "00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1", None),
    ("ngboost/data/uci/YearPredictionMSD.txt", f"{UCI}/00203/YearPredictionMSD.txt.zip",
     "4b6f8e50235b359e01689ae7fb33ad0f89677e9a15f25f3d6259327a6bb927bb",
     "YearPredictionMSD.txt"),
]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: pathlib.Path, member: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    if member is None:
        dest.write_bytes(payload)
        return
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        if member not in names:
            raise RuntimeError(f"{member!r} not in archive; found {names}")
        with zf.open(member) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="verify existing files only; download nothing")
    args = ap.parse_args()

    failed = 0
    for rel, url, want, member in DATASETS:
        path = REPO_ROOT / rel
        name = path.name

        if not path.exists() and not args.verify and url:
            print(f"downloading {name} ...", flush=True)
            try:
                download(url, path, member)
            except Exception as exc:                      # noqa: BLE001
                print(f"FAILED   {name:<28} {type(exc).__name__}: {exc}")
                failed += 1
                continue

        if not path.exists():
            hint = ("run: git submodule update --init ngboost" if url is None
                    else "run without --verify to download")
            print(f"MISSING  {name:<28} {hint}")
            failed += 1
            continue

        got = sha256(path)
        if got == want:
            print(f"ok       {name:<28} {got[:12]}...")
        else:
            print(f"MISMATCH {name:<28} got {got[:12]} want {want[:12]}")
            print(f"         the upstream copy changed after publication")
            failed += 1

    print()
    if failed:
        print(f"{failed} of {len(DATASETS)} datasets missing or changed -- see above.")
        return 1
    print(f"All {len(DATASETS)} UCI datasets present and byte-identical to the "
          "published runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

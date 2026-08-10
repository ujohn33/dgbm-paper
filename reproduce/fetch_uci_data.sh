#!/bin/bash
# Fetch the UCI benchmark datasets and verify them against the checksums of the
# files actually used for the published results.
#
#   bash reproduce/fetch_uci_data.sh          # download what is missing + verify
#   bash reproduce/fetch_uci_data.sh --verify # verify only, download nothing
#
# Why this exists
# ---------------
# The UCI loaders in uci/UCI_*_single_run_HP.py get their data three ways:
#
#   1. Four files come from the vendored `ngboost` submodule and are already in
#      a --recurse-submodules clone (kin8nm, naval-propulsion, power-plant,
#      protein).
#   2. Five are read straight from archive.ics.uci.edu over the network at run
#      time. They resolved when this script was written, but the URLs are
#      outside our control. This script caches them locally so a rerun does not
#      depend on UCI staying up or stable.
#   3. YearPredictionMSD.txt (448 MB) is gitignored inside the ngboost
#      submodule, so a fresh clone does NOT have it and UCI dataset index 9
#      fails with FileNotFoundError until this script downloads it.
#
# The checksums below are of the exact bytes behind the published numbers.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
DEST="ngboost/data/uci"
CACHE="reproduce/uci_cache"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1
mkdir -p "$DEST" "$CACHE"

# path|url|sha256   (url of "-" means it ships with the ngboost submodule)
ENTRIES=(
"$DEST/kin8nm.csv|-|7b9bf0301ac936d88122557a151e1ba8f1ebc278fcf46d9f3c6d462debdbc8ad"
"$DEST/naval-propulsion.txt|-|de0ea69da1efaab8b9655ffed828547d10dd68c1fb8c6e0163e6a988def393a6"
"$DEST/power-plant.xlsx|-|ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651"
"$DEST/protein.csv|-|4277cfcb4e91a181746cbc654f001b57951c9e6a80f4f795fdb5c807e0848f40"
"$CACHE/housing.data|https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data|baadf72995725d76efe787b664e1f083388c79ba21ef9a7990d87f774184735a"
"$CACHE/Concrete_Data.xls|https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls|710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4"
"$CACHE/ENB2012_data.xlsx|https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx|0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4"
"$CACHE/winequality-red.csv|https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv|4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e"
"$CACHE/yacht_hydrodynamics.data|http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data|00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1"
"$DEST/YearPredictionMSD.txt|https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip|4b6f8e50235b359e01689ae7fb33ad0f89677e9a15f25f3d6259327a6bb927bb"
)

fail=0
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r path url want <<< "$entry"
  name="$(basename "$path")"

  if [ ! -f "$path" ] && [ "$VERIFY_ONLY" -eq 0 ] && [ "$url" != "-" ]; then
    echo "downloading $name ..."
    if [[ "$url" == *.zip ]]; then
      tmp="$(mktemp -d)"
      curl -sL --max-time 3600 -o "$tmp/f.zip" "$url" \
        && unzip -qo "$tmp/f.zip" -d "$tmp" \
        && mv "$tmp"/*.txt "$path"
      rm -rf "$tmp"
    else
      curl -sL --max-time 600 -o "$path" "$url"
    fi
  fi

  if [ ! -f "$path" ]; then
    if [ "$url" = "-" ]; then
      echo "MISSING  $name -- run: git submodule update --init ngboost"
    else
      echo "MISSING  $name"
    fi
    fail=1
    continue
  fi

  got="$(sha256sum "$path" | cut -d' ' -f1)"
  if [ "$got" = "$want" ]; then
    printf 'ok       %-28s %s\n' "$name" "${got:0:12}..."
  else
    printf 'MISMATCH %-28s got %s want %s\n' "$name" "${got:0:12}" "${want:0:12}"
    echo "         the upstream copy has changed since the published runs"
    fail=1
  fi
done

echo
if [ "$fail" -eq 0 ]; then
  echo "All 10 UCI datasets present and byte-identical to the published runs."
else
  echo "Some datasets are missing or differ -- see above."
fi
exit "$fail"

# DGBM — Distributional Gradient Boosting Machines

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21875432.svg)](https://doi.org/10.5281/zenodo.21875432)

Code, experiment scripts and result artifacts for the paper **Distributional
Gradient Boosting Machines** by Alexander März, Evgenii Genov, Thomas Kneib and
Christoph Bergmeir.

DGBM formulates distributional regression as multi-parameter Newton boosting
with observed-information Hessians obtained by automatic differentiation,
implemented on top of the LightGBM and XGBoost histogram backends:

- **DGBM-LGB**, built on [LightGBMLSS](https://github.com/ujohn33/LightGBMLSS_fork)
- **DGBM-XGB**, built on [XGBoostLSS](https://github.com/ujohn33/XGBoostLSS)

Both are vendored as git submodules, pinned to the exact commits behind the
published numbers.

## Quick start

**1. Install.** DGBM lives in the submodules, so clone recursively.

```bash
git clone --recurse-submodules https://github.com/ujohn33/dgbm-paper
cd dgbm-paper
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r reproduce/requirements-dgbm.txt
```

**2. Get the data.** OpenML downloads itself at run time. UCI needs one command
— it also fetches `YearPredictionMSD.txt`, which is too large to ship, and
checksums all ten datasets against the copies used for the paper.

```bash
python reproduce/fetch_uci_data.py
```

**3a. Reproduce the OpenML results** (19 datasets, all six methods, one
command). Add `--dry-run` first to see what it would launch.

```bash
python reproduce/run_openml_benchmarks.py --tasks 0-18
```

**3b. Reproduce the UCI results** (10 datasets, indices 0–9). Each method takes
its own arguments, so there is no single runner:

```bash
for i in $(seq 0 9); do
  python uci/UCI_lssboost_single_run_HP.py   "$i" exp False None None False  # DGBM-LGB
  python uci/UCI_xglssboost_HP_single_run.py "$i" exp False None None False  # DGBM-XGB
  python uci/UCI_ngboost_single_run.py       "$i"                            # NGBoost
  python uci/UCI_pgbm_single_run.py          "$i"                            # PGBM
done
```

GPBoost and XLSF need their own environments — see
[`reproduce/README.md`](reproduce/README.md).

**4. Regenerate the paper's tables.**

```bash
python reproduce/make_figures.py --out reproduce/figures   # CD diagrams + runtime plots
python reproduce/export_final_hps.py --out reproduce/tables
python reproduce/aggregate_results.py --out reproduce/tables
```

> **[`reproduce/README.md`](reproduce/README.md) is the complete guide** — the
> four environments and why they are separate, the full experimental protocol
> (splits, seeds, TPE budgets, early stopping, CRPS sampling), running on a
> batch cluster, a paper element → script → result-file map, and the known
> limitations of this release. Start there for anything beyond the commands
> above.

## Where things are

| Path | Contents |
|---|---|
| [`reproduce/README.md`](reproduce/README.md) | **The full reproduction guide** |
| `reproduce/` | Pinned environments, data fetcher, unified OpenML runner, figure and table generators, per-job hardware record |
| `openml/`, `uci/` | One experiment entry point per dataset–method pair |
| `slurm/` | The batch job scripts used for the published runs |
| `results/` | Per-dataset result CSVs behind every table in the paper |
| `logs/openml/` | Final tuned hyperparameters per dataset (supplementary tables) |
| `synthetic/`, `notebooks/` | CUDA scaling benchmarks and the analysis behind the supplementary GPU figures |

## Reproducibility notes

- Seeds, splits, TPE budgets and early-stopping rules are documented in
  [`reproduce/README.md`](reproduce/README.md) and match the paper's
  Reproducibility section.
- `reproduce/openml_suite336_tasks.json` pins the OpenML task IDs, dataset IDs
  and dataset versions.
- `reproduce/fetch_uci_data.py` verifies all ten UCI datasets against the
  SHA-256 of the copies used for the published results.
- `reproduce/job_hardware.csv` records the partition and node of every batch
  array task, so reported runtimes can be attributed to concrete hardware.

GitHub's auto-generated release tarballs do **not** include submodule contents.
Clone with `--recurse-submodules`, or use the self-contained archive attached to
the [latest release](https://github.com/ujohn33/dgbm-paper/releases/latest).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper. The archived snapshot
is at [10.5281/zenodo.21875432](https://doi.org/10.5281/zenodo.21875432) — the
concept DOI, which always resolves to the latest version.

## License

[Apache-2.0](LICENSE), matching the vendored upstreams.

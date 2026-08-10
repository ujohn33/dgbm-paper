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

```bash
git clone --recurse-submodules https://github.com/ujohn33/dgbm-paper
cd dgbm-paper
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r reproduce/requirements-distiboost_311.txt
bash reproduce/fetch_uci_data.sh          # UCI data + checksum verification
```

Reproduce the OpenML benchmark from one entry point:

```bash
python reproduction/run_openml_benchmarks.py --list-profiles   # see what is available
python reproduction/run_openml_benchmarks.py --tasks 0-18 --dry-run
python reproduction/run_openml_benchmarks.py --tasks 0-18      # all methods, seed 123
```

Or run a single dataset-method pair directly:

```bash
# DGBM-LGB, OpenML task index 0, the configuration reported in the paper
python openml/openml_lssboost_HP_single_run.py 0 exp False None None False 123 False
```

## Where things are

| Path | Contents |
|---|---|
| [`reproduce/README.md`](reproduce/README.md) | **Start here.** Environments, the full experimental protocol, and a paper table/figure → script → result-file mapping |
| `reproduction/run_openml_benchmarks.py` | Unified OpenML runner — every method and ablation profile from one command |
| `openml/`, `uci/` | One experiment entry point per dataset–method pair |
| `slurm/` | The job scripts used to launch every run |
| `results/` | Per-dataset result CSVs behind every table in the paper |
| `logs/openml/` | Final tuned hyperparameters per dataset (supplementary tables) |
| `reproduce/` | Pinned environments, environment capture, per-job hardware record, table generators |

## Reproducibility notes

- Seeds, splits, TPE budgets and early-stopping rules are documented in
  [`reproduce/README.md`](reproduce/README.md) and match the paper's
  Reproducibility section.
- `reproduce/openml_suite336_tasks.json` pins the OpenML task IDs, dataset IDs
  and dataset versions.
- `reproduce/fetch_uci_data.sh` verifies all ten UCI datasets against the
  SHA-256 of the copies used for the published results.
- `reproduce/job_hardware.csv` records the partition and node of every Slurm
  task, so reported runtimes can be attributed to concrete hardware.

Note that GitHub's auto-generated release tarballs do **not** include submodule
contents. Use `--recurse-submodules` when cloning, or the self-contained
archive attached to the release.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the paper. The archived
snapshot is at [10.5281/zenodo.21875432](https://doi.org/10.5281/zenodo.21875432)
— that is the concept DOI, which always resolves to the latest version.

## License

[Apache-2.0](LICENSE), matching the vendored upstreams.

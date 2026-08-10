# DGBM — Distributional Gradient Boosting Machines

Code and experiment scripts for the paper *Distributional Gradient Boosting
Machines*. This repository reproduces every table and figure in the paper and
its supplementary material.

DGBM has two interchangeable backends:

- **DGBM-LGB**, built on [LightGBMLSS](https://github.com/ujohn33/LightGBMLSS_fork)
- **DGBM-XGB**, built on [XGBoostLSS](https://github.com/ujohn33/XGBoostLSS)

Both are vendored as git submodules and pinned to the exact commits used for
the published results.

---

## 1. Installation

```bash
git clone --recurse-submodules https://github.com/ujohn33/dgbm-paper LSSboost
cd LSSboost
```

The OpenML scripts read your OpenML API key from the environment; public suite
336 tasks download without one, but set it if you hit rate limits:

```bash
export OPENML_APIKEY=<your key>
```

Submodule commits used for the published results:

| Submodule    | Origin                                       | Commit    | Version         |
|--------------|----------------------------------------------|-----------|-----------------|
| LightGBMLSS  | `github.com/ujohn33/LightGBMLSS_fork`        | `67b7698` | v0.4.0-43       |
| XGBoostLSS   | `github.com/ujohn33/XGBoostLSS`              | `6e72532` | v0.4.0          |
| ngboost      | `github.com/ujohn33/ngboost_fork`            | `db4d19b` | 0.5.1.dev0      |
| pgbm         | `github.com/elephaint/pgbm`                  | —         | 2.3.0           |
| properscoring| `github.com/tozech/properscoring`            | —         | 0.1             |

### 1.1 On a plain machine

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r reproduce/requirements-distiboost_311.txt
```

`reproduce/requirements-*.txt` cover the four environments that were used. The
first three are `pip freeze` dumps of the live venvs; `requirements-gpboost.txt`
is hand-written, because that venv no longer exists (see below):

| File                                  | Used for                                        |
|---------------------------------------|-------------------------------------------------|
| `requirements-distiboost_311.txt`     | DGBM-LGB, DGBM-XGB (CPU), NGBoost               |
| `requirements-gpu_friendly.txt`       | DGBM CUDA benchmarks, PGBM                      |
| `requirements-lsf_openml_311.txt`     | XLSF (GluonTS)                                  |
| `requirements-gpboost.txt`            | GPBoost (Python 3.10, `gpboost==1.5.6`)         |

### 1.2 On VSC Hydra (how the published runs were produced)

On the cluster these venvs are layered on top of EasyBuild modules, and the
**module versions take precedence** over the same packages inside the venv.
The effective versions at run time are therefore the module ones. Load the
modules first, then activate the venv:

```bash
module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load LightGBM/4.5.0-foss-2023a
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0
source $VSC_SCRATCH/distiboost_311/bin/activate
```

Load the **whole** list, not just Python and LightGBM. The `jupyter-server`
module is what supplies `argon2-cffi`, a transitive dependency of `minio` and
therefore of `openml`; without it `import openml` fails with
`ModuleNotFoundError: No module named 'argon2'`. Off the cluster, add
`argon2-cffi` when installing from
`requirements-distiboost_311.txt` (the `pip freeze` does not capture
module-provided packages).

Effective versions under this combination (verified with
`reproduce/collect_env.sh`): Python 3.11.3, LightGBM 4.5.0, XGBoost 2.0.3,
PyTorch 2.1.2, NumPy 1.25.1, pandas 2.0.3, SciPy 1.11.1, scikit-learn 1.3.1,
Optuna 3.5.0, GluonTS 0.16.2, PGBM 2.3.0, openml 0.15.1, properscoring 0.1.

GPU jobs (PGBM, CUDA benchmarks) use instead:

```bash
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load LightGBM/4.5.0-foss-2023a-CUDA-12.1.1
source $VSC_SCRATCH/gpu_friendly/bin/activate
```

The GPBoost baseline used its own Python 3.10 venv, which no longer exists on
the cluster. `requirements-gpboost.txt` pins `gpboost==1.5.6` — the release
current when the GPBoost jobs ran, verified to expose exactly the API those
scripts call. Note that `openml_gpboost_HP_single_run.py` puts the sampled
`num_neighbors` into the booster parameter dictionary rather than passing it to
`GPModel`, so GPBoost ignores it (`[GPBoost] [Warning] Unknown parameter:
num_neighbors`) and the Vecchia approximation runs with its default neighbour
count. This is documented in the supplementary material rather than silently
fixed, because changing it would no longer reproduce the published numbers.

Run `bash reproduce/collect_env.sh` to regenerate the full environment report.

---

## 1.3 Datasets

Neither benchmark's raw data lives in this repository; both are fetched from
their canonical archives, which are themselves persistent.

**OpenML (19 datasets).** Retrieved at run time from OpenML benchmark suite 336
("Tabular benchmark numerical regression"). The scripts index
`openml.study.get_suite(336).tasks` by Slurm array position, so
`reproduce/openml_suite336_tasks.json` pins the mapping -- array index, task ID,
dataset ID, dataset **version** and target -- in case the suite is ever
reordered or a dataset is revised. Nothing needs downloading by hand; set
`OPENML_APIKEY` only if you hit rate limits.

**UCI (10 datasets).** These arrive three different ways, and one of them is a
trap for anyone cloning fresh:

| Source | Datasets |
|---|---|
| Vendored in the `ngboost` submodule | Kin8nm, Naval Propulsion, Combined Cycle Power Plant, Protein Structure |
| Downloaded from `archive.ics.uci.edu` at run time | Boston Housing, Concrete Compression Strength, Energy Efficiency, Wine Quality Red, Yacht Hydrodynamics |
| Gitignored inside the submodule, **absent from a fresh clone** | Year Prediction MSD (448 MB) |

Run this once after cloning:

```bash
bash reproduce/fetch_uci_data.sh
```

It downloads what is missing (including Year Prediction MSD, which otherwise
makes UCI dataset index 9 fail with `FileNotFoundError`), caches the five
network-fetched files under `reproduce/uci_cache/` so later runs do not depend
on UCI staying reachable, and verifies every file against the SHA-256 of the
exact bytes behind the published numbers. `--verify` checks without
downloading.

All five UCI URLs resolved when this was written, and all ten checksums matched.
A `MISMATCH` therefore means the upstream copy changed after publication, and
any difference in results should be attributed there first.

---

## 2. Experimental protocol (must match the paper)

| Item | OpenML suite 336 | UCI (NGBoost protocol) |
|------|------------------|------------------------|
| Splits | task-defined folds; fold 0 for HPO, remaining folds for evaluation | 20 random 90/10 permutations (5 for Protein Structure; 1 fixed split for Year MSD, first 463,715 rows = train) |
| Seed | `run_seed = 123` (TPE, booster seeds, train/val split `123+fold`, CRPS sampler `123 + 1000*task_id + fold`) | dataset index (0–9) seeds NumPy, TPE and the train/val split |
| HPO | once per dataset | repeated inside every fold |
| TPE trials | DGBM 80, GPBoost/PGBM 20, NGBoost/XLSF 10 | DGBM-LGB 200, DGBM-XGB/GPBoost 20, XLSF 10; NGBoost/PGBM use published configs |
| Study cap | 24 h wall-clock per study | 24 h wall-clock per study |
| Boosting rounds | ≤2000, chosen by early stopping | ≤2000, chosen by early stopping |
| Early-stopping patience | 20 rounds on the 20% validation split (inside tuning CV: 2000 for DGBM-LGB, 20 otherwise) | 20 rounds |
| CRPS | 100 samples per predictive distribution | 100 samples per predictive distribution |

---

## 3. Paper table/figure → script mapping

| Paper element | Script | Result file |
|---|---|---|
| Table: NLL (main) — DGBM-LGB, OpenML | `openml/openml_lssboost_HP_single_run.py` via `slurm/run_DGB-LGB_eval_HP_openml.sbatch` | `results/openml/openml_LSSboost_no_natural_exp_None_std_False_safety_False_job_*.csv` |
| Table: NLL (main) — DGBM-XGB, OpenML | `openml/openml_xglssboost_HP_single_run.py` via `slurm/run_DGB-XGB_eval_HP_openml.sbatch` | `results/openml/openml_XGBoostLSS_no_natural_exp_None_safety_False_job_*.csv` |
| Table: NLL (main) — DGBM-LGB, UCI | `uci/UCI_lssboost_single_run_HP.py` via `slurm/run_DGB-LGB_eval_HP_uci.sbatch` | `results/uci/uci_LSSboost_no_natural_exp_None.csv` |
| Table: NLL (main) — DGBM-XGB, UCI | `uci/UCI_xglssboost_HP_single_run.py` via `slurm/run_DGB-XGB_eval_HP_uci.sbatch` | `results/uci/uci_XGLSSboost_no_natural_exp_None.csv` |
| Table: NLL/CRPS — NGBoost | `openml/openml_ngboost_HP_single_run.py`, `uci/UCI_ngboost_single_run.py` | `results/openml/`, `results/uci/` |
| Table: NLL/CRPS — GPBoost | `openml/openml_gpboost_HP_single_run.py`, `uci/UCI_gpboost_HP_single_run.py` | `results/uci/uci_GPboost.csv` |
| Table: NLL/CRPS — PGBM | `openml/openml_pgbm_HP_single_run.py`, `uci/UCI_pgbm_single_run.py` | `results/openml/openml_PGBM_NLL_seeded.csv`, `results/uci/uci_pgbm.csv` |
| Table: NLL/CRPS — XLSF | `openml/openml_lsf_HP_single_run.py`, `uci/UCI_lsf_HP_single_run.py` | `results/openml/openml_GluonTS_LSF.csv`, `results/uci/uci_GluonTS_LSF.csv` |
| Supplementary: final selected hyperparameters | `reproduce/export_final_hps.py` | `logs/openml/lssboost/normal/exp/*.json`, `logs/openml/xgboost/normal/exp/*.json` |
| Supplementary: GPU acceleration figures | `slurm/run_LSS_cuda_synthetic_benchmark.sbatch`, `slurm/run_LGBM_cuda_synthetic_benchmark.sbatch` | `notebooks/lightgbm_vs_lightgbmlss_cuda_benchmark_analysis.ipynb` |
| Stabilization / natural-gradient ablation | same scripts with `natural_grad` and `stabilization` arguments (`None`, `L2`, `MAD`) | `results/*/..._{natural,no_natural}_exp_{None,L2,MAD}_*.csv` |

Configuration naming in result filenames:

- `natural` / `no_natural` — natural-gradient updates on/off
- `exp` / `softplus` — response function for the scale parameter
- `None` / `L2` / `MAD` — stabilization variant
- `std_True/False` — target standardization
- `safety_True/False` — post-hoc safety net on predicted parameters

The configuration reported in the main paper is
**`no_natural` + `exp` + stabilization `None` + `std_False` + `safety_False`**.

---

## 4. Reproducing a single result

```bash
# DGBM-LGB, OpenML task index 0, reported configuration
python openml/openml_lssboost_HP_single_run.py 0 exp False None None False 123 False
```

Positional arguments: `task_index mode natural_grad stabilization clip_value standardize run_seed apply_safety`.

```bash
# DGBM-LGB, UCI dataset index 0 (Boston Housing), reported configuration
python uci/UCI_lssboost_single_run_HP.py 0 exp False None None False
```

Positional arguments: `dataset_index mode natural_grad stabilization clip_value standardize [apply_safety]`.

To run every method over the whole OpenML suite from a single entry point,
use the unified runner instead of invoking each script by hand:

```bash
python reproduction/run_openml_benchmarks.py --list-profiles
python reproduction/run_openml_benchmarks.py --tasks 0-18 --dry-run
python reproduction/run_openml_benchmarks.py --tasks 0-18
```

Its default profiles are the configurations reported in the paper (`exp`
response function, no natural gradient, no stabilization, seed 123); the
`*_natural` and `*_mad` profiles are the ablations.

On the cluster, submit the corresponding array job instead:

```bash
sbatch slurm/run_DGB-LGB_eval_HP_openml.sbatch   # OpenML, 19 tasks
sbatch slurm/run_DGB-LGB_eval_HP_uci.sbatch      # UCI,    10 tasks
```

The array ranges are `--array=0-18` for the full OpenML suite and `--array=0-9`
for UCI.

Per-method Slurm wall-clock limits range from 36 h to 3 days; dataset–method
pairs that exceed them are reported as `---` in the paper.

---

## 5. Regenerating paper artifacts

```bash
# Supplementary tables of final selected hyperparameters
python reproduce/export_final_hps.py --out reproduce/tables

# Merge per-job result CSVs into one long-format table per suite
python reproduce/aggregate_results.py --out reproduce/tables

# Per-job hardware record (partition, node, memory, cores, elapsed, state)
sacct -j <job-ids> --format=JobID,JobName,Partition,NodeList,ReqMem,AllocCPUS,Elapsed,State -P -X \
  > reproduce/job_hardware.csv
```

`reproduce/job_hardware.csv` is shipped with the release. It records, for every
Slurm array task, the partition and node it ran on, so the reported runtimes can
be attributed to concrete hardware. It covers the OpenML run epoch (the UCI
result CSVs predate the job-id naming convention) and includes failed, cancelled
and out-of-memory attempts alongside the completed ones, so the sequence of
attempts is visible rather than filtered.

---

## 6. Known limitations of this release

- The environment used for **GPBoost** (`$VSC_SCRATCH/distiboost`, Python 3.10)
  no longer exists on the cluster, so its exact package version is not pinned
  here. Reproducing the GPBoost baseline requires reinstalling `gpboost` in a
  Python 3.10 environment.
- On UCI, hyperparameter optimization is repeated inside every fold, so there
  is no single selected configuration per dataset to tabulate; the selections
  are regenerated deterministically by re-running the scripts with the
  documented seeds.
- Result CSVs are written per Slurm job and several jobs cover overlapping
  subsets of datasets. Use `reproduce/aggregate_results.py` to merge them; it
  reports any dataset covered by more than one job so that conflicts are
  visible rather than silently resolved.

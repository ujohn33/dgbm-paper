# Reproducing the DGBM paper

Everything needed to rerun the experiments behind *Distributional Gradient
Boosting Machines* and to regenerate the paper's tables.

Nothing here is tied to a particular machine. The published runs were produced
on an HPC cluster under Slurm, and the `slurm/` job scripts are included as a
record, but every experiment is a plain Python invocation that runs anywhere.

---

## 1. Install

```bash
git clone --recurse-submodules https://github.com/ujohn33/dgbm-paper
cd dgbm-paper
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r reproduce/requirements-dgbm.txt
```

Four environments were used, because the baselines conflict with each other:

| File | Used for |
|---|---|
| `requirements-dgbm.txt` | DGBM-LGB, DGBM-XGB (CPU) and NGBoost |
| `requirements-gpu.txt` | PGBM and the CUDA scaling benchmarks |
| `requirements-xlsf.txt` | XLSF (GluonTS) |
| `requirements-gpboost.txt` | GPBoost (needs Python 3.10) |

Install each into its own virtualenv and activate the one matching the method
you are running. The first three are `pip freeze` dumps of the environments
that produced the results; `requirements-gpboost.txt` is written by hand,
because that environment was not preserved — see the note in the file.

Versions behind the published numbers: Python 3.11.3, LightGBM 4.5.0, XGBoost
2.0.3, PyTorch 2.1.2, NumPy 1.25.1, pandas 2.0.3, SciPy 1.11.1, scikit-learn
1.3.1, Optuna 3.5.0, openml 0.15.1, properscoring 0.1; baselines NGBoost
0.5.1.dev0, PGBM 2.3.0, GPBoost 1.5.6, GluonTS 0.16.2.

DGBM itself lives in the pinned submodules:

| Submodule | Upstream | Commit |
|---|---|---|
| `LightGBMLSS/` | ujohn33/LightGBMLSS_fork | `67b7698` |
| `XGBoostLSS/` | ujohn33/XGBoostLSS | `6e72532` |
| `ngboost/` | ujohn33/ngboost_fork | `db4d19b` |
| `pgbm/` | elephaint/pgbm | `e62b340` |
| `properscoring/` | tozech/properscoring | vendored |

If your site uses an environment-module stack layered under a virtualenv, load
the modules **before** activating it: module-provided packages take precedence
over the copies inside the virtualenv, and they are what actually gets
imported. `bash reproduce/collect_env.sh` reports what the active environment
resolves to.

---

## 2. Data

Neither benchmark's raw data is redistributed; both come from their canonical
archives.

**OpenML (19 datasets)** — fetched automatically at run time from benchmark
suite 336. Nothing to download by hand. The experiment scripts index the suite
*by position*, so `reproduce/openml_suite336_tasks.json` pins array index →
task ID → dataset ID → dataset **version**, in case the suite is ever reordered
or a dataset revised. Set `OPENML_APIKEY` only if you hit rate limits; public
tasks download without one.

**UCI (10 datasets)** — run this once:

```bash
python reproduce/fetch_uci_data.py           # download what is missing, then verify
python reproduce/fetch_uci_data.py --verify  # verify only
```

It uses the standard library only — no curl, unzip or sha256sum needed. It
downloads the five datasets the loaders would otherwise pull from
`archive.ics.uci.edu` at run time, and `YearPredictionMSD.txt` (448 MB), which
is gitignored inside the `ngboost` submodule and therefore absent from a fresh
clone — without it, UCI dataset index 9 fails with `FileNotFoundError`. Every
file is checked against the SHA-256 of the exact bytes behind the published
results, so a `MISMATCH` means the upstream copy changed after publication and
any difference in results should be attributed there first.

UCI dataset indices, in the order the scripts expect:

| Index | Dataset | Index | Dataset |
|---|---|---|---|
| 0 | Boston Housing | 5 | Combined Cycle Power Plant |
| 1 | Concrete Compression Strength | 6 | Protein Structure |
| 2 | Energy Efficiency | 7 | Wine Quality Red |
| 3 | Kin8nm | 8 | Yacht Hydrodynamics |
| 4 | Naval Propulsion | 9 | Year Prediction MSD |

---

## 3. Experimental protocol

| Item | OpenML suite 336 | UCI (NGBoost protocol) |
|------|------------------|------------------------|
| Splits | task-defined folds; fold 0 for HPO, the rest for evaluation | 20 random 90/10 permutations (5 for Protein Structure; 1 fixed split for Year MSD, first 463,715 rows = train) |
| Seed | `run_seed = 123` — TPE, booster seeds, train/val split `123+fold`, CRPS sampler `123 + 1000*task_id + fold` | the dataset index (0–9) seeds NumPy, TPE and the train/val split |
| HPO | once per dataset, on fold 0 | repeated inside every fold |
| TPE trials | DGBM 80; GPBoost, PGBM 20; NGBoost, XLSF 10 | DGBM-LGB 200; DGBM-XGB, GPBoost 20; XLSF 10. NGBoost and PGBM use their published configurations |
| Study cap | 24 h wall-clock per study | 24 h wall-clock per study |
| Boosting rounds | ≤2000, chosen by early stopping | ≤2000, chosen by early stopping |
| Early-stopping patience | 20 rounds on the 20% validation split (inside the tuning CV: 2000 for DGBM-LGB, 20 otherwise) | 20 rounds |
| CRPS | 100 samples per predictive distribution | 100 samples per predictive distribution |

---

## 4. Running the experiments

### OpenML — all methods from one command

```bash
python reproduce/run_openml_benchmarks.py --list-profiles
python reproduce/run_openml_benchmarks.py --tasks 0-18 --dry-run
python reproduce/run_openml_benchmarks.py --tasks 0-18
```

The default profiles are the configurations reported in the paper (`exp`
response function, no natural gradient, no stabilization, seed 123); the
`*_natural` and `*_mad` profiles are the ablations. Output goes to
`reproduce/runs/run_<timestamp>/`. `--max-runs` is useful for a smoke test.

A single OpenML dataset–method pair directly:

```bash
python openml/openml_lssboost_HP_single_run.py 0 exp False None None False 123 False
```

Arguments: `task_index mode natural_grad stabilization clip_value standardize run_seed apply_safety`.

### UCI

There is no unified runner for UCI, because each method takes a different
argument list. Each command below takes a dataset index from the table in
section 2; loop `0..9` for the full suite.

```bash
python reproduce/fetch_uci_data.py            # once, before anything else

# DGBM-LGB and DGBM-XGB, the configuration reported in the paper
for i in $(seq 0 9); do
  python uci/UCI_lssboost_single_run_HP.py   "$i" exp False None None False
  python uci/UCI_xglssboost_HP_single_run.py "$i" exp False None None False
done
```

Arguments for both: `dataset_index mode natural_grad stabilization clip_value standardize [apply_safety]`.
Swap `False` → `True` in the third position for the natural-gradient ablation,
or `None` → `L2` / `MAD` in the fourth for the stabilization ablation.

Baselines, one dataset index each:

```bash
python uci/UCI_ngboost_single_run.py    "$i"        # published configuration
python uci/UCI_pgbm_single_run.py       "$i"        # published configuration
python uci/UCI_gpboost_HP_single_run.py "$i"        # needs the gpboost env
python uci/UCI_lsf_HP_single_run.py     "$i" 123    # needs the xlsf env
```

Results are written to `results/uci/`.

### On a batch cluster

`slurm/` holds the job scripts used for the published runs, as array jobs:
`--array=0-18` for OpenML and `--array=0-9` for UCI. They are site-specific —
the module loads and environment paths at the top need replacing with your own
— but they record the resources each method was given (36 h to 3 days
wall-clock, 40–100 GB memory, GPU only for PGBM).

---

## 5. Regenerating the paper's artifacts

```bash
# The paper's figures: CD diagrams and the runtime boxplots
python reproduce/make_figures.py --out reproduce/figures

# Supplementary tables of final selected hyperparameters
python reproduce/export_final_hps.py --out reproduce/tables

# Merge the per-job result CSVs into one long-format table per suite
python reproduce/aggregate_results.py --out reproduce/tables

# Record the environment
bash reproduce/collect_env.sh > reproduce/env_report.txt
```

`make_figures.py` writes `cd_diagram_{NLL,CRPS,RMSE}_{uci,openml}.png` and the
`run_time_figure.png` / `hp_time_figure.png` boxplots. The CD diagrams follow
Demšar (2006) as refined by Benavoli et al. (2016) — per-dataset ranks, a
Friedman test, then pairwise Wilcoxon signed-rank tests with Holm correction to
decide which methods are joined by a bar. The published figures were drawn with
`critdd`; this reimplements the same procedure with scipy so the repository
carries no extra dependency. Only datasets on which every method has a result
are ranked, and the count is printed with each diagram.

**These figures do not reproduce the published ones cell for cell.** On UCI,
DGBM-LGB, GPBoost, PGBM and XLSF match the paper's NLL table exactly, but the
DGBM-XGB and NGBoost CSVs kept here come from different runs than the ones
tabulated, and no GPBoost OpenML result file was preserved. The regenerated
diagrams therefore differ slightly in rank values, though not in the ordering of
the leading methods. Rerunning those method/suite combinations with the scripts
in `openml/` and `uci/` regenerates the missing inputs.

`reproduce/job_hardware.csv` records the partition, node, memory and wall-clock
time of every batch array task behind the published OpenML runs, so reported
runtimes can be attributed to concrete hardware. It includes failed, cancelled
and out-of-memory attempts alongside the completed ones.

---

## 6. Paper element → script → result file

| Paper element | Script | Result file |
|---|---|---|
| NLL/CRPS — DGBM-LGB, OpenML | `openml/openml_lssboost_HP_single_run.py` | `results/openml/openml_LSSboost_no_natural_exp_None_std_False_safety_False_job_*.csv` |
| NLL/CRPS — DGBM-XGB, OpenML | `openml/openml_xglssboost_HP_single_run.py` | `results/openml/openml_XGBoostLSS_no_natural_exp_None_safety_False_job_*.csv` |
| NLL/CRPS — DGBM-LGB, UCI | `uci/UCI_lssboost_single_run_HP.py` | `results/uci/uci_LSSboost_no_natural_exp_None.csv` |
| NLL/CRPS — DGBM-XGB, UCI | `uci/UCI_xglssboost_HP_single_run.py` | `results/uci/uci_XGLSSboost_no_natural_exp_None.csv` |
| NLL/CRPS — NGBoost | `openml/openml_ngboost_HP_single_run.py`, `uci/UCI_ngboost_single_run.py` | `results/openml/`, `results/uci/` |
| NLL/CRPS — GPBoost | `openml/openml_gpboost_HP_single_run.py`, `uci/UCI_gpboost_HP_single_run.py` | `results/uci/uci_GPboost.csv` |
| NLL/CRPS — PGBM | `openml/openml_pgbm_HP_single_run.py`, `uci/UCI_pgbm_single_run.py` | `results/openml/openml_PGBM_NLL_seeded.csv`, `results/uci/uci_pgbm.csv` |
| NLL/CRPS — XLSF | `openml/openml_lsf_HP_single_run.py`, `uci/UCI_lsf_HP_single_run.py` | `results/openml/openml_GluonTS_LSF.csv`, `results/uci/uci_GluonTS_LSF.csv` |
| Supplementary: final hyperparameters | `reproduce/export_final_hps.py` | `logs/openml/{lssboost,xgboost}/normal/exp/*.json` |
| Supplementary: GPU acceleration figures | `synthetic/lightgbmlss_cuda_benchmark.py`, `synthetic/lightgbm_cuda_benchmark.py` | `notebooks/lightgbm_vs_lightgbmlss_cuda_benchmark_analysis.ipynb` |
| Stabilization / natural-gradient ablation | the same scripts with different `natural_grad` and `stabilization` arguments | `results/*/..._{natural,no_natural}_exp_{None,L2,MAD}_*.csv` |

Result filenames encode the configuration: `natural`/`no_natural`
(natural-gradient updates), `exp`/`softplus` (scale response function),
`None`/`L2`/`MAD` (stabilization), `std_*` (target standardization) and
`safety_*` (post-hoc guard on predicted parameters). **The configuration
reported in the main paper is `no_natural` + `exp` + `None` + `std_False` +
`safety_False`.**

---

## 7. Known limitations of this release

- GPBoost ran in a Python 3.10 environment that was not preserved, so
  `requirements-gpboost.txt` pins a compatible version rather than a captured
  one. It was verified to expose exactly the API the scripts call.
- `openml/openml_gpboost_HP_single_run.py` puts the sampled `num_neighbors`
  into the booster parameter dictionary instead of passing it to `GPModel`, so
  GPBoost ignores it and the Vecchia approximation uses its default neighbour
  count. This is documented rather than fixed, because changing it would no
  longer reproduce the published numbers.
- On UCI, hyperparameter optimization is repeated inside every fold, so there
  is no single selected configuration per dataset to tabulate. The selections
  are regenerated deterministically by rerunning with the documented seeds.
- Result CSVs are written per job and several jobs cover overlapping subsets of
  datasets. `reproduce/aggregate_results.py` merges them and writes
  `duplicates.csv` listing every `(configuration, dataset)` pair that appears in
  more than one job, so conflicts are visible rather than silently resolved.

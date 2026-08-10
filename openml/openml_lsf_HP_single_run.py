import os
import sys
import json
import csv
import random
import time
import warnings

_worker_warning_filters = [
    "ignore::UserWarning:sklearn.utils.parallel",
    "ignore::UserWarning:gluonts.json",
]
_existing_pywarnings = os.environ.get("PYTHONWARNINGS", "").strip()
if _existing_pywarnings:
    os.environ["PYTHONWARNINGS"] = ",".join(_worker_warning_filters + [_existing_pywarnings])
else:
    os.environ["PYTHONWARNINGS"] = ",".join(_worker_warning_filters)

import openml
import numpy as np
import optuna
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gluonts.ext.rotbaum._model import LSF
from utils.metrics import crps, quantile_loss
from utils.logging import log_predictions


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


print("Usage: python openml_lsf_HP_single_run.py <task_idx> [run_seed]")
run_seed = 123 if len(sys.argv) <= 2 else int(sys.argv[2])
seed_everything(run_seed)

# Set OpenML API key
openml.config.apikey = os.environ.get("OPENML_APIKEY", "")

# Suppress repetitive warnings that would bloat logs
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.utils\.parallel")
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"gluonts\.json",
)

args = {
    "SUITE_ID": 336,  # Regression on numerical features
    "n_splits": 5,
    "n_trials": 10,
    "n_forecasts": 100,
    "score": "LSF",
    "distn": "Empirical",
    "natural_grad": False,
}

method_name = "GluonTS_LSF"
benchmark_suite = openml.study.get_suite(args["SUITE_ID"])
WRITE_PREDICTIONS = os.environ.get("OPENML_LSF_WRITE_PREDICTIONS", "0") == "1"
WRITE_OPT_PARAMS = os.environ.get("OPENML_LSF_WRITE_OPT_PARAMS", "0") == "1"


def encode_categorical_series(y):
    if y.dtype.name == "category" or y.dtype == "object":
        y = y.astype("category").cat.codes
    return y


def encode_categorical_columns(df):
    for col in df.select_dtypes(include=["category", "object"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


class LevelSetForecaster:
    """Wrapper around exact GluonTS Rotbaum LSF implementation (LSF=QRX)."""

    class _SafeBoolModel:
        """Avoid sklearn unfitted __len__ call when GluonTS does `if model:`."""

        def __init__(self, model):
            self._model = model

        def __bool__(self):
            return True

        def fit(self, *args, **kwargs):
            return self._model.fit(*args, **kwargs)

        def predict(self, *args, **kwargs):
            return self._model.predict(*args, **kwargs)

        @property
        def wrapped_model(self):
            return self._model

    def __init__(self, params, seed):
        self.seed = seed
        base_model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            n_jobs=-1,
            random_state=seed,
        )
        self.model = LSF(
            model=LevelSetForecaster._SafeBoolModel(base_model),
            min_bin_size=int(params["min_bin_size"]),
        )

    def fit(self, X, y):
        self.model.fit(
            X,
            y,
            seed=self.seed,
            x_train_is_dataframe=False,
            model_is_already_trained=False,
        )
        return self

    def predict_point(self, X):
        return np.asarray(self.model.model.predict(X), dtype=float)

    def predict_samples(self, X, n_forecasts, seed):
        bins = self.model.estimate_dist(X.tolist())
        point_pred = self.predict_point(X)
        rng = np.random.default_rng(seed)
        n_test = len(bins)
        samples = np.empty((n_test, n_forecasts), dtype=float)
        for i, bin_values in enumerate(bins):
            values = np.asarray(bin_values, dtype=float)
            if values.size == 0:
                samples[i, :] = point_pred[i]
            else:
                samples[i, :] = rng.choice(values, size=n_forecasts, replace=True)
        return samples


def tune_params(X_train_opt, y_train_opt):
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_train_opt,
        y_train_opt,
        test_size=0.2,
        random_state=run_seed,
        shuffle=True,
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_bin_size": trial.suggest_int("min_bin_size", 100, 500),
            "n_forecasts": args["n_forecasts"],
        }
        model = LevelSetForecaster(params=params, seed=run_seed).fit(X_tune, y_tune)
        samples = model.predict_samples(
            X_val,
            n_forecasts=args["n_forecasts"],
            seed=run_seed + 17,
        )
        mu = samples.mean(axis=1)
        floor = max(5e-2 * np.std(y_tune), 1e-3)
        std = np.maximum(samples.std(axis=1, ddof=1), floor)
        nll = -norm(mu, std).logpdf(y_val).mean()
        return nll

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=run_seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=args["n_trials"], timeout=86400)
    return study.best_params


def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    X = encode_categorical_columns(X)
    y = encode_categorical_series(y)
    X = X.to_numpy()
    y = np.asarray(y)

    lss_rmse, lss_nll, times = [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    print(
        f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}"
    )
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(
        f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}."
    )

    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X[train_indices], X[test_indices]
    y_train_opt, y_test_opt = y[train_indices], y[test_indices]

    start_time_hp = time.time()
    print("Hyperparameter tuning...")
    best_params = tune_params(X_train_opt, y_train_opt)
    elapsed_time_hp = time.time() - start_time_hp
    print("Best hyperparameters:", best_params)

    if WRITE_OPT_PARAMS:
        out_dir = "logs/openml/lsf"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{dataset.name}_opt_params.json", "w") as f:
            json.dump(best_params, f)

    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_trainall, X_test = X[train_indices], X[test_indices]
        y_trainall, y_test = y[train_indices], y[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainall,
            y_trainall,
            test_size=0.2,
            random_state=run_seed + fold,
            shuffle=True,
        )

        val_model = LevelSetForecaster(params=best_params, seed=run_seed + fold).fit(X_train, y_train)
        val_point = val_model.predict_point(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_point))

        start_time = time.time()
        final_model = LevelSetForecaster(params=best_params, seed=run_seed + fold).fit(X_trainall, y_trainall)
        yhat_point = final_model.predict_point(X_test)
        samples = final_model.predict_samples(
            X_test,
            n_forecasts=args["n_forecasts"],
            seed=run_seed + task_id * 1000 + fold,
        )
        runtime_pred = time.time() - start_time

        mu = samples.mean(axis=1)
        y_scale = np.std(y_trainall)
        std_floor = max(5e-2 * y_scale, 1e-3)
        std = np.maximum(samples.std(axis=1, ddof=1), std_floor)

        rmse = np.sqrt(mean_squared_error(y_test, yhat_point))
        nll_test = -norm(mu, std).logpdf(y_test).mean()
        crps_comps = crps(y_test, samples)
        crps_test = crps_comps[0]
        crps_cal, crps_sha = crps_comps[1], crps_comps[2]

        lss_rmse.append(rmse)
        lss_nll.append(nll_test)
        lss_crps.append(crps_test)
        lss_crps_cal.append(crps_cal)
        lss_crps_sha.append(crps_sha)
        times.append(runtime_pred)

        quantiles = [0.1, 0.5, 0.9]
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = np.quantile(samples, q, axis=1)
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)

        if WRITE_PREDICTIONS:
            log_predictions(
                fold,
                dataset.name,
                y_test,
                mu,
                std,
                quantile_preds,
                f"logs/openml/predictions/{method_name}.csv",
            )

        wql_avg_fold = np.mean(quantile_losses)
        wql_01.append(quantile_losses[0])
        wql_05.append(quantile_losses[1])
        wql_09.append(quantile_losses[2])
        wql_avg.append(wql_avg_fold)

        print(
            "[%d/%d] RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
            % (
                fold,
                n_folds,
                val_rmse,
                lss_rmse[-1],
                lss_nll[-1],
                lss_crps[-1],
                lss_crps_cal[-1],
                lss_crps_sha[-1],
                runtime_pred,
            )
        )

    return (
        dataset.name,
        np.mean(lss_rmse),
        np.std(lss_rmse),
        np.mean(lss_nll),
        np.std(lss_nll),
        np.mean(lss_crps),
        np.std(lss_crps),
        np.mean(lss_crps_cal),
        np.std(lss_crps_cal),
        np.mean(lss_crps_sha),
        np.std(lss_crps_sha),
        np.mean(times),
        elapsed_time_hp,
        np.mean(wql_01),
        np.std(wql_01),
        np.mean(wql_05),
        np.std(wql_05),
        np.mean(wql_09),
        np.std(wql_09),
        np.mean(wql_avg),
        np.std(wql_avg),
    )


if __name__ == "__main__":
    print("GLUONTS LSF")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    task = openml.tasks.get_task(task_number)
    dataset = task.get_dataset()
    print(f"Selected OpenML task: {task_number} ({dataset.name})")
    results = run_single_argument(task_number)

    file_path = "results/openml/openml_GluonTS_LSF.csv"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    header = [
        "dset",
        "RMSE-mean",
        "RMSE-std",
        "NLL-mean",
        "NLL-std",
        "CRPS-mean",
        "CRPS-std",
        "CRPS-calibration-mean",
        "CRPS-calibration-std",
        "CRPS-sharpness-mean",
        "CRPS-sharpness-std",
        "time_run",
        "time_HP",
        "WQL01-mean",
        "WQL01-std",
        "WQL05-mean",
        "WQL05-std",
        "WQL09-mean",
        "WQL09-std",
        "WQL_avg-mean",
        "WQL_avg-std",
    ]

    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a+", newline="") as file:
        writer = csv.writer(file)
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(header)
        writer.writerow(
            [
                results[0],
                results[1],
                results[2],
                results[3],
                results[4],
                results[5],
                results[6],
                results[7],
                results[8],
                results[9],
                results[10],
                results[11],
                results[12],
                results[13],
                results[14],
                results[15],
                results[16],
                results[17],
                results[18],
                results[19],
                results[20],
            ]
        )

import csv
import os
import random
import sys
import time

import numpy as np
import optuna
import pandas as pd
from gluonts.ext.rotbaum._model import LSF
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logging import log_predictions
from utils.metrics import crps, quantile_loss


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


dataset_name_to_loader = {
    "Boston Housing": lambda: pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        header=None,
        delim_whitespace=True,
    ),
    "Concrete Compression Strength": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    ),
    "Energy Efficiency": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    ).iloc[:, :-1],
    "Kin8nm": lambda: pd.read_csv("ngboost/data/uci/kin8nm.csv"),
    "Naval Propulsion": lambda: pd.read_csv(
        "ngboost/data/uci/naval-propulsion.txt", delim_whitespace=True, header=None
    ).iloc[:, :-1],
    "Combined Cycle Power Plant": lambda: pd.read_excel("ngboost/data/uci/power-plant.xlsx"),
    "Protein Structure": lambda: pd.read_csv("ngboost/data/uci/protein.csv")[
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "RMSD"]
    ],
    "Wine Quality Red": lambda: pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        delimiter=";",
    ),
    "Yacht Hydrodynamics": lambda: pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
        header=None,
        delim_whitespace=True,
    ),
    "Year Prediciton MSD": lambda: pd.read_csv("ngboost/data/uci/YearPredictionMSD.txt").iloc[:, ::-1],
}

dataset_list = [
    "Boston Housing",
    "Concrete Compression Strength",
    "Energy Efficiency",
    "Kin8nm",
    "Naval Propulsion",
    "Combined Cycle Power Plant",
    "Protein Structure",
    "Wine Quality Red",
    "Yacht Hydrodynamics",
    "Year Prediciton MSD",
]

args = {
    "n_splits": 20,
    "n_trials": 20,
    "n_forecasts": 200,
    "score": "LSF",
    "distn": "Empirical",
}

method_name = "GluonTS_LSF"
WRITE_PREDICTIONS = os.environ.get("UCI_WRITE_PREDICTIONS", "0") == "1"


class LevelSetForecaster:
    class _SafeBoolModel:
        def __init__(self, model):
            self._model = model

        def __bool__(self):
            return True

        def fit(self, *fit_args, **fit_kwargs):
            return self._model.fit(*fit_args, **fit_kwargs)

        def predict(self, *predict_args, **predict_kwargs):
            return self._model.predict(*predict_args, **predict_kwargs)

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
        samples = np.empty((len(bins), n_forecasts), dtype=float)
        for i, bin_values in enumerate(bins):
            values = np.asarray(bin_values, dtype=float)
            if values.size == 0:
                samples[i, :] = point_pred[i]
            else:
                samples[i, :] = rng.choice(values, size=n_forecasts, replace=True)
        return samples


def build_folds(X, dataset_name, seed):
    if dataset_name == "Year Prediciton MSD":
        return [(np.arange(463715), np.arange(463715, len(X)))]

    n = X.shape[0]
    n_folds = 5 if dataset_name == "Protein Structure" else args["n_splits"]
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(n_folds):
        permutation = rng.choice(np.arange(n), n, replace=False)
        end_train = round(n * 9.0 / 10)
        train_index = permutation[:end_train]
        test_index = permutation[end_train:]
        folds.append((train_index, test_index))
    return folds


def tune_params(X_train, y_train, seed):
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_bin_size": trial.suggest_int("min_bin_size", 20, 500),
        }
        model = LevelSetForecaster(params=params, seed=seed).fit(X_tune, y_tune)
        samples = model.predict_samples(X_val, n_forecasts=args["n_forecasts"], seed=seed + 7)
        mu = samples.mean(axis=1)
        std = np.maximum(samples.std(axis=1), 1e-6)
        return -norm(mu, std).logpdf(y_val).mean()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=args["n_trials"], timeout=86400)
    return study.best_params


def run_single_argument(dataset_idx, run_seed):
    dset = dataset_list[int(dataset_idx)]
    data = dataset_name_to_loader[dset]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={dset} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    folds = build_folds(X, dset, seed=run_seed)

    lss_rmse, lss_nll, times, times_hp = [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    for fold_idx, (train_index, test_index) in enumerate(folds):
        print(f"{dset}: fold {fold_idx + 1}/{len(folds)}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        hp_start = time.time()
        best_params = tune_params(X_train, y_train, seed=run_seed + fold_idx)
        elapsed_time_hp = time.time() - hp_start
        times_hp.append(elapsed_time_hp)
        print(f"Best hyperparameters for fold {fold_idx + 1}: {best_params}")

        start_time = time.time()
        model = LevelSetForecaster(params=best_params, seed=run_seed + fold_idx).fit(X_train, y_train)
        yhat_point = model.predict_point(X_test)
        samples = model.predict_samples(
            X_test,
            n_forecasts=args["n_forecasts"],
            seed=run_seed + int(dataset_idx) * 1000 + fold_idx,
        )
        runtime_pred = time.time() - start_time
        times.append(runtime_pred)

        mu = samples.mean(axis=1)
        std = np.maximum(samples.std(axis=1), 1e-6)

        rmse = np.sqrt(mean_squared_error(y_test, yhat_point))
        nll_test = -norm(mu, std).logpdf(y_test).mean()
        crps_comps = crps(y_test, samples)

        lss_rmse.append(rmse)
        lss_nll.append(nll_test)
        lss_crps.append(crps_comps[0])
        lss_crps_cal.append(crps_comps[1])
        lss_crps_sha.append(crps_comps[2])

        quantiles = [0.1, 0.5, 0.9]
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = np.quantile(samples, q, axis=1)
            quantile_losses.append(quantile_loss(q, y_test, quantile_preds[str(q)]).mean())

        if WRITE_PREDICTIONS:
            try:
                log_predictions(
                    fold_idx,
                    dset,
                    y_test,
                    mu,
                    std,
                    quantile_preds,
                    f"logs/uci/predictions/{method_name}.csv",
                )
            except OSError as exc:
                print(f"[warn] Skipping prediction logging due to I/O error: {exc}")

        wql_01.append(quantile_losses[0])
        wql_05.append(quantile_losses[1])
        wql_09.append(quantile_losses[2])
        wql_avg.append(np.mean(quantile_losses))

        print(
            "[%d/%d] RMSE=%.4f NLL=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
            % (
                fold_idx + 1,
                len(folds),
                lss_rmse[-1],
                lss_nll[-1],
                lss_crps[-1],
                lss_crps_cal[-1],
                lss_crps_sha[-1],
                runtime_pred,
            )
        )

    print(f"Completed dataset: {dset}")
    return (
        dset,
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
        np.mean(times_hp),
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
    print("GLUONTS LSF UCI")
    print("______________________")
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python uci/UCI_lsf_HP_single_run.py <dataset_idx> [run_seed]")

    dataset_idx = int(sys.argv[1])
    run_seed = 123 if len(sys.argv) <= 2 else int(sys.argv[2])
    seed_everything(run_seed)

    results = run_single_argument(dataset_idx=dataset_idx, run_seed=run_seed)

    file_path = f"results/uci/uci_{method_name}.csv"
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
        writer.writerow(list(results))

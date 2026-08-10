import openml
import torch
import os
import sys
import json
import numpy as np
import pandas as pd
import time
import csv
from sklearn.metrics import mean_squared_error
from pgbm.torch import PGBMRegressor
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path
import optuna
from pgbm.torch import PGBM
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = os.environ.get("OPENML_APIKEY", "")

SEED = 123

import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define constants and parameters
SUITE_ID = 336 # Regression on numerical features
method_name = 'pgbm'
np.random.seed(1)
mode = 'exp'
natural_flag = False
n_forecasts = 200
# Global model instance for reuse
GLOBAL_MODEL = None

# Hardcoded parameters for testing
args = {
    "distn": "Normal",
    "SUITE_ID": 336, # Regression on numerical features
    "n_splits": 5,
    "score": "MLE",
}


# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(args["SUITE_ID"]) 

def encode_categorical_series(y):
    # Check if the series is of type 'category' or 'object' (strings)
    if y.dtype.name == 'category' or y.dtype == 'object':
        y = y.astype('category').cat.codes  # Convert to category first, then encode
    return y

def encode_categorical_columns(df):
    # Iterate over columns that are either categorical or contain strings (objects)
    for col in df.select_dtypes(include=['category', 'object']).columns:
        df[col] = df[col].astype('category').cat.codes  # Convert to category first, then encode
    return df

def objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()
    return loss

def warmup_pgbm_jit():
    """Pre-compile PGBM JIT functions globally"""
    print("Warming up PGBM JIT compilation globally...")
    dummy_X = np.random.randn(100, 5)
    dummy_y = np.random.randn(100)
    dummy_params = {
        'n_estimators': 10,
        'learning_rate': 0.1,
        'max_leaves': 8,
        'device': 'gpu',
        'verbose': 0,
    }
    warmup_model = PGBM()
    warmup_model.train(
        (dummy_X, dummy_y),
        objective=objective,
        metric=rmseloss_metric,
        params=dummy_params,
    )
    print("Global JIT compilation complete.")

class ObjectivePGBM:
    def __init__(self, X_train, y_train, dataset_name=None, bagging_fraction=1.0,
                 nfold=5, num_boost_round=2000, seed=123, metric="rmse",
                 n_forecasts=200):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.dataset_name = dataset_name
        self.bagging_fraction = bagging_fraction
        self.nfold = nfold
        self.num_boost_round = num_boost_round
        self.seed = seed
        self.metric = metric.lower()
        self.n_forecasts = n_forecasts

    def __call__(self, trial):
        params = {
            "n_estimators": self.num_boost_round,
            "bagging_fraction": self.bagging_fraction,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
            "max_leaves": trial.suggest_int("max_leaves", 8, 32),
            "max_bin": trial.suggest_int("max_bin", 32, 256),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "device": "gpu",
            "verbose": 0,
            "feature_fraction": 1.0,
            "derivatives": "exact",
            "distribution": "normal",
        }

        kf = KFold(n_splits=self.nfold, shuffle=True, random_state=self.seed)
        fold_scores = []
        fold_rounds = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(self.X_train)):
            X_tr, X_va = self.X_train[tr_idx], self.X_train[va_idx]
            y_tr, y_va = self.y_train[tr_idx], self.y_train[va_idx]

            model = PGBM()
            model.train(
                (X_tr, y_tr),
                objective=objective,
                metric=rmseloss_metric,
                valid_set=(X_va, y_va),
                params=params,
            )
            torch.cuda.synchronize()

            best_iter = int(model.best_iteration)
            fold_rounds.append(best_iter)

            if self.metric == "rmse":
                yhat = model.predict(X_va, parallel=False).cpu().numpy()
                score = np.sqrt(mean_squared_error(y_va, yhat))

            elif self.metric == "nll":
                _, mu, var = model.predict_dist(
                    X_va,
                    n_forecasts=self.n_forecasts,
                    parallel=False,
                    output_sample_statistics=True,
                )
                mu = mu.cpu().numpy()
                std = np.sqrt(var.cpu().numpy())
                score = -norm(mu, std).logpdf(y_va).mean()

            elif self.metric == "crps":
                yhat_dist = model.predict_dist(
                    X_va,
                    n_forecasts=self.n_forecasts,
                    parallel=False,
                )
                yhat_dist = yhat_dist.reshape(yhat_dist.shape[1], yhat_dist.shape[0]).cpu().numpy()
                score = crps(y_va, yhat_dist)[0]

            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            fold_scores.append(score)

        opt_round = int(np.median(fold_rounds))
        trial.set_user_attr("opt_round", opt_round)

        # Optuna minimizes here
        return float(np.mean(fold_scores))

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    # Encode categorical columns
    X = encode_categorical_columns(X)
    y = encode_categorical_series(y)

    global GLOBAL_MODEL

    # Set bagging_fraction based on dataset size
    bagging_fraction = 0.1 if len(y) > 50000 else 1.0
    print(f"Dataset size: {len(y)}, using bagging_fraction: {bagging_fraction}")

    lss_rmse, lss_nll, times, times_HP = [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['distn']}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X.iloc[train_indices], X.iloc[test_indices]
    y_train_opt, y_test_opt = y.iloc[train_indices], y.iloc[test_indices]

    train_opt_data = (X_train_opt.values, y_train_opt.values)

    # Hyperparameter optimization with Optuna
    start_time = time.time()
    print('Hyperparameter tuning...')
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective_tuning = ObjectivePGBM(
        X_train_opt.values,
        y_train_opt.values,
        dataset_name=dataset.name,
        bagging_fraction=bagging_fraction,
        nfold=5,
        num_boost_round=200,
        seed=123,
        metric="nll",   # or "rmse" or "crps"
        n_forecasts=n_forecasts,
    )
    study.optimize(objective_tuning, n_trials=20, timeout=86400)
    end_time = time.time()  # End time measurement
    elapsed_time_HP = end_time - start_time  # Calculate elapsed time

    # Set the best parameters and number of estimators from hyperparameter tuning
    best_params = study.best_trial.params
    best_params["opt_rounds"] = int(study.best_trial.user_attrs["opt_round"])
    print(best_params)
    print(f'Best hyperparameters for fold 0: {best_params}')

    base_train_params = {
        'n_estimators': 200,
        'bagging_fraction': bagging_fraction,
        'device': 'gpu',
        'verbose': 2,
        'feature_fraction': 1,
        'derivatives': 'exact',
        'distribution': 'normal',
    }

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_trainall, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_trainall, y_test = y.iloc[train_indices], y.iloc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2, random_state=SEED)

        train_data = (X_trainall.values, y_trainall.values)
        train_val_data = (X_train.values, y_train.values)

        fold_params = {**base_train_params, **best_params}

        print("X_train.shape:", X_train.shape)
        print("monotone_constraints len:",
            None if fold_params.get("monotone_constraints") is None
            else len(fold_params["monotone_constraints"]))

        if 'monotone_constraints' in fold_params:
            n_features = X_val.shape[1]
            if len(fold_params['monotone_constraints']) != n_features:
                del fold_params['monotone_constraints']

        print('Training final model...')
        start_time = time.time()
        n_rounds = fold_params.pop("opt_rounds")
        fold_params["n_estimators"] = n_rounds

        final_model = PGBM()
        final_model.train(
            train_data,
            objective=objective,
            metric=rmseloss_metric,
            params=fold_params,
        )
        torch.cuda.synchronize()
        training_time = time.time() - start_time

        print('Prediction...')
        yhat_point = final_model.predict(X_test.values)
        yhat_dist, mu, var = final_model.predict_dist(
            X_test.values,
            n_forecasts=n_forecasts,
            parallel=False,
            output_sample_statistics=True,
        )
        std = np.sqrt(var.cpu().numpy())

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(yhat_point.cpu().numpy(), y_test))
        nll_test = -norm(mu.cpu().numpy(), std).logpdf(y_test).mean()
    
        yhat_dist = yhat_dist.reshape(yhat_dist.shape[1], yhat_dist.shape[0])
        crps_comps = crps(y_test, yhat_dist.cpu().numpy())
        crps_test = crps_comps[0]
        crps_cal, crps_sha = crps_comps[1], crps_comps[2]

        # Store results
        lss_rmse.append(rmse)
        lss_nll.append(nll_test)
        lss_crps.append(crps_test)
        lss_crps_cal.append(crps_cal)
        lss_crps_sha.append(crps_sha)
        times += [training_time]
        times_HP += [elapsed_time_HP]

        # Define the quantiles to evaluate
        quantiles = [0.1, 0.5, 0.9]

        # Compute the quantiles for each observation
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = norm.ppf(q, loc=mu.cpu().numpy(), scale=std)
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)

        # Log predictions for each fold
        #log_predictions(fold, dset_name, y_test.values, mu, std, quantile_preds, f"logs/openml/predictions/{method_name}.csv")
        
        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]  

    print(f'Completed dataset: {dataset.name}')
    # return a dictonary of val
    return  dataset.name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP), np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 



if __name__ == "__main__":
    print("PGBM")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    vsc_data = os.environ['VSC_DATA']
    # Call this once at module level
    warmup_pgbm_jit()

    results = run_single_argument(task_number)
    file_path = "results/openml/openml_PGBM_NLL_seeded.csv"
    header = ["dset","RMSE-mean","RMSE-std","NLL-mean","NLL-std","CRPS-mean","CRPS-std","CRPS-calibration-mean","CRPS-calibration-std","CRPS-sharpness-mean","CRPS-sharpness-std","time_run","time_HP","WQL01-mean", "WQL01-std","WQL05-mean", "WQL05-std","WQL09-mean", "WQL09-std", "WQL_avg-mean", "WQL_avg-std"]
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    # Open the file in append mode ('a+')
    with open(file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist or is empty, write the header
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(header)  # Write header

        # Write the results to the file as a list
        row_to_write = [results[0], results[1], results[2], results[3], results[4],
                        results[5], results[6], results[7], results[8], results[9],
                        results[10], results[11], results[12], results[13],
                        results[14], results[15], results[16], results[17],
                        results[18], results[19], results[20]]

        writer.writerow(row_to_write)
import os
import sys
import openml
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import time
from scipy.stats import norm as norm_dist
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import Bernoulli, Normal
from ngboost.scores import LogScore
from ngboost import NGBRegressor
from ngboost.learners import default_linear_learner, default_tree_learner
import optuna
from properscoring._mean_crps import _mean_crps_hersbach

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = '0fc137c28db32cdfecb6347178c7be68'

# Define constants and parameters
SUITE_ID = 336  # Regression on numerical features
np.random.seed(1)
mode = 'exp'
natural_flag = True
args = {
    "reps": 3,
    "n_est": 2000,
    "n_splits": 5,
    "score": "LogScore",
    "distn": "Normal",
    "base": "tree",
    "lr": 0.01,
    "natural": True,
    "verbose": True,
    "verbose_eval": 1,
    "random_state": 1
}

b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
base_learner_choices = [b1, b2, b3]

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
    return df

def objective(trial, X, y):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1)
    n_estimators = trial.suggest_int("n_estimators", 500, 5000)
    minibatch_frac = trial.suggest_float("minibatch_frac", 0.1, 1.0)
    base_learner = trial.suggest_categorical('Base', [0,1,2])

    ngb = NGBRegressor(
        Base=base_learner_choices[base_learner],
        Dist=Normal,
        Score=LogScore,
        n_estimators=n_estimators,
        learning_rate=lr,
        natural_gradient=args["natural"],
        minibatch_frac=minibatch_frac,
        verbose=args["verbose"],
    )

    kf = KFold(n_splits=args["n_splits"], shuffle=True, random_state=args["random_state"])
    ngb_nll = []

    for train_index, test_index in kf.split(X):
        X_trainall, X_test = X[train_index], X[test_index]
        y_trainall, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)

        ngb.fit(X_train, y_train)
        y_preds = ngb.staged_predict(X_val)
        y_forecasts = ngb.staged_pred_dist(X_val)

        val_rmse = [mean_squared_error(y_pred, y_val) for y_pred in y_preds]
        val_nll = [-y_forecast.logpdf(y_val.flatten()).mean() for y_forecast in y_forecasts]
        best_itr = np.argmin(val_rmse) + 1

        ngb = NGBRegressor(
            Base=base_learner_choices[base_learner],
            Dist=Normal,
            Score=LogScore,
            n_estimators=n_estimators,
            learning_rate=lr,
            natural_gradient=args["natural"],
            minibatch_frac=minibatch_frac,
            verbose=args["verbose"],
        )
        ngb.fit(X_trainall, y_trainall)
        forecast = ngb.pred_dist(X_test, max_iter=best_itr)

        ngb_nll.append(-forecast.logpdf(y_test.flatten()).mean())

    return np.mean(ngb_nll)

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    # Encode categorical columns
    X = encode_categorical_columns(X)
    
    lss_rmse, lss_nll, times = [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X.iloc[train_indices], X.iloc[test_indices]
    y_train_opt, y_test_opt = y.iloc[train_indices], y.iloc[test_indices]
    
    if natural_flag:
        pset_path = f'logs/openml/ngboost/natural/exp/{dataset.name}_opt_params.json'
        # Check if the file exists
        if not os.path.isfile(pset_path):
            raise FileNotFoundError(f"The JSON file {pset_path} does not exist.")
        # Check if the file is empty
        if os.path.getsize(pset_path) == 0:
            raise ValueError(f"The JSON file {pset_path} is empty.")
        with open(pset_path) as pset:
            opt_params = json.load(pset)
    else:
        with open(f'logs/openml/ngboost/normal/exp/{dataset.name}_opt_params.json') as pset:
            opt_params = json.load(pset)

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        start_time = time.time()
        X_trainall, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_trainall, y_test = y.iloc[train_indices], y.iloc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)
        
        ngb = NGBRegressor(
            Base=base_learner_choices[opt_params["Base"]],
            Dist=Normal,
            Score=LogScore,
            n_estimators=opt_params["n_estimators"],
            learning_rate=opt_params["lr"],
            natural_gradient=args["natural"],
            minibatch_frac=opt_params["minibatch_frac"],
            verbose=args["verbose"],
        )
        
        ngb.fit(X_train, y_train)
        y_val_pred = ngb.predict(X_val)
        y_test_pred = ngb.predict(X_test)

        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time

        lss_rmse += [np.sqrt(mean_squared_error(y_test, y_test_pred))]
        val_rmse = [np.sqrt(mean_squared_error(y_val, y_val_pred))]
        lss_nll += [-norm(y_test_pred, 1).logpdf(y_test).mean()]
        samples = np.array([[np.random.normal(loc=loc, scale=1, size=100) for loc in y_test_pred]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test, samples)
        lss_crps += [crps_comps[0]]
        lss_crps_cal += [crps_comps[1]]
        lss_crps_sha += [crps_comps[2]]
        times += [elapsed_time]

        print(
                "[%d/%d] RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    fold,
                    n_folds,
                    np.sqrt(val_rmse[0]),
                    np.sqrt(mean_squared_error(y_test, y_test_pred)),
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time,
                )
            )

    print(task_id)
    print(dataset.name)
    print(
            "== RMSE NGBoost=%.4f ± %.4f, NLL NGBoost=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_cal =  %.4f +/- %.4f, CRPS_sha =  %.4f +/- %.4f,  TIME = %.4f"
            % (
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
                np.mean(times)  # Include elapsed time in the output
            )
        )
    return dataset.name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times)

if __name__ == "__main__":
    results = []
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    result = run_single_argument(task_number)
    results.append(result)
    if natural_flag:
        file = open("logs/openml_NGBoost_natural.csv", "a+")
    else:
        file = open("logs/openml_NGBoost_no_natural.csv", "a+")
    file.write(f"\n{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}, {result[6]}, {result[7]}, {result[8]}, {result[9]}, {result[10]}, {result[11]}")
    file.close()

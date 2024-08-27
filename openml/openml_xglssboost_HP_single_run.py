import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from scipy.stats import norm
from properscoring._mean_crps import _mean_crps_hersbach

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = '0fc137c28db32cdfecb6347178c7be68'

# Define constants and parameters
SUITE_ID = 336 # Regression on numerical features
np.random.seed(1)
mode = 'exp'
natural_flag = True
args = {
    "reps": 3,
    "n_est": 2000,
    "n_splits": 5,
    "score": "MLE",
    "distn": "Normal",
    "base": "tree",
    "verbose": True,
}

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

# Define your hyperparameter space
param_dict = {
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "min_child_weight": ["int", {"low": 1, "high": 100, "log": True}],
    "eta": ["float", {"low": 1e-5, "high": 0.4, "log": True}],
    "subsample": ["float", {"low": 0.5, "high": 1.0, "log": False}]
}

def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
    return df

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

    dtrain = xgb.DMatrix(X_train_opt, label=y_train_opt)

    start_time = time.time()  # Start time measurement

    xgblss = XGBoostLSS(Gaussian(stabilization="None", response_fn=mode, loss_fn="nll", natural_gradient=natural_flag))
    xgblss.start_values = np.array([np.array(0.5) for _ in range(xgblss.dist.n_dist_param)])

    np.random.seed(123)
    opt_param = xgblss.hyper_opt(param_dict, dtrain, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=20, max_minutes=3000, n_trials=80,
                                    silence=True, seed=1, hp_seed=1)

    end_time = time.time()  # End time measurement
    elapsed_time_HP = end_time - start_time  # Calculate elapsed time

    print(opt_param)
    opt_params = opt_param.copy()
    if natural_flag:
        with open(f'logs/openml/xgboost/natural/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    else:
        with open(f'logs/openml/xgboost/normal/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    
    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_trainall, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_trainall, y_test = y.iloc[train_indices], y.iloc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        evals_result = {}
        opt_params['early_stopping'] = 20

        gbm = xgblss.train(opt_params, dtrain, 
                            num_boost_round=n_rounds,
                            evals=[(dtrain, 'train'), (deval, 'eval')],
                            evals_result=evals_result,
                            early_stopping_rounds=20
                            )

        full_train_data = xgb.DMatrix(X_trainall, label=y_trainall)
        opt_params['early_stopping'] = None
        best_iter = xgblss.booster.best_iteration

        best_iter = xgblss.booster.best_iteration
        runtime_start = time.time()
        final_gbm = xgblss.train(opt_params, full_train_data, 
                            num_boost_round=best_iter,
                        )

        forecast = xgblss.predict(dtest)
        runtime_pred = time.time() - runtime_start
        forecast_val = xgblss.predict(deval)

        lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'], y_test))]
        val_rmse = [np.sqrt(mean_squared_error(forecast_val['loc'], y_val))]
        lss_nll += [-norm(forecast['loc'], forecast['scale']).logpdf(y_test).mean()]
        samples = np.array([[np.random.normal(loc=loc, scale=scale, size=100) for loc, scale in zip(forecast['loc'], forecast['scale'])]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test, samples)
        lss_crps += [crps_comps[0]]
        lss_crps_cal += [crps_comps[1]]
        lss_crps_sha += [crps_comps[2]]
        times += [runtime_pred]

        print(
                "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    fold,
                    n_folds,
                    best_iter,
                    np.sqrt(val_rmse),
                    np.sqrt(mean_squared_error(forecast['loc'], y_test)),
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time_HP,
                )
            )

    print(task_id)
    print(dataset.name)
    print(
            "== RMSE XGBoostLSS=%.4f ± %.4f, NLL XGBoostLSS=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_cal =  %.4f +/- %.4f, CRPS_sha =  %.4f +/- %.4f,  TIME = %.4f"
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
    return dataset.name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), elapsed_time_HP

if __name__ == "__main__":
    print("XGBOOSTLSS")
    print("______________________")
    results = []
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    result = run_single_argument(task_number)
    results.append(result)
    if natural_flag:
        file = open("logs/openml_XGBoostLSS_natural.csv", "a+")
    else:
        file = open("logs/openml_XGBoostLSS_no_natural.csv", "a+")
    file.write(f"\n{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}, {result[6]}, {result[7]}, {result[8]}, {result[9]}, {result[10]}, {result[11]}, {result[12]}")
    file.close()

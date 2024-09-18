import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from scipy.stats import norm
from properscoring._mean_crps import _mean_crps_hersbach

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss

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
    "n_est": 5,
    "n_splits": 20,
    "score": "MLE",
    "distn": "Normal",
    "base": "tree",
    "verbose": True,
}

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    lss_rmse, lss_nll, times = [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    start_time = time.time()  # Start time measurement

    lgblss = LightGBMLSS(Gaussian(stabilization="None", response_fn=mode, loss_fn="nll", natural_gradient=natural_flag))
    lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])

    default_params = {
        "eta": 0.01,
        "max_depth": 6,
        "num_leaves": 83,
        "min_data_in_leaf": 23,
    }

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    for fold in range(n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_trainall, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_trainall, y_test = y.iloc[train_indices], y.iloc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)

        dtrain = lgb.Dataset(X_train, y_train)
        deval = lgb.Dataset(X_val, y_val)
        dtest = lgb.Dataset(X_test, y_test)
        # Training with early stopping
        evals_result = {}
        default_params['early_stopping'] = 20
        # Train Model with optimized hyperparameters
        gbm = lgblss.train(default_params, dtrain, 
                            num_boost_round=args["n_est"],
                            valid_sets=[dtrain, deval]
                            )

        # Best iteration
        print(f"Best iteration: {lgblss.booster.best_iteration}")

        full_train_data = lgb.Dataset(X_trainall, y_trainall)
        default_params['early_stopping'] = None

        final_gbm = lgblss.train(default_params, full_train_data, 
                            num_boost_round=lgblss.booster.best_iteration,
                        )
        # the final prediction for this fold
        forecast = lgblss.predict(X_test)
        forecast_val = lgblss.predict(X_val)

        # After processing all folds for a dataset:
        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time

        lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'].values, y_test))]
        val_rmse = [np.sqrt(mean_squared_error(forecast_val['loc'].values, y_val))]
        lss_nll += [-norm(forecast['loc'], forecast['scale']).logpdf(y_test).mean()]
        samples = np.array([[np.random.normal(loc=loc, scale=scale, size=100) for loc, scale in zip(forecast['loc'], forecast['scale'])]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test, samples)
        lss_crps += [crps_comps[0]]
        lss_crps_cal += [crps_comps[1]]
        lss_crps_sha += [crps_comps[2]]
        times += [elapsed_time]

        # Define the quantiles to evaluate
        quantiles = [0.1, 0.5, 0.9]

        # Compute the quantiles for each observation
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[q] = norm.ppf(q, loc=forecast['loc'], scale=forecast['scale'])
            q_loss = quantile_loss(q, y_test, quantile_preds[q]).mean()
            quantile_losses.append(q_loss)
        
        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]

        print(
                "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f WQL_01=%.4f WQL_05=%.4f WQL_09=%.4f WQL_AVG=%.4f"
                % (
                    fold + 1,
                    n_folds,
                    lgblss.booster.best_iteration,
                    np.sqrt(val_rmse),
                    np.sqrt(mean_squared_error(forecast['loc'].values, y_test)),
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time,
                    wql_01[-1],
                    wql_05[-1],
                    wql_09[-1],
                    wql_avg[-1]
                )
            )
    print(task_id)
    print(dataset.name)
    print(
            "== RMSE GBMLSS=%.4f ± %.4f, NLL GBMLSS=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_cal =  %.4f +/- %.4f, CRPS_sha =  %.4f +/- %.4f,  TIME = %.4f, "
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
                np.mean(times),  # Include elapsed time in the output,
            )
        )
    # return a dictonary of val
    return dataset.name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 

if __name__ == "__main__":
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    results = run_single_argument(task_number)
    print(results)
    #results.append(result)
    # if natural_flag:
    #     file = open("logs/openml_LSSboost_natural.csv", "a+")
    # else:
    #     file = open("logs/openml_LSSboost_no_natural.csv", "a+")
    # file.write(f"\n{result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}, {result[5]}, {result[6]}, {result[7]}, {result[8]}, {result[9]}, {result[10]}, {result[11]}")
    # file.close()

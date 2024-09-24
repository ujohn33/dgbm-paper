import os
import sys
import openml
import json
import numpy as np
import pandas as pd
import time
import csv
from scipy.stats import norm 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import Bernoulli, Normal
from ngboost.scores import LogScore
from ngboost import NGBRegressor
from ngboost.learners import default_linear_learner, default_tree_learner
import optuna

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = '0fc137c28db32cdfecb6347178c7be68'

# Random seed
np.random.seed(1)

# Define constants and parameters
args = {
    "n_splits": 5,
    "score": "LogScore",
    "distn": "Normal",
    "SUITE_ID": 336,  # Regression on numerical features 
    "natural_grad": True,
    "verbose": True,
    "random_state": 1
}

b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
base_learner_choices = [b1, b2, b3]

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(args['SUITE_ID'])  # obtain the benchmark suite

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
        natural_gradient=args["natural_grad"],
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
            natural_gradient=args["natural_grad"],
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

    # Switch to numpy
    X, y = X.to_numpy(), y.to_numpy()
    
    lss_rmse, lss_nll, times = [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X[train_indices], X[test_indices]
    y_train_opt, y_test_opt = y[train_indices], y[test_indices]

    start_time_HP = time.time()
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train_opt, y_train_opt), n_trials=20, timeout=3000*60)
    elapsed_time_HP = time.time() - start_time_HP

    print("Best hyperparameters: ", study.best_params)

    opt_params = study.best_params
    
    if args['natural_grad']:
        with open(f'logs/openml/ngboost/natural/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    else:
        with open(f'logs/openml/ngboost/normal/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_trainall, X_test = X[train_indices], X[test_indices]
        y_trainall, y_test = y[train_indices], y[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)
        
        ngb = NGBRegressor(
            Base=base_learner_choices[opt_params["Base"]],
            Dist=Normal,
            Score=LogScore,
            n_estimators=opt_params["n_estimators"],
            learning_rate=opt_params["lr"],
            natural_gradient=args["natural_grad"],
            minibatch_frac=opt_params["minibatch_frac"],
            verbose=args["verbose"],
        )
        
        ngb.fit(X_train, y_train)

        # pick the best iteration on the validation set
        y_preds = ngb.staged_predict(X_val)
        y_forecasts = ngb.staged_pred_dist(X_val)

        val_rmse = [mean_squared_error(y_pred, y_val) for y_pred in y_preds]
        val_nll = [
            -y_forecast.logpdf(y_val.flatten()).mean() for y_forecast in y_forecasts
        ]
        best_itr = np.argmin(val_rmse) + 1

        start_time = time.time()
        ngb = NGBRegressor(
            Base=base_learner_choices[opt_params["Base"]],
            Dist=Normal,
            Score=LogScore,
            n_estimators=opt_params["n_estimators"],
            learning_rate=opt_params["lr"],
            natural_gradient=args["natural_grad"],
            minibatch_frac=opt_params["minibatch_frac"],
            verbose=args["verbose"],
        )

        ngb.fit(X_trainall, y_trainall)

        # the final prediction for this fold
        forecast = ngb.pred_dist(X_test, max_iter=best_itr)
        forecast_val = ngb.pred_dist(X_val, max_iter=best_itr)

        # After processing all folds for a dataset:
        runtime_pred = time.time() - start_time  # Calculate elapsed time

        # set the appropriate scale if using a homoskedastic Normal
        if args["distn"] == "NormalFixedVar":
            scale = (
                forecast.var * ((forecast_val.loc - y_val.flatten()) ** 2).mean() ** 0.5
            )
            forecast = norm(loc=forecast.loc, scale=scale)

        lss_rmse += [np.sqrt(mean_squared_error(forecast.mean(), y_test))]
        lss_nll += [-forecast.logpdf(y_test.flatten()).mean()]
        samples = np.array([[np.random.normal(loc=loc, scale=scale, size=100) for loc, scale in zip(forecast.loc, forecast.scale)]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test.flatten(), samples)
        lss_crps += [crps_comps[0]]
        lss_crps_cal += [crps_comps[1]]
        lss_crps_sha += [crps_comps[2]]
        times += [runtime_pred]

        # Define the quantiles to evaluate
        quantiles = [0.1, 0.5, 0.9]

        # Compute the quantiles for each observation
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = norm.ppf(q, loc=forecast['loc'], scale=forecast['scale'])
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)
        
        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]        

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
                    elapsed_time_HP,
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
    return dataset.name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), elapsed_time_HP, np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 

if __name__ == "__main__":
    print("NGBOOST")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    results = run_single_argument(task_number)
    if args['natural_grad']:
        file_path = "logs/openml/openml_NGBoost_natural.csv"
    else:
        file_path = "logs/openml/openml_NGBoost_no_natural.csv"
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


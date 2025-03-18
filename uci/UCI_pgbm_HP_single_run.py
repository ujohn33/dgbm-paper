import openml
import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import time
import torch
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
from utils.logging import log_predictions

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

dataset_list = ["Boston Housing", "Concrete Compression Strength", "Energy Efficiency", "Kin8nm", "Naval Propulsion", "Combined Cycle Power Plant", "Protein Structure", "Wine Quality Red", "Yacht Hydrodynamics", "Year Prediciton MSD"]
n_forecasts = 200
method_name = "pgbm"

# Hardcoded parameters for testing
args = {
    "dataset": "Concrete Compression Strength",
    "n_splits": 20,
    "distn": "Normal",
    "verbose": True,
    "verbose_eval":1,
    "random_state":1
}

def objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()
    return loss

# Define the Optuna objective class for hyperparameter tuning
class Objective(object):
    def __init__(self, X_train, y_train, dataset_name):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_name = dataset_name
        
    def __call__(self, trial):
        try:
            # Set bagging_fraction based on dataset
            if self.dataset_name == "Year Prediciton MSD":
                bagging_fraction = 0.1
            else:
                bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.5, 1.0)
                
            params = {
                'n_estimators': 2000,
                'bagging_fraction': bagging_fraction,
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
                'max_leaves': trial.suggest_int('max_leaves', 8, 32),
                'max_bin': trial.suggest_int('max_bin', 32, 128),
                'max_depth': -1,
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),  # Constant for this example
                'device': 'gpu',
                'verbose': 2,
                'feature_fraction':  1,
                'derivatives': 'exact',
                'distribution': 'normal',
            }
            model = PGBMRegressor()
            model.set_params(**params)
            score = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=5, n_jobs=5, scoring='neg_root_mean_squared_error', error_score="raise"))
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

def run_single_argument(run_seed):
    dset = dataset_list[int(run_seed)]
    print(dset)
    args["dataset"] = dset
    y_true, lss_rmse, lss_nll, times, times_HP = [], [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []    

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={args['dataset']} X.shape={str(X.shape)} {args['distn']}")
    lgbm_rmse = []
    if args["dataset"] == "Year Prediciton MSD":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
    elif args["dataset"] == "Protein Structure":
        kf = KFold(n_splits=5)
        folds = kf.split(X)
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(1)
        folds = []
        for i in range(5):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))        
    else:
        kf = KFold(n_splits=args["n_splits"])
        folds = kf.split(X)
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(1)
        folds = []
        for i in range(args['n_splits']):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n
            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))


    for itr, (train_index, test_index) in enumerate(folds):
        print(f'{dset}: fold {itr + 1}/{len(folds)}')
        #X_train, X_test, y_train, y_test = get_fold(dataset_name, data, fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2)

        train_data = (X_train, y_train)
        train_val_data = (X_train_val, y_train_val)
        valid_data = (X_val, y_val)

        print(f'The Input data shape is {X_train.shape}')
        assert not np.any(np.isnan(X_train)), "NaN values found in X_train"
        assert not np.any(np.isnan(y_train)), "NaN values found in y_train"
        assert not np.any(np.isinf(X_train)), "Infinity values found in X_train"

        # Hyperparameter optimization with Optuna
        start_time = time.time()
        print('Hyperparameter tuning...')
        study = optuna.create_study(direction='maximize')
        objective_tuning = Objective(X_train, y_train, args['dataset'])
        time_limit = 86400/len(folds)
        study.optimize(objective_tuning, n_trials=20)
        end_time = time.time()  # End time measurement
        elapsed_time_HP = end_time - start_time  # Calculate elapsed time

        # Set the best parameters and number of estimators from hyperparameter tuning
        best_params = study.best_params
        print(f'Best hyperparameters for fold {itr + 1}: {best_params}')

        # Train the final model on the full training set (including validation)
        print('Training validation model...')
        model = PGBM()
        model.train(train_val_data, objective=objective, metric=rmseloss_metric, valid_set=(X_val, y_val), params=best_params)
        best_params['n_estimators'] = model.best_iteration

        print('Training final model...')
        start_time = time.time()
        model = PGBM()
        model.train(train_data,  objective=objective, metric=rmseloss_metric, params=best_params)
        training_time = time.time() - start_time
        print(f'Training time for fold {itr + 1}: {training_time:.2f} seconds')

        # Make predictions
        print('Prediction...')
        yhat_point = model.predict(X_test)
        yhat_dist, mu, var = model.predict_dist(X_test, n_forecasts=n_forecasts, parallel=False, output_sample_statistics=True)
        std = np.sqrt(var.cpu().numpy())

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(yhat_point.cpu().numpy(), y_test))
        nll_test = -norm(mu.cpu().numpy(), std).logpdf(y_test.flatten()).mean()
    
        yhat_dist = yhat_dist.reshape(yhat_dist.shape[1], yhat_dist.shape[0])
        crps_comps = crps(y_test.flatten(), yhat_dist.cpu().numpy())
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
        log_predictions(itr, dset, y_test, mu, std, quantile_preds, f"logs/uci/predictions/{method_name}.csv")

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]

    print(f'Completed dataset: {dset}')
    # return a dictonary of val
    return  dset, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP), np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 



if __name__ == "__main__":
    print("PGBM")
    print("______________________")
    vsc_data = os.environ['VSC_DATA']
    results = run_single_argument(sys.argv[1])
    file_path = f"results/uci/uci_{method_name}.csv"
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
import openml
import os
import sys
import json
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
from properscoring._mean_crps import _mean_crps_hersbach

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps

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
n_forecasts = 1000

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
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def __call__(self, trial):
        params = {
            'n_estimators': 200,
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.4),
            'max_leaves': trial.suggest_int('max_leaves', 20, 200),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # Constant for this example
            'n_estimators': trial.suggest_int('n_estimators', 20, 200),
            'device': 'gpu'
        }
        model = PGBMRegressor()
        model.set_params(**params)
        score = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=5, n_jobs=5, scoring='neg_root_mean_squared_error'))
        return score

def run_single_argument(run_seed):
    dset = dataset_list[int(run_seed)]
    args["dataset"] = dset
    y_true, lss_rmse, lss_nll, times, times_HP = [], [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []

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
        # default_params = {
        #     "max_depth":                9,
        #     "num_leaves":               110,
        #     "min_data_in_leaf":         22,
        #     "subsample":                1,
        # }
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

        # Hyperparameter optimization with Optuna
        start_time = time.time()
        print('Hyperparameter tuning...')
        study = optuna.create_study(direction='maximize')
        objective_tuning = Objective(X_train, y_train)
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

        # Compute metrics
        rmse = rmseloss_metric(yhat_point.cpu(), y_test).numpy()
        crps_test = model.crps_ensemble(yhat_dist, y_test).mean().numpy()
        nll_test = -norm(mu, var).logpdf(y_test.flatten()).mean()
    
        yhat_dist = yhat_dist.reshape(yhat_dist.shape[1], yhat_dist.shape[0])
        crps_comps = crps(y_test.flatten(), yhat_dist)
        crps_cal, crps_sha = crps_comps[1], crps_comps[2]

        # Store results
        lss_rmse.append(rmse)
        lss_nll.append(nll_test)
        lss_crps.append(crps_test)
        lss_crps_cal.append(crps_cal)
        lss_crps_sha.append(crps_sha)
        times += [training_time]
        times_HP += [elapsed_time_HP]

    print(f'Completed dataset: {dset}')
    # return a dictonary of val
    return  dset, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP)



if __name__ == "__main__":
    vsc_data = os.environ['VSC_DATA']
    results = run_single_argument(sys.argv[1])
    file = open("logs/uci/PGBM_no_natural.csv", "a+")
    file.write(f"\n{results[0]}, {results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}, {results[6]}, {results[7]}, {results[8]}, {results[9]}, {results[10]}, {results[11]}")
    file.close()


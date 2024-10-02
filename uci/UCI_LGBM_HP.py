import openml
import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import time
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from optuna.integration import OptunaSearchCV
from sklearn.metrics import mean_pinball_loss

from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss

np.random.seed(123)

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

# Hardcoded parameters for testing
args = {
    "quantiles": [0.1, 0.5, 0.9],
    "n_est": 11,
    "n_trials": 100,
    "n_splits": 2000,
}

# Define hyperparameter search space
param_distributions = {
    "eta": optuna.distributions.FloatDistribution(1e-5, 0.4, log=True),
    "max_depth": optuna.distributions.IntDistribution(2, 10),
    "num_leaves": optuna.distributions.IntDistribution(20, 200),
    "min_data_in_leaf": optuna.distributions.IntDistribution(20, 100),
    "bagging_fraction": optuna.distributions.FloatDistribution(0.5, 1, log=False),
    "feature_pre_filter": optuna.distributions.CategoricalDistribution([False])
}


def run_single_arguement(run_seed):
    dset = dataset_list[int(run_seed)]
    times, times_HP = [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[dset]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={dset} X.shape={str(X.shape)}")
    if dset == "Year Prediciton MSD":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
    elif dset == "Protein Structure":
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
        # print('train_index: ')
        # print(train_index)
        # print('test_index: ')
        # print(test_index)
        start_time = time.time()  # Start time measurement
        X_trainall, X_test = X[train_index], X[test_index]
        y_trainall, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainall, y_trainall, test_size=0.2
        )

        full_train_data = lgb.Dataset(X_trainall, y_trainall)

        start_time = time.time()

        # Define the LightGBM regressor
        estimator = lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=1, n_estimators=args["n_est"])

        # Define cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=1)

        # Use OptunaSearchCV for hyperparameter optimization
        optuna_search = OptunaSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            cv=cv,
            n_trials=args['n_trials'],
            refit=True,
            random_state=1,
            verbose=1
        )

        optuna_search.fit(X_trainall, y_trainall)
        
        elapsed_time_HP = time.time() - start_time

        # Get the best parameters
        best_params = optuna_search.best_params_
        print(f"Best parameters: {best_params}")

        dtrain = lgb.Dataset(X_train, y_train)
        deval = lgb.Dataset(X_val, y_val)
        dtest = lgb.Dataset(X_test, y_test)

        quantile_models = {}
        quantile_preds = {}
        quantile_losses = []

        runtime_start = time.time()

        # Train separate models for each quantile (0.1, 0.5, 0.9)
        for count, q in enumerate(args["quantiles"]):
            # Train Model with optimized hyperparameters
            model = lgb.train(params=best_params, train_set=dtrain, valid_sets=[deval], num_boost_round=args['n_est'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),
                ])
            best_iter = model.best_iteration
            # Best iteration
            print(f"Best iteration: {best_iter}")
            final_model = lgb.train(params=best_params, train_set=dtrain, valid_sets=[deval], num_boost_round=best_iter)
            quantile_models[q] = model

        runtime_pred = time.time() - runtime_start

        for q in args["quantiles"]:
            q_pred = quantile_models[q].predict(X_test)
            q_loss = quantile_loss(q, y_test, q_pred).mean()
            if q == 0.1:
                quantile_losses.append(q_loss)
            elif q == 0.5:
                quantile_losses.append(q_loss)
            else:
                quantile_losses.append(q_loss)

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]
        times += [runtime_pred]

    print(run_seed)
    print(dset)
    # return a dictonary of val
    return  dset, np.mean(times), elapsed_time_HP, np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 

if __name__ == "__main__":
    print("LGBM quantiles")
    print("______________________")
    results = run_single_arguement(sys.argv[1])
    file_path = f"logs/uci/uci_LGBM.csv"
    header = ["dset","time_run","time_HP","WQL01-mean", "WQL01-std","WQL05-mean", "WQL05-std","WQL09-mean", "WQL09-std", "WQL_avg-mean", "WQL_avg-std"]
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
                        results[5], results[6], results[7], results[8], results[9]]

        writer.writerow(row_to_write)
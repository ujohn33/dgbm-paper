import os
import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import time
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from ucimlrepo import fetch_ucirepo 
from scipy.io import arff
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from scipy.stats import norm

np.random.seed(1)
mode = 'softplus'

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
    "dataset": "Concrete Compression Strength",
    "reps": 3,
    "n_est": 2000,
    "n_splits": 20,
    "score": "MLE",
    "distn": "Normal",
    "base": "tree",
    "verbose": True,
}


def run_single_arguement(run_seed):
    dset = dataset_list[int(run_seed)]
    args["dataset"] = dset
    y_true, lss_rmse, lss_nll, times = [], [], [], []

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={args['dataset']} X.shape={str(X.shape)} {args['score']}/{args['distn']}")
    lgbm_rmse = []
    with open(f'logs/{mode}/{args["dataset"]}_opt_params.json') as pset:
        default_params = json.load(pset)

    if args["dataset"] == "Year Prediciton MSD":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
        default_params = {
            "eta":                     0.1,
            "max_depth":                7,
            "num_leaves":               92,
            "min_data_in_leaf":         60,
            "subsample":                0.1,
        }
    elif args["dataset"] == "Protein Structure":
        default_params = {
            "eta":                     0.03,
            "max_depth":                7,
            "num_leaves":               92,
            "min_data_in_leaf":         60,
            "subsample":                1,
        }
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
        default_params = {
            "eta":                     0.03,
            "max_depth":                9,
            "num_leaves":               110,
            "min_data_in_leaf":         22,
            "subsample":                1,
        }
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

        y_true += list(y_test.flatten())


        lgblss = LightGBMLSS(
            Gaussian(stabilization="None",
                    response_fn = mode,
                    loss_fn = "nll")
        )
        # Modify start values     
        lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])

        dtrain = lgb.Dataset(X_train, y_train)
        deval = lgb.Dataset(X_val, y_val)
        dtest = lgb.Dataset(X_test, y_test)
        # Training with early stopping
        evals_result = {}
        default_params['early_stopping'] = 20
        # Train Model with optimized hyperparameters
        gbm = lgblss.train(default_params, dtrain, 
                            num_boost_round = args["n_est"],
                            valid_sets = [dtrain, deval]
                            )

        # Best iteration
        print(f"Best iteration: {lgblss.booster.best_iteration}")

        full_train_data = lgb.Dataset(X_trainall, y_trainall)
        default_params['early_stopping'] = None

        final_gbm = lgblss.train(default_params, full_train_data, 
                            num_boost_round = lgblss.booster.best_iteration,
                        )
        # the final prediction for this fold
        forecast = lgblss.predict(X_test)
        forecast_val = lgblss.predict(X_val)

        # After processing all folds for a dataset:
        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time

        lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'].values, y_test))]
        val_rmse = [np.sqrt(mean_squared_error(forecast_val['loc'].values, y_val))]
        lss_nll += [-norm(forecast['loc'], forecast['scale']).logpdf(y_test.flatten()).mean()]
        times += [elapsed_time]

        print(
                "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f"
                % (
                    itr + 1,
                    args['n_splits'],
                    lgblss.booster.best_iteration,
                    np.sqrt(val_rmse),
                    np.sqrt(mean_squared_error(forecast['loc'].values, y_test)),
                    lss_nll[-1],
                )
            )
    print(dset)
    print(
            "== GBM=%.4f +/- %.4f, RMSE GBMLSS=%.4f ± %.4f, NLL GBMLSS=%.4f ± %.4f, TIME = %.4f"
            % (
                0.0,
                0.0,
                np.mean(lss_rmse),
                np.std(lss_rmse),
                np.mean(lss_nll),
                np.std(lss_nll),
                np.mean(times)  # Include elapsed time in the output
            )
        )
    # return a dictonary of val
    return  dset, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(times)

if __name__ == "__main__":
    vsc_data = os.environ['VSC_DATA']
    results = run_single_arguement(sys.argv[1])
    file = open("logs/LSSboost_logloss.csv", "a+")
    file.write(f"\n{results[0]}, {results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}")
    file.close()
   
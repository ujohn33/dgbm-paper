import os
import sys
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from ucimlrepo import fetch_ucirepo 
from scipy.io import arff
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from scipy.stats import norm

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
    "dataset": "Concrete Compression Strength",
    "reps": 3,
    "n_est": 2000,
    "n_splits": 20,
    "score": "MLE",
    "distn": "Normal",
    "base": "tree",
    "verbose": True,
}

# Define your hyperparameter space
param_dict = {
    "eta": ["float", {"low": 1e-3, "high": 1e-1, "log": True}],
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "num_leaves": ["int", {"low": 20, "high": 200, "log": False}],  # Constant for this example
    "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "feature_pre_filter": ["categorical", [False]]
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

    start_time = time.time()  # Start time measurement

    lgblss = LightGBMLSS(Gaussian(stabilization="None", response_fn="exp", loss_fn="nll"))
    # Modify start values     
    lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])
    
    # Perform hyperparameter optimization
    np.random.seed(123)
    dtrain = lgb.Dataset(X, y)
    if args["dataset"] == "Year Prediciton MSD":
        args['n_splits'] = 2
        param_dict["subsample"] = ["categorical", [0.1]]
    elif ["dataset"] == "Protein Structure":
        args['n_splits'] = 5
    else:
        pass 
    opt_param = lgblss.hyper_opt(param_dict, dtrain, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=20, max_minutes=3000, n_trials=80,
                                    silence=True, seed=1, hp_seed=1)

    print(opt_param)
    opt_params = opt_param.copy()
    # Assuming opt_params is your dictionary of optimized parameters
    with open('logs/{}_opt_params.json'.format(dset), 'w') as f:
        json.dump(opt_params, f)
    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]
    # # Use optimized parameters to train your model
    # # Note: You might need to adjust the following line to use the `opt_param` properly, depending on how `lgblss.hyper_opt` returns the optimized parameters.
    # final_gbm = lgblss.train(opt_param, dtrain, num_boost_round=n_rounds)
    
    # # the final prediction for this fold
    # forecast = lgblss.predict(X_test)
    # #forecast_val = lgblss.predict(X_val)

    # # After processing all folds for a dataset:
    # end_time = time.time()  # End time measurement
    # elapsed_time = end_time - start_time  # Calculate elapsed time

    # lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'].values, y_test))]
    # #val_rmse = [np.sqrt(mean_squared_error(forecast_val['loc'].values, y_val))]
    # lss_nll += [-norm(forecast['loc'], forecast['scale']).logpdf(y_test.flatten()).mean()]
    # times += [elapsed_time]

    # return 

if __name__ == "__main__":
    #vsc_data = os.environ['VSC_DATA']
    results = run_single_arguement(sys.argv[1])
    # file = open("logs/LSSboost_logloss.csv", "a+")
    # file.write(f"\n{results[0]}, {results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}")
    # file.close()

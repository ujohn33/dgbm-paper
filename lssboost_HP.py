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

np.random.seed(1)

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

dataset_list = ["Boston Housing", "Concrete Compression Strength", "Energy Efficiency", "Kin8nm", "Naval Propulsion", "Combined Cycle Power Plant", "Protein Structure", "Wine Quality Red", "Yacht Hydrodynamics"]

def objective(trial):
    # select hyperparameters
    # param_dict = {
    #     "eta": ["float", {"low": 1e-3, "high": 1e-1, "log": True}],
    #     "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    #     "num_leaves": ["int", {"low": 20, "high": 200, "log": False}],  # Constant for this example
    #     "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    #     "feature_pre_filter": ["categorical", [False]]
    # }
    eta = trial.suggest_float('eta', 1e-3, 1e-1, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_leaves = trial.suggest_int('num_leaves', 20, 200)
    min_d_leaf = trial.suggest_int('min_data_in_leaf', 20, 100)

    # Initialize metrics
    all_rmse = []
    all_nll = []

    # Loop over each dataset
    for dataset_name in dataset_list:
        data = dataset_name_to_loader[dataset_name]()
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        if dataset_name == "Protein Structure":
            n_splits_num = 5
        else:
            n_splits_num = 20
        kf = KFold(n_splits=n_splits_num, shuffle=True, random_state=1)
        dataset_rmse = []
        dataset_nll = []
        for train_index, test_index in kf.split(X):     
            X_train, X_val, y_train, y_val = train_test_split(X[train_index], y[train_index], test_size=0.2, random_state=1)

            # Set up the model
            lgblss = LightGBMLSS(
                Gaussian(stabilization="None", response_fn="exp", loss_fn="nll")
            )
            # Modify start values     
            lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])
            
            dtrain = lgb.Dataset(X_train, y_train)
            deval = lgb.Dataset(X_val, y_val)

            # Train the model with the suggested parameters
            lgblss.train({
                'eta': eta,
                'max_depth': max_depth,
                'num_leaves': n_leaves, 
                'min_data_in_leaf': min_d_leaf,
                'feature_pre_filer': False,
                'early_stopping': 20
            }, dtrain, num_boost_round=2000, valid_sets=[dtrain, deval])

            # Evaluate the model
            forecast = lgblss.predict(X[test_index])
            rmse = np.sqrt(mean_squared_error(forecast['loc'].values, y[test_index]))
            nll = -norm(forecast['loc'], forecast['scale']).logpdf(y[test_index].flatten()).mean()

            dataset_rmse.append(rmse)
            dataset_nll.append(nll)

        print(f"\tAverage RMSE for {dataset_name}: {np.mean(dataset_rmse)}")
        print(f"\tAverage NLL for {dataset_name}: {np.mean(dataset_nll)}")

        all_rmse.append(np.mean(dataset_rmse))
        all_nll.append(np.mean(dataset_nll))

    # Average performance across datasets
    overall_rmse = np.mean(all_rmse)
    overall_nll = np.mean(all_nll)

    # Objective is to minimize RMSE and NLL
    return overall_rmse 

# Create an Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# best RMSE
print(f"\tBest value (rmse): {study.best_value:.5f}")

print(f"\tBest params:")
for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")
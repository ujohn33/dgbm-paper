import os
import sys
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import time
import csv
from scipy.stats import norm as norm_dist
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

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
}

dataset_list = ["Boston Housing", "Concrete Compression Strength", "Energy Efficiency", "Kin8nm", "Naval Propulsion", "Combined Cycle Power Plant", "Protein Structure", "Wine Quality Red", "Yacht Hydrodynamics", "Year Prediciton MSD"]

args = {
    "dataset": "Concrete Compression Strength",
    "n_est": 2000,
    "n_splits": 20,
    "distn": "Normal",
    "lr": 0.01,
    "natural": True,
    "score": "LogScore",
    "base": "tree",
    "minibatch_frac": 1.0,
    "verbose": True,
    "verbose_eval":1,
    "random_state":1
}
 
b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
base_learner_choices = [b1, b2, b3]

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1)
    n_estimators = trial.suggest_int("n_estimators", 500, 5000)
    minibatch_frac = trial.suggest_float("minibatch_frac", 0.1, 1.0)
    base_learner = trial.suggest_categorical('Base', [0,1,2])

    args["lr"] = lr
    args["n_est"] = n_estimators
    args["minibatch_frac"] = minibatch_frac
    args["base"] = base_learner_choices[base_learner]

    dset = args["dataset"]
    data = dataset_name_to_loader[dset]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    kf = KFold(n_splits=args["n_splits"])
    folds = kf.split(X)

    ngb_nll = []

    for train_index, test_index in folds:
        X_trainall, X_test = X[train_index], X[test_index]
        y_trainall, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)

        ngb = NGBRegressor(
            Base=args["base"],
            Dist=eval(args["distn"]),
            Score=eval(args["score"]),
            n_estimators=args["n_est"],
            learning_rate=args["lr"],
            natural_gradient=args["natural"],
            minibatch_frac=args["minibatch_frac"],
            verbose=args["verbose"],
        )

        ngb.fit(X_train, y_train)

        y_preds = ngb.staged_predict(X_val)
        y_forecasts = ngb.staged_pred_dist(X_val)

        val_rmse = [mean_squared_error(y_pred, y_val) for y_pred in y_preds]
        val_nll = [
            -y_forecast.logpdf(y_val.flatten()).mean() for y_forecast in y_forecasts
        ]
        best_itr = np.argmin(val_rmse) + 1

        ngb = NGBRegressor(
            Base=args["base"],
            Dist=eval(args["distn"]),
            Score=eval(args["score"]),
            n_estimators=args["n_est"],
            learning_rate=args["lr"],
            natural_gradient=args["natural"],
            minibatch_frac=args["minibatch_frac"],
            verbose=args["verbose"],
        )
        ngb.fit(X_trainall, y_trainall)
        forecast = ngb.pred_dist(X_test, max_iter=best_itr)

        ngb_nll.append(-forecast.logpdf(y_test.flatten()).mean())

    return np.mean(ngb_nll)


def run_single_arguement(run_seed):
    dset = dataset_list[int(run_seed)]
    args["dataset"] = dset
    y_true, lss_rmse, lss_nll, times = [], [], [], []

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={args['dataset']} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20, timeout=86400)

    print("Best hyperparameters: ", study.best_params)

    opt_params = study.best_params
    # Assuming opt_params is your dictionary of optimized parameters
    with open('logs/ngboost/natural/exp/{}_opt_params.json'.format(dset), 'w') as f:
        json.dump(opt_params, f)
    #n_rounds = opt_params["opt_rounds"]
    #del opt_params["opt_rounds"]

if __name__ == "__main__":
    results = run_single_arguement(sys.argv[1])
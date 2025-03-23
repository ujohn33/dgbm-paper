import os
import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import time
import csv
from scipy.stats import norm as norm_dist
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from ngboost.distns import Bernoulli, Normal
from ngboost.scores import LogScore, MLE
from ngboost import NGBRegressor
from ngboost.learners import default_linear_learner, default_tree_learner

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
    "score": "MLE",
    "base": "tree",
    "minibatch_frac": 1.0,
    "verbose": True,
    'random_state': 1,
    "seed": 1,
}


def run_single_arguement(run_seed):
    dset = dataset_list[int(run_seed)]
    start_time = time.time()  # Start time measurement
    args["dataset"] = dset
    y_true, ngb_rmse, ngb_nll, times = [], [], [], []
    ngb_crps, ngb_crps_cal, ngb_crps_sha = [], [], []

    # Load dataset -- use last column as label
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={args['dataset']} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    lgbm_rmse = []
    if args["dataset"] == "Year Prediciton MSD":
        args["lr"] = 0.1
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
        args["minibatch_frac"] = 0.1 
    elif args["dataset"] == "Protein Structure":
        args["lr"] = 0.01
        args["minibatch_frac"] = 1.0
        # kf = KFold(n_splits=5, random_state=args["seed"])
        # folds = kf.split(X)
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(args["random_state"])
        folds = []
        for i in range(5):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))        
    else:
        if args["dataset"] == "Concrete Compression Strength":
            args["lr"] = 0.002
            args["n_est"] = 5000
        elif args["dataset"] == "Energy Efficiency":
            args["lr"] = 0.002
            args["n_est"] = 5000
        elif args["dataset"] == "Boston Housing":
            args["lr"] = 0.0007
            args["n_est"] = 5000
        else:
            args["lr"] = 0.01
        args["minibatch_frac"] = 1.0 
        # kf = KFold(n_splits=args["n_splits"], random_state=args["seed"])
        # folds = kf.split(X)
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(args["random_state"])
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
        start_time = time.time()
        X_trainall, X_test = X[train_index], X[test_index]
        y_trainall, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainall, y_trainall, test_size=0.2, random_state=args["seed"]
        )

        y_true += list(y_test.flatten())


        ngb = NGBRegressor(
            Base=base_name_to_learner[args["base"]],
            Dist=eval(args["distn"]),
            Score=eval(args["score"]),
            n_estimators=args["n_est"],
            learning_rate=args["lr"],
            natural_gradient=args["natural"],
            minibatch_frac=args["minibatch_frac"],
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

        # re-train using all the data after tuning number of iterations
        ngb = NGBRegressor(
            Base=base_name_to_learner[args["base"]],
            Dist=eval(args["distn"]),
            Score=eval(args["score"]),
            n_estimators=args["n_est"],
            learning_rate=args["lr"],
            natural_gradient=args["natural"],
            minibatch_frac=args["minibatch_frac"],
            verbose=args["verbose"],
        )
        ngb.fit(X_trainall, y_trainall)

        # the final prediction for this fold
        forecast = ngb.pred_dist(X_test, max_iter=best_itr)
        forecast_val = ngb.pred_dist(X_val, max_iter=best_itr)

        # After processing all folds for a dataset:
        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time

        # set the appropriate scale if using a homoskedastic Normal
        if args["distn"] == "NormalFixedVar":
            scale = (
                forecast.var * ((forecast_val.loc - y_val.flatten()) ** 2).mean() ** 0.5
            )
            forecast = norm_dist(loc=forecast.loc, scale=scale)

        ngb_rmse += [np.sqrt(mean_squared_error(forecast.mean(), y_test))]
        ngb_nll += [-forecast.logpdf(y_test.flatten()).mean()]
        samples = np.array([[np.random.normal(loc=loc, scale=scale, size=100) for loc, scale in zip(forecast.loc, forecast.scale)]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test.flatten(), samples)
        ngb_crps += [crps_comps[0]]
        ngb_crps_cal += [crps_comps[1]]
        ngb_crps_sha += [crps_comps[2]]
        times += [elapsed_time]

        print(
                "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    itr + 1,
                    args['n_splits'],
                    best_itr,
                    np.sqrt(val_rmse[best_itr - 1]),
                    np.sqrt(mean_squared_error(forecast.mean(), y_test)),
                    ngb_nll[-1],
                    ngb_crps[-1],
                    ngb_crps_cal[-1],
                    ngb_crps_sha[-1],
                    elapsed_time,
                )
            )
    print(dset)
    print(
            "== GBM=%.4f +/- %.4f, RMSE NGBOOST =%.4f ± %.4f, NLL NGBOOST=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_CALIBRATION =  %.4f +/- %.4f, CRPS_SHARPNESS =  %.4f +/- %.4f,  TIME = %.4f"
            % (
                0.0,
                0.0,
                np.mean(ngb_rmse),
                np.std(ngb_rmse),
                np.mean(ngb_nll),
                np.std(ngb_nll),
                np.mean(ngb_crps),
                np.std(ngb_crps),
                np.mean(ngb_crps_cal),
                np.std(ngb_crps_cal),
                np.mean(ngb_crps_sha),
                np.std(ngb_crps_sha),
                np.mean(times)  # Include elapsed time in the output
            )
        )
    return dset, np.mean(ngb_rmse), np.std(ngb_rmse), np.mean(ngb_nll), np.std(ngb_nll), np.mean(ngb_crps), np.std(ngb_crps),  np.mean(times)

if __name__ == "__main__":
    header = ["dset","RMSE-mean","RMSE-std","NLL-mean","NLL-std","CRPS-mean","CRPS-std","CRPS-calibration-mean","CRPS-calibration-std","CRPS-sharpness-mean","CRPS-sharpness-std","time_run"]
    vsc_data = os.environ['VSC_DATA']
    print(sys.argv)
    results = run_single_arguement(sys.argv[1])
    file_path = "results/NGboost_natural_crps_calibration_sharpness.csv"
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    # Open the file in append mode ('a+')
    with open(file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist or is empty, write the header
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(header)  # Write header

        row_to_write = [results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]
        writer.writerow(row_to_write)
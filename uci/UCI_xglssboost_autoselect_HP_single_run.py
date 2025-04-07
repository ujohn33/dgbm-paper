import os
import sys
import json
import numpy as np
import pandas as pd
import time
import csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboostlss.model import *
from xgboostlss.distributions.distribution_utils import DistributionClass
from xgboostlss.distributions import *
from scipy.stats import norm
# Add import for evaluation functions from utils
from xgboostlss.utils import evaluate_nll, evaluate_crps

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.logging import log_predictions
from utils.safety import apply_safety_net

print("Usage: python openml_lssboost_HP_single_run.py <seed_id> <mode> <natural_grad> <stabilization> <clip value> <standardize>")


mode = sys.argv[2]  # e.g., 'exp'
natural_grad = sys.argv[3].lower() == 'true'  # Convert 'True' or 'False' to boolean
stabilization = sys.argv[4]  # e.g., 'L2', 'MAD', or 'None'
# turn "None" into None
# if clip_value not provided, clip_value is None
clip_value = None if len(sys.argv) <= 5 or sys.argv[5] == 'None' else float(sys.argv[5])
# If standardize not provided, default to False
standardize = False if len(sys.argv) <= 6 else sys.argv[6].lower() == 'true'


if natural_grad:
    method_name = f"XGBoostLSS_natural_{mode}_{stabilization}_{clip_value}_clip_std_{standardize}_auto_dist"
else:
    method_name = f"XGBoostLSS_no_natural_{mode}_{stabilization}_{clip_value}_clip_std_{standardize}_auto_dist"

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
    "dataset": None,
    "mode": mode,
    "natural_grad": natural_grad,
    "clip_value": clip_value,
    "stabilization": stabilization, # None, 'L2', "MAD"    
    "n_est": 200,
    "n_splits": 20,
    "score": "MLE",
    "distn": "auto",
    "random_state": 1,
    "standardize": standardize,
}

# Define your hyperparameter space
param_dict = {
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "min_child_weight": ["int", {"low": 1, "high": 100, "log": True}],
    "eta": ["float", {"low": 1e-5, "high": 0.4, "log": True}],
    #"subsample": ["float", {"low": 0.5, "high": 1.0, "log": False}],
    #'device':  ["categorical", ['cuda']],
}


def run_single_arguement(run_seed):
    dset = dataset_list[int(run_seed)]
    args["dataset"] = dset
    y_true, lss_nll, times, times_HP = [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []
    selected_distributions = []  # Add this line to track selected distributions

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={args['dataset']} X.shape={str(X.shape)} {args['score']}/{args['distn']} Standardize={args['standardize']}")
    lgbm_rmse = []
    if args["dataset"] == "Year Prediciton MSD":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
    elif args["dataset"] == "Protein Structure":
        # kf = KFold(n_splits=5)
        # folds = kf.split(X)
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(args['random_state'])
        folds = []
        for i in range(5):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))        
    else:
        # kf = KFold(n_splits=args["n_splits"])
        # folds = kf.split(X)
        # # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(args['random_state'])
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
            X_trainall, y_trainall, test_size=0.2, random_state=args['random_state']
        )

        # Auto-select the distribution if requested
        if args['distn'] == "auto":
            print(f"Selecting best distribution for fold {itr+1}...")
            lgblss_dist_class = DistributionClass()
            candidate_distributions = [Gaussian, StudentT, Gamma, LogNormal, Weibull, Gumbel, Laplace]
            print('Candidate distributions:', candidate_distributions)
            
            dist_fit_df = lgblss_dist_class.dist_select(
                target=y_train, 
                candidate_distributions=candidate_distributions, 
                max_iter=50, 
                plot=False
            )
            
            # Select the distribution with the lowest NLL
            best_dist_name = dist_fit_df.iloc[0]["distribution"]
            best_dist_module = next(d for d in candidate_distributions 
                                  if d.__name__.split(".")[-1] == best_dist_name)
            selected_distributions.append(best_dist_name)
            print(f"Selected distribution: {best_dist_name} with NLL: {dist_fit_df.iloc[0][lgblss_dist_class.loss_fn]}")
            
            # Create distribution instance
            distribution = getattr(best_dist_module, best_dist_name)(
                stabilization=args['stabilization'],
                response_fn=args['mode'],
                loss_fn="nll",
            )
        else:
            # Use the specified distribution
            distribution = Gaussian(
                stabilization=args['stabilization'],
                response_fn=args['mode'],
                loss_fn="nll"
            )
            selected_distributions.append("Gaussian")

        # Apply standardization based on the parameter
        if args['standardize'] or args['dataset'] == "Year Prediciton MSD":
            y_mean = np.mean(y_trainall)
            y_std = np.std(y_trainall)
            y_trainall = (y_trainall - y_mean) / y_std
            y_train = (y_train - y_mean) / y_std
            y_val = (y_val - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
        else:
            pass

        full_train_data = xgb.DMatrix(X_trainall, label=y_trainall)

        start_time = time.time()
        xgblss = XGBoostLSS(distribution)
        # Modify start values     
        # Change this line

        # Modify parameter dictionary for Year Prediciton MSD dataset
        current_param_dict = param_dict.copy()
        if dset == "Year Prediciton MSD":
            current_param_dict["subsample"] = ["categorical", [0.1]]
        else:
            current_param_dict = param_dict

        xgblss.start_values = np.array([np.array(0.5) for _ in range(xgblss.dist.n_dist_param)])

        opt_param = xgblss.hyper_opt(current_param_dict, full_train_data, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=20, max_minutes=80, n_trials=20,
                                    silence=True, seed=args['random_state'], hp_seed=args['random_state'])
        opt_params = opt_param.copy()

        end_time = time.time()  # End time measurement
        elapsed_time_HP = end_time - start_time  # Calculate elapsed time

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        # Training with early stopping
        evals_result = {}
        opt_params['early_stopping'] = 20
        # Train Model with optimized hyperparameters
        gbm = xgblss.train(opt_params, dtrain, 
                            num_boost_round = args["n_est"],
                            evals = [(dtrain, 'train'), (deval, 'eval')],
                            evals_result = evals_result,
                            early_stopping_rounds=20
                            )
        # Best iteration
        print(f"Best iteration: {xgblss.booster.best_iteration}")
        opt_params['early_stopping'] = None
        start_time = time.time()
        best_iter = xgblss.booster.best_iteration

        xgblss.start_values = np.array([np.array(0.5) for _ in range(xgblss.dist.n_dist_param)])

        final_gbm = xgblss.train(opt_params, full_train_data, 
                            num_boost_round = xgblss.booster.best_iteration,
                        )
        # the final prediction for this fold
        forecast = xgblss.predict(dtest)

        for param in xgblss.dist.distribution_arg_names:
            print(f" Raw Predictions for {param} - min: {forecast[param].min()}, max: {forecast[param].max()}, mean: {forecast[param].mean()}")


        # Handle rescaling for standardized data
        if args['standardize'] or args['dataset'] == "Year Prediciton MSD":
            for param in xgblss.dist.distribution_arg_names:  # Fix: use distribution_arg_names instead of params
                if param == xgblss.dist.distribution_arg_names[0]:  # location parameter
                    forecast[param] = forecast[param] * y_std + y_mean
                if param == xgblss.dist.distribution_arg_names[1]:  # scale parameter
                    forecast[param] = forecast[param] * y_std
            y_test = y_test * y_std + y_mean
            y_trainall = y_trainall * y_std + y_mean
        else:
            pass

        for param in xgblss.dist.distribution_arg_names:
            print(f" Rescaled Predictions for {param} - min: {forecast[param].min()}, max: {forecast[param].max()}, mean: {forecast[param].mean()}")

        forecast_val = xgblss.predict(deval)
        # Time the duration for forecast deployment
        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time
    
        # Calculate RMSE (no change needed)
        #lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'].values, y_test))]
        #val_rmse = [np.sqrt(mean_squared_error(forecast_val['loc'].values, y_val))]
        
        # Create DataFrame with predicted parameters for evaluation
        forecast_df = pd.DataFrame({param: forecast[param].values for param in xgblss.dist.distribution_arg_names})
        
        # Use the imported evaluate_nll function from utils
        nll_score = evaluate_nll(xgblss.dist.distribution, forecast_df, y_test)
        lss_nll += [nll_score]
        
        # Use the imported evaluate_crps function from utils
        crps_score, crps_cal, crps_sha = evaluate_crps(xgblss.dist.distribution, forecast_df, y_test, n_samples=100)
        lss_crps += [crps_score]
        lss_crps_cal += [crps_cal]
        lss_crps_sha += [crps_sha]
        
        times += [elapsed_time]
        times_HP += [elapsed_time_HP]

        print(
                "[%d/%d] BestIter=%d NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    itr + 1,
                    args['n_splits'],
                    best_iter,
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time,
                )
            )
    # After the folds are evaluated
    
    # Print summary of selected distributions if auto mode was used
    if args['distn'] == "auto":
        dist_counts = {}
        for dist in selected_distributions:
            dist_counts[dist] = dist_counts.get(dist, 0) + 1
        print("Selected distributions across folds:")
        for dist, count in dist_counts.items():
            print(f"- {dist}: {count} folds ({count/len(selected_distributions)*100:.1f}%)")
    
    print(dset)
    print(
            "== NLL GBMLSS=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_cal =  %.4f +/- %.4f, CRPS_sha =  %.4f +/- %.4f,  TIME = %.4f"
            % (
                np.mean(lss_nll),
                np.std(lss_nll),
                np.mean(lss_crps),
                np.std(lss_crps),
                np.mean(lss_crps_cal),
                np.std(lss_crps_cal),
                np.mean(lss_crps_sha),
                np.std(lss_crps_sha),
                np.mean(times),  # Include elapsed time in the output
            )
        )
    # return a dictonary of val
    return  dset, np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP)


if __name__ == "__main__":
    vsc_data = os.environ['VSC_DATA']
    results = run_single_arguement(sys.argv[1])
    if args['natural_grad']:
        file_path = f"results/uci/uci_{method_name}.csv"
    else:
        file_path = f"results/uci/uci_{method_name}.csv"
    header = ["dset","NLL-mean","NLL-std","CRPS-mean","CRPS-std","CRPS-calibration-mean","CRPS-calibration-std","CRPS-sharpness-mean","CRPS-sharpness-std","time_run","time_HP"]
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
                        results[10]]

        writer.writerow(row_to_write)
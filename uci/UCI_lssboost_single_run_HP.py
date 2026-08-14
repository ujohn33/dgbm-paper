import os
import sys
import json
import csv
import random
import warnings
import numpy as np
import pandas as pd
import time
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.utils\.parallel")
warnings.filterwarnings("ignore", category=UserWarning, module=r"gluonts\.json")
warnings.filterwarnings("ignore", category=FutureWarning, message=r"The 'delim_whitespace' keyword in pd\.read_csv is deprecated.*")

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.logging import log_predictions
from utils.safety import apply_safety_net 


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


print("Usage: python UCI_lssboost_single_run_HP.py <seed_id> <mode> <natural_grad> <stabilization> <clip_value> <standardize> [apply_safety]")

mode = sys.argv[2]  # e.g., 'exp'
natural_grad = sys.argv[3].lower() == 'true'  # Convert 'True' or 'False' to boolean
stabilization = sys.argv[4]  # e.g., 'L2', 'MAD', or 'None'
clip_value = None if len(sys.argv) <= 5 or sys.argv[5] == 'None' else float(sys.argv[5])
# If standardize not provided, default to False
standardize = False if len(sys.argv) <= 6 else sys.argv[6].lower() == 'true'
apply_safety = False if len(sys.argv) <= 7 else sys.argv[7].lower() == 'true'
    
if natural_grad:
    method_name = f'LSSboost_natural_{mode}_{stabilization}_std_{standardize}_safety_{apply_safety}'
else:
    method_name = f'LSSboost_no_natural_{mode}_{stabilization}_std_{standardize}_safety_{apply_safety}'

dataset_name_to_loader = {
    "Boston Housing": lambda: pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        header=None,
        sep=r"\s+",
    ),
    "Concrete Compression Strength": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    ),
    "Energy Efficiency": lambda: pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    ).iloc[:, :-1],
    "Kin8nm": lambda: pd.read_csv("ngboost/data/uci/kin8nm.csv"),
    "Naval Propulsion": lambda: pd.read_csv(
        "ngboost/data/uci/naval-propulsion.txt", sep=r"\s+", header=None
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
        sep=r"\s+",
    ),
    "Year Prediciton MSD": lambda: pd.read_csv("ngboost/data/uci/YearPredictionMSD.txt").iloc[:, ::-1],
}

dataset_list = ["Boston Housing", "Concrete Compression Strength", "Energy Efficiency", "Kin8nm", "Naval Propulsion", "Combined Cycle Power Plant", "Protein Structure", "Wine Quality Red", "Yacht Hydrodynamics", "Year Prediciton MSD"]

# Hardcoded parameters for testing
args = {
    "mode": mode,
    "natural_grad": natural_grad,
    "stabilization": stabilization, # None, 'L2', "MAD"  
    "clip_value": clip_value,
    "n_est": 2000,
    "n_splits": 20,
    "score": "MLE",
    "distn": "Normal",
    "standardize": standardize,
    "apply_safety": apply_safety,
    "random_state": 1,
}



# Initialisation of the additive predictor. "constant" reproduces the published
# runs, which start every distributional parameter at 0.5 regardless of the
# data. "moment" starts at the marginal maximum-likelihood fit, so boosting does
# not have to spend rounds walking the intercept to the target's location and
# scale. Select with DGBM_INIT=moment.
INIT_MODE = os.environ.get("DGBM_INIT", "constant")


def _start_values(n_param, y, mode):
    """Starting value of the additive predictor for a Gaussian NLL."""
    if INIT_MODE != "moment":
        return np.array([np.array(0.5) for _ in range(n_param)])
    eps = 1e-8
    mu0 = float(np.mean(y))
    sigma0 = max(float(np.std(y)), eps)
    if mode == "exp":                       # sigma = exp(f)
        return np.array([mu0, np.log(sigma0)], dtype=float)
    if mode == "softplus":                  # sigma = softplus(f)
        return np.array([mu0, np.log(np.expm1(sigma0) + eps)], dtype=float)
    return np.array([np.array(0.5) for _ in range(n_param)])


def detect_categorical_features(df, threshold_unique=20, threshold_ratio=0.05):
    """
    Detect likely categorical features in a DataFrame.
    Parameters:
    - df: pandas DataFrame
    - threshold_unique: maximum number of unique values to consider a feature categorical
    - threshold_ratio: maximum ratio of unique values to total samples to consider categorical
    Returns:
    - List of column names likely to be categorical
    """
    categorical_cols = []
    for col in df.columns:
        num_unique = df[col].nunique()
        total_samples = len(df[col])
        if pd.api.types.is_integer_dtype(df[col]):
            if (num_unique <= threshold_unique) or (num_unique / total_samples <= threshold_ratio):
                categorical_cols.append(col)
        elif (pd.api.types.is_object_dtype(df[col]) or 
              pd.api.types.is_categorical_dtype(df[col]) or 
              pd.api.types.is_string_dtype(df[col])):
            categorical_cols.append(col)
    return categorical_cols

# Define your hyperparameter space
# param_dict = {
#     "eta": ["float", {"low": 1e-5, "high": 0.4, "log": True}],
#     "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
#     "num_leaves": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
#     "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
#     "lambda_l1": ["float", {"low": 1e-8, "high": 10, "log": True}],
#     "lambda_l2": ["float", {"low": 1e-8, "high": 10, "log": True}],
#     "bagging_fraction": ["float", {"low": 0.6, "high": 1.0, "log": False}],
#     "bagging_freq": ["int", {"low": 1, "high": 10, "log": False}],
#     "feature_pre_filter": ["categorical", [False]],
#     #'device':  ["categorical", ['cuda']],
#     #'clip_value': ["float", {"low": 1e-6, "high": 1e-1, "log": True}],
#     #'max_bin': ["int", {"low": 16, "high": 255, "log": False}],
#     # "min_child_weight": ["categorical", [1.0]],
#     # "device": ["categorical", ['gpu']]
# }

param_dict = {
    "eta": ["float", {"low": 1e-5, "high": 0.4, "log": True}],
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "num_leaves": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "feature_pre_filter": ["categorical", [False]],
}

param_dict["device"] = ["categorical", ['cuda']] if torch.cuda.is_available() else ["categorical", ['cpu']]

def run_single_arguement(run_seed):
    dset = dataset_list[int(run_seed)]
    lss_rmse, lss_nll, times, times_HP = [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []
    fold_metrics = []

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[dset]()
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # Detect categorical features using the integrated function
    # cat_features = detect_categorical_features(X, threshold_unique=20, threshold_ratio=0.05)
    # print(f"Detected categorical features: {cat_features}")


    print(f"== Dataset={dset} X.shape={str(X.shape)} {args['score']}/{args['distn']} Standardize={args['standardize']}")
    if dset == "Year Prediciton MSD":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
    elif dset == "Protein Structure":
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
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
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
        # Use loc to keep as DataFrames (not values attribute)
        X_trainall = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_trainall = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainall, y_trainall, test_size=0.2, random_state=args['random_state']
        )


        # Apply standardization based on the parameter
        if args['standardize'] or dset == "Year Prediciton MSD":
            y_mean = np.mean(y_trainall)
            y_std = np.std(y_trainall)
            y_trainall = (y_trainall - y_mean) / y_std
            y_train = (y_train - y_mean) / y_std
            y_val = (y_val - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
        else:
            pass

        full_train_data = lgb.Dataset(X_trainall, y_trainall)
        #full_train_data = lgb.Dataset(X_trainall, y_trainall, categorical_feature=cat_features, free_raw_data=False)

        start_time = time.time()
        lgblss = LightGBMLSS(
            Gaussian(stabilization=args['stabilization'],
                    response_fn = args['mode'],
                    loss_fn = "nll"
                    )
        )
        # Modify start values     
        lgblss.start_values = _start_values(lgblss.dist.n_dist_param, y_trainall, args['mode'])
        
        # Modify parameter dictionary for Year Prediciton MSD dataset
        current_param_dict = param_dict.copy()
        if dset == "Year Prediciton MSD":
            current_param_dict["bagging_fraction"] = ["categorical", [0.1]]
        else:
            current_param_dict = param_dict

        opt_param = lgblss.hyper_opt(current_param_dict, full_train_data, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=20, max_minutes=1440, n_trials=200,
                                    silence=True, seed=args['random_state'], hp_seed=args['random_state'])
        opt_params = opt_param.copy()

        end_time = time.time()  # End time measurement
        elapsed_time_HP = end_time - start_time  # Calculate elapsed time

        dtrain = lgb.Dataset(X_train, y_train)
        deval = lgb.Dataset(X_val, y_val)
        dtest = lgb.Dataset(X_test, y_test)

        # dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
        # deval = lgb.Dataset(X_val, y_val, categorical_feature=cat_features, free_raw_data=False)
        # dtest = lgb.Dataset(X_test, y_test, categorical_feature=cat_features, free_raw_data=False)

        # Training with early stopping
        evals_result = {}
        opt_params['early_stopping'] = 20
        # Train Model with optimized hyperparameters
        gbm = lgblss.train(opt_params, dtrain, 
                            num_boost_round = args["n_est"],
                            valid_sets = [dtrain, deval]
                            )

        # Best iteration
        print(f"Best iteration: {lgblss.booster.best_iteration}")

        opt_params['early_stopping'] = None
        best_iter = lgblss.booster.best_iteration

        start_time = time.time()
        final_gbm = lgblss.train(opt_params, full_train_data, 
                            num_boost_round = lgblss.booster.best_iteration,
                        )
        # the final prediction for this fold
        forecast = lgblss.predict(X_test)

        print(f"Raw Pedictions - min: {forecast['loc'].min()}, max: {forecast['loc'].max()}, mean: {forecast['loc'].mean()}")

        # Handle rescaling for standardized data
        if args['standardize'] or dset == "Year Prediciton MSD":
            forecast['loc'] = forecast['loc'] * y_std + y_mean
            forecast['scale'] = forecast['scale'] * y_std
            y_test = y_test * y_std + y_mean
            y_trainall = y_trainall * y_std + y_mean
        else:
            pass

        print(f"Pedictions after rescaling - min: {forecast['loc'].min()}, max: {forecast['loc'].max()}, mean: {forecast['loc'].mean()}")

        forecast_val = lgblss.predict(X_val)
        if args["apply_safety"]:
            forecast = apply_safety_net(forecast, y_trainall.values)
            forecast_val = apply_safety_net(forecast_val, y_trainall.values)
        
        # Time the duration for forecast deployment
        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time

        lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'].values, y_test))]
        val_rmse = np.sqrt(mean_squared_error(forecast_val['loc'].values, y_val))
        lss_nll += [-norm(forecast['loc'], forecast['scale']).logpdf(y_test).mean()]
        samples = np.array([[np.random.normal(loc=loc, scale=scale, size=100) for loc, scale in zip(forecast['loc'], forecast['scale'])]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(np.asarray(y_test).ravel(), samples)
        lss_crps += [crps_comps[0]]
        lss_crps_cal += [crps_comps[1]]
        lss_crps_sha += [crps_comps[2]]
        times += [elapsed_time]
        times_HP += [elapsed_time_HP]

        # Define the quantiles to evaluate
        quantiles = [0.1, 0.5, 0.9]

        # Compute the quantiles for each observation
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = norm.ppf(q, loc=forecast['loc'], scale=forecast['scale'])
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)

        # Log predictions for each fold
        log_predictions(itr, dset, y_test, forecast['loc'], forecast['scale'], quantile_preds, f"logs/uci/predictions/{method_name}.csv")

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]
        fold_metrics.append(
            {
                "dataset": dset,
                "fold": itr + 1,
                "n_folds": args["n_splits"],
                "best_iter": int(best_iter),
                "rmse_val": float(val_rmse),
                "rmse_test": float(lss_rmse[-1]),
                "nll_test": float(lss_nll[-1]),
                "crps_test": float(lss_crps[-1]),
                "crps_cal_test": float(lss_crps_cal[-1]),
                "crps_sha_test": float(lss_crps_sha[-1]),
                "wql01_test": float(wql_01[-1]),
                "wql05_test": float(wql_05[-1]),
                "wql09_test": float(wql_09[-1]),
                "wql_avg_test": float(wql_avg[-1]),
                "time_pred": float(elapsed_time),
                "time_hp": float(elapsed_time_HP),
                "dataset_idx": int(run_seed),
                "standardize": args["standardize"],
                "apply_safety": args["apply_safety"],
                "stabilization": args["stabilization"],
                "response_mode": args["mode"],
                "natural_grad": args["natural_grad"],
            }
        )

        print(
                "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    itr + 1,
                    args['n_splits'],
                    best_iter,
                    val_rmse,
                    np.sqrt(mean_squared_error(forecast['loc'].values, y_test)),
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time,
                )
            )
    log_dir = "logs/uci/fold_metrics"
    os.makedirs(log_dir, exist_ok=True)
    fold_log_path = os.path.join(
        log_dir,
        f"{method_name}_dataset_{str(dset).replace(' ', '_')}_idx{int(run_seed)}.csv",
    )
    pd.DataFrame(fold_metrics).to_csv(fold_log_path, index=False)
    print(f"Saved per-fold metrics to {fold_log_path}")
    print(dset)
    print(
            "== RMSE GBMLSS=%.4f ± %.4f, NLL GBMLSS=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_cal =  %.4f +/- %.4f, CRPS_sha =  %.4f +/- %.4f,  TIME = %.4f"
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
    # return a dictonary of val
    return  dset, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP), np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 

if __name__ == "__main__":
    vsc_data = os.environ['VSC_DATA']
    run_seed = int(sys.argv[1])
    seed_everything(run_seed)
    args["random_state"] = run_seed
    results = run_single_arguement(run_seed)
    method_name = f"{method_name}_n_est_{args['n_est']}"
    if INIT_MODE != "constant":
        method_name = f"{method_name}_init_{INIT_MODE}"  # Assuming args['n_estimators'] exists
    results_dir = "results/uci"
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"uci_{method_name}_seed{run_seed}.csv")
    header = ["dset","RMSE-mean","RMSE-std","NLL-mean","NLL-std","CRPS-mean","CRPS-std","CRPS-calibration-mean","CRPS-calibration-std","CRPS-sharpness-mean","CRPS-sharpness-std","time_run","time_HP","WQL01-mean", "WQL01-std","WQL05-mean", "WQL05-std","WQL09-mean", "WQL09-std", "WQL_avg-mean", "WQL_avg-std"]
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    # saving the results
    print(f"Saving results to {file_path}")
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

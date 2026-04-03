import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
import csv
import random
import torch
import lightgbm as lgb
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# Set OpenML API key
openml.config.apikey = '0fc137c28db32cdfecb6347178c7be68'

# Get command-line arguments
print("Usage: python UCI_lssboost_single_run_HP.py <task_idx> <mode> <natural_grad> <stabilization> <clip_value> <standardize> [run_seed] [apply_safety] [scale_floor_rel]")

mode = sys.argv[2]  # e.g., 'exp'
natural_grad = sys.argv[3].lower() == 'true'  # Convert 'True' or 'False' to boolean
stabilization = sys.argv[4]  # e.g., 'L2', 'MAD', or 'None'
clip_value = None if len(sys.argv) <= 5 or sys.argv[5] == 'None' else float(sys.argv[5])
standardize = False if len(sys.argv) <= 6 else sys.argv[6].lower() == 'true'
run_seed = 123 if len(sys.argv) <= 7 else int(sys.argv[7])
apply_safety = False if len(sys.argv) <= 8 else sys.argv[8].lower() == 'true'

seed_everything(run_seed)
    

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

# Define constants and parameters
args = {
    "mode": mode,
    "natural_grad": natural_grad,
    "stabilization": stabilization, # None, 'L2', "MAD"
    #"quantile_clipping": quantile_clipping,
    #"clip_value": None,
    "SUITE_ID": 336, # Regression on numerical features
    "n_est": 200,
    "n_trials": 80,
    "n_splits": 5,
    "score": "MLE",
    "distn": "Normal",
    "apply_safety": apply_safety,
}


if natural_grad:
    method_name = f'LSSboost_natural_{mode}_{stabilization}_std_{standardize}_safety_{apply_safety}_srel_{scale_floor_rel}'
else:
    method_name = f'LSSboost_no_natural_{mode}_{stabilization}_std_{standardize}_safety_{apply_safety}_srel_{scale_floor_rel}'


# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(args['SUITE_ID'])  # obtain the benchmark suite

# Define your hyperparameter space
param_dict = {
    "eta": ["float", {"low": 1e-5, "high": 1e-1, "log": True}],
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "num_leaves": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "feature_pre_filter": ["categorical", [False]],
    "lambda_l1": ["float", {"low": 1e-8, "high": 10, "log": True}],
    #"lambda_l1": ["float", {"low": 1e-8, "high": 10, "log": True}],
    #'clip_value': ["float", {"low": 1e-6, "high": 1e-1, "log": True}],
    #'deterministic': ["categorical", [True]],
    #'min_child_weight': ["categorical", [1]],
    #"histogram_pool_size": ["categorical", [16384]],
}

param_dict["device"] = ["categorical", ['cuda']] if torch.cuda.is_available() else ["categorical", ['cpu']]
if torch.cuda.is_available():
    print("CUDA is available. Using GPU for training.")
else:
    print("CUDA is not available. Using CPU for training.")

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    dset_name = dataset.name
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    # Detect categorical features using the integrated function
    # cat_features = detect_categorical_features(X, threshold_unique=20, threshold_ratio=0.05)
    # print(f"Detected categorical features: {cat_features}")

    lss_rmse, lss_nll, times = [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X.iloc[train_indices], X.iloc[test_indices]
    y_train_opt, y_test_opt = y.iloc[train_indices], y.iloc[test_indices]

    if standardize:
        y_opt_mean = float(np.mean(y_train_opt))
        y_opt_std = float(np.std(y_train_opt))
        y_opt_std = 1.0 if y_opt_std < 1e-12 else y_opt_std
        y_train_opt_fit = (y_train_opt - y_opt_mean) / y_opt_std
    else:
        y_train_opt_fit = y_train_opt

    dtrain = lgb.Dataset(X_train_opt, y_train_opt_fit)
    #dtrain = lgb.Dataset(X_train_opt, y_train_opt, categorical_feature=cat_features, free_raw_data=False)

    start_time = time.time()  # Start time measurement

    lgblss = LightGBMLSS(
        Gaussian(
            stabilization=args['stabilization'],
            response_fn=args['mode'],
            loss_fn="nll",
            natural_gradient=args['natural_grad'],
        )
    )
    #lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])
    lgblss.start_values = np.array([np.mean(y_train_opt), np.std(y_train_opt)])

    opt_param = lgblss.hyper_opt(param_dict, dtrain, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=args["n_est"], max_minutes=1440, n_trials=args['n_trials'],
                                    silence=True, seed=run_seed, hp_seed=run_seed)

    end_time = time.time()  # End time measurement
    elapsed_time_HP = end_time - start_time  # Calculate elapsed time

    print(opt_param)
    opt_params = opt_param.copy()
    if args['natural_grad']:
        with open(f'logs/openml/lssboost/natural/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    else:
        with open(f'logs/openml/lssboost/normal/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    
    print("OPT PARAM OUTPUT:", opt_param)  # Debugging print
    n_rounds = opt_param.get("opt_rounds", 100)  # Default to 100 if missing
    #n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]
    opt_params.update({
        "seed": run_seed,
        "bagging_seed": run_seed,
        "feature_fraction_seed": run_seed,
        "data_random_seed": run_seed,
        "deterministic": True,
        "num_threads": 1,
    })

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_trainall, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_trainall, y_test = y.iloc[train_indices], y.iloc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainall,
            y_trainall,
            test_size=0.2,
            random_state=run_seed + fold,
            shuffle=True,
        )

        if standardize:
            y_mean = float(np.mean(y_trainall))
            y_std = float(np.std(y_trainall))
            y_std = 1.0 if y_std < 1e-12 else y_std
            y_train_fit = (y_train - y_mean) / y_std
            y_val_fit = (y_val - y_mean) / y_std
            y_trainall_fit = (y_trainall - y_mean) / y_std
        else:
            y_train_fit = y_train
            y_val_fit = y_val
            y_trainall_fit = y_trainall

        dtrain = lgb.Dataset(X_train, y_train_fit)
        deval = lgb.Dataset(X_val, y_val_fit)
        dtest = lgb.Dataset(X_test, y_test)

        # dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
        # deval = lgb.Dataset(X_val, y_val, categorical_feature=cat_features, free_raw_data=False)
        # dtest = lgb.Dataset(X_test, y_test, categorical_feature=cat_features, free_raw_data=False)
        opt_params['early_stopping'] = 20

        gbm = lgblss.train(opt_params, dtrain, 
                            num_boost_round=n_rounds,
                            valid_sets=[dtrain, deval],
                            )

        full_train_data = lgb.Dataset(X_trainall, y_trainall_fit)
        opt_params['early_stopping'] = None

        runtime_start = time.time()

        final_gbm = lgblss.train(opt_params, full_train_data, 
                            num_boost_round=lgblss.booster.best_iteration,
                        )

        forecast = lgblss.predict(X_test)
        
        runtime_pred = time.time() - runtime_start

        forecast_val = lgblss.predict(X_val)

        if standardize:
            forecast["loc"] = forecast["loc"] * y_std + y_mean
            forecast["scale"] = forecast["scale"] * y_std
            forecast_val["loc"] = forecast_val["loc"] * y_std + y_mean
            forecast_val["scale"] = forecast_val["scale"] * y_std

        if args["apply_safety"]:
            forecast = apply_safety_net(forecast, y_trainall.values)
            forecast_val = apply_safety_net(forecast_val, y_trainall.values)

        lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'].values, y_test))]
        val_rmse = [np.sqrt(mean_squared_error(forecast_val['loc'].values, y_val))]
        lss_nll += [-norm(forecast['loc'], forecast['scale']).logpdf(y_test).mean()]
        rng = np.random.default_rng(run_seed + task_id * 1000 + fold)
        samples = np.array([[rng.normal(loc=loc, scale=scale, size=100) for loc, scale in zip(forecast['loc'], forecast['scale'])]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test, samples)
        lss_crps += [crps_comps[0]]
        lss_crps_cal += [crps_comps[1]]
        lss_crps_sha += [crps_comps[2]]
        times += [runtime_pred]

        # Define the quantiles to evaluate
        quantiles = [0.1, 0.5, 0.9]

        # Compute the quantiles for each observation
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = norm.ppf(q, loc=forecast['loc'], scale=forecast['scale'])
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)
        
        #print(y_test)
        # Log predictions for each fold
        #log_predictions(fold, dataset.name, y_test.values, forecast['loc'], forecast['scale'], quantile_preds, f"logs/openml/predictions/{method_name}.csv")

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]
        print(
                "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    fold,
                    n_folds,
                    lgblss.booster.best_iteration,
                    np.sqrt(val_rmse),
                    np.sqrt(mean_squared_error(forecast['loc'].values, y_test)),
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time_HP,
                )
            )

    print(task_id)
    print(dataset.name)
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
    return  dset_name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), elapsed_time_HP, np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 

if __name__ == "__main__":
    print("LIGHTGBMLSS")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    results = run_single_argument(task_number)
    batch_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "11697976"
    file_path = f"results/openml/openml_{method_name}_job_{batch_job_id}.csv"
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

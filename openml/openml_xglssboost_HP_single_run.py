import openml
import os
import sys
import json
import random
import socket
import traceback
import warnings
import faulthandler
import numpy as np
import pandas as pd
import time
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from scipy.stats import norm
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.logging import log_predictions
from utils.safety import apply_safety_net


DEBUG_LOG_PATH = None
DEBUG_LOG_DISABLED = False


def _output_dir(env_var, default_relative_path):
    path = os.environ.get(env_var) or default_relative_path
    os.makedirs(path, exist_ok=True)
    return path


def _safe_repr(value):
    try:
        return repr(value)
    except Exception:
        return f"<unreprable {type(value).__name__}>"


def log_event(message, **kwargs):
    global DEBUG_LOG_DISABLED
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = " ".join(f"{key}={_safe_repr(value)}" for key, value in sorted(kwargs.items()))
    line = f"[{timestamp}] {message}"
    if payload:
        line = f"{line} | {payload}"
    print(line, flush=True)
    if DEBUG_LOG_PATH and not DEBUG_LOG_DISABLED:
        try:
            with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as debug_file:
                debug_file.write(line + "\n")
                debug_file.flush()
        except OSError as exc:
            DEBUG_LOG_DISABLED = True
            print(
                f"[{timestamp}] Debug logging disabled after write failure: {exc}",
                file=sys.stderr,
                flush=True,
            )


def configure_debug_logging():
    global DEBUG_LOG_PATH
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or "na"
    pid = os.getpid()
    log_dir = _output_dir("OPENML_DEBUG_DIR", "logs/openml/debug")
    DEBUG_LOG_PATH = os.path.join(log_dir, f"xgboost_job{job_id}_task{task_id}_pid{pid}.log")
    with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as debug_file:
        debug_file.write("")
    faulthandler.enable(all_threads=True)
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as fault_log:
        faulthandler.enable(file=fault_log, all_threads=True)
    return DEBUG_LOG_PATH


def log_runtime_context():
    log_event(
        "Runtime context",
        argv=sys.argv,
        cwd=os.getcwd(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        python=sys.executable,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        openml_cache=os.environ.get("OPENML_CACHE_DIR"),
        tmpdir=os.environ.get("TMPDIR"),
        debug_dir=os.environ.get("OPENML_DEBUG_DIR"),
        fold_metrics_dir=os.environ.get("OPENML_FOLD_METRICS_DIR"),
        results_dir=os.environ.get("OPENML_RESULTS_DIR"),
        torch_cuda_available=torch.cuda.is_available(),
    )


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
openml.config.apikey = os.environ.get("OPENML_APIKEY", "")
configure_debug_logging()
warnings.simplefilter("always")
log_runtime_context()

# Get command-line arguments
print("Usage: python openml_xglssboost_HP_single_run.py <task_idx> <mode> <natural_grad> <stabilization> <clip_value> <standardize> [run_seed] [safety]")

mode = sys.argv[2]  # e.g., 'exp'
natural_grad = sys.argv[3].lower() == 'true'  # Convert 'True' or 'False' to boolean
stabilization = sys.argv[4]  # e.g., 'L2', 'MAD', or 'None'
clip_value = None if len(sys.argv) <= 5 or sys.argv[5] == 'None' else float(sys.argv[5])
standardize = False if len(sys.argv) <= 6 else sys.argv[6].lower() == 'true'
run_seed = 123 if len(sys.argv) <= 7 else int(sys.argv[7])
safety = False if len(sys.argv) <= 8 else sys.argv[8].lower() == 'true'
seed_everything(run_seed)
log_event(
    "Parsed CLI arguments",
    mode=mode,
    natural_grad=natural_grad,
    stabilization=stabilization,
    clip_value=clip_value,
    standardize=standardize,
    run_seed=run_seed,
    safety=safety,
)
    

if natural_grad:
    method_name = f"XGBoostLSS_natural_{mode}_{stabilization}_safety_{safety}"
else:
    method_name = f"XGBoostLSS_no_natural_{mode}_{stabilization}_safety_{safety}"

# Define constants and parameters
args = {
    "SUITE_ID": 336, # Regression on numerical features
    "mode": mode,
    "stabilization": stabilization, #"MAD", "L2", None
    "natural_grad": natural_grad, #True, False
    "n_est": 2000,
    "n_splits": 5,
    "score": "MLE",
    "distn": "Normal",
}

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(args["SUITE_ID"])  # obtain the benchmark suite

# Define your hyperparameter space
param_dict = {
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "min_child_weight": ["int", {"low": 1, "high": 100, "log": True}],
    "eta": ["float", {"low": 1e-5, "high": 0.4, "log": True}],
    "subsample": ["float", {"low": 0.5, "high": 1.0, "log": False}],
    "reg_alpha": ["float", {"low": 1e-8, "high": 10, "log": True}],
    #'clip_value': ["float", {"low": 1e-6, "high": 1e-1, "log": True}],
}

param_dict["device"] = ["categorical", ['cuda']] if torch.cuda.is_available() else ["categorical", ['cpu']]


def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
    return df

def run_single_argument(task_id):
    log_event("Fetching OpenML task", task_id=task_id)
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    # Encode categorical columns
    X = encode_categorical_columns(X)
    
    lss_rmse, lss_nll, times = [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []
    fold_metrics = []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}")
    log_event(
        "Loaded dataset",
        task_id=task_id,
        dataset=dataset.name,
        shape=X.shape,
        target_dtype=str(y.dtype),
        n_missing_features=int(X.isna().sum().sum()),
        n_missing_target=int(y.isna().sum()),
    )

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")
    log_event("Task split dimensions", n_repeats=n_repeats, n_folds=n_folds, n_samples=n_samples)

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X.iloc[train_indices], X.iloc[test_indices]
    y_train_opt, y_test_opt = y.iloc[train_indices], y.iloc[test_indices]

    dtrain = xgb.DMatrix(X_train_opt, label=y_train_opt)

    start_time = time.time()  # Start time measurement
    log_event(
        "Starting hyperparameter search",
        train_shape=X_train_opt.shape,
        test_shape=X_test_opt.shape,
        y_train_mean=float(np.mean(y_train_opt)),
        y_train_std=float(np.std(y_train_opt)),
        param_space=param_dict,
    )

    xgblss = XGBoostLSS(Gaussian(stabilization=args['stabilization'], response_fn=args['mode'], loss_fn="nll", 
                                 natural_gradient=args["natural_grad"]))
    eps = 1e-8
    mu0 = float(np.mean(y_train_opt))
    sigma0 = max(float(np.std(y_train_opt)), eps)
    # if args["mode"] == "exp":
    #     xgblss.start_values = np.array([mu0, np.log(sigma0)], dtype=float)
    # elif args["mode"] == "softplus":
    #     xgblss.start_values = np.array([mu0, np.log(np.expm1(sigma0) + eps)], dtype=float)
    # else:
    #     xgblss.start_values = np.array([np.array(0.5) for _ in range(xgblss.dist.n_dist_param)])
    xgblss.start_values = np.array([np.array(0.5) for _ in range(xgblss.dist.n_dist_param)])
    log_event("Initialized start values", start_values=xgblss.start_values.tolist())

    opt_param = xgblss.hyper_opt(param_dict, dtrain, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=20, max_minutes=1440, n_trials=80,
                                    silence=True, seed=run_seed, hp_seed=run_seed)

    end_time = time.time()  # End time measurement
    elapsed_time_HP = end_time - start_time  # Calculate elapsed time

    print(opt_param)
    log_event("Hyperparameter search finished", elapsed_time_hp=elapsed_time_HP, opt_param=opt_param)
    opt_params = opt_param.copy()
    opt_param_dir = _output_dir(
        "OPENML_XGB_OPT_PARAMS_DIR",
        f"logs/openml/xgboost/{'natural' if args['natural_grad'] else 'normal'}/exp",
    )
    opt_param_path = os.path.join(opt_param_dir, f"{dataset.name}_opt_params.json")
    if args['natural_grad']:
        with open(opt_param_path, 'w') as f:
            json.dump(opt_params, f)
    else:
        with open(opt_param_path, 'w') as f:
            json.dump(opt_params, f)
    
    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]
    opt_params.update({
        "seed": run_seed,
        "random_state": run_seed,
    })

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        log_event("Starting evaluation fold", fold=fold, n_folds=n_folds, dataset=dataset.name)
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

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        evals_result = {}
        opt_params['early_stopping'] = 20

        gbm = xgblss.train(opt_params, dtrain, 
                            num_boost_round=n_rounds,
                            evals=[(dtrain, 'train'), (deval, 'eval')],
                            evals_result=evals_result,
                            early_stopping_rounds=20
                            )

        full_train_data = xgb.DMatrix(X_trainall, label=y_trainall)
        opt_params['early_stopping'] = None
        best_iter = xgblss.booster.best_iteration
        log_event("Finished early-stopped training", fold=fold, best_iter=best_iter)

        best_iter = xgblss.booster.best_iteration
        runtime_start = time.time()
        mu_fold = float(np.mean(y_trainall))
        sigma_fold = max(float(np.std(y_trainall)), eps)

        final_gbm = xgblss.train(opt_params, full_train_data, 
                            num_boost_round=best_iter,
                        )
        log_event("Finished full-train refit", fold=fold, best_iter=best_iter)

        forecast = xgblss.predict(dtest)
        runtime_pred = time.time() - runtime_start

        forecast_val = xgblss.predict(deval)
        if safety:
            forecast = apply_safety_net(forecast, y_trainall.values)
            forecast_val = apply_safety_net(forecast_val, y_trainall.values)

        lss_rmse += [np.sqrt(mean_squared_error(forecast['loc'], y_test))]
        val_rmse = np.sqrt(mean_squared_error(forecast_val['loc'], y_val))
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
        
        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        #log_predictions(fold, dataset.name, y_test.values, forecast['loc'], forecast['scale'], quantile_preds, f"logs/openml/predictions/{method_name}_{str(args['mode'])}_{str(args['stabilization'])}.csv")

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]
        fold_metrics.append(
            {
                "task_id": task_id,
                "dataset": dataset.name,
                "fold": fold,
                "n_folds": n_folds,
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
                "time_pred": float(runtime_pred),
                "time_hp": float(elapsed_time_HP),
                "run_seed": run_seed,
                "stabilization": args["stabilization"],
                "response_mode": args["mode"],
                "natural_grad": args["natural_grad"],
            }
        )
        log_event(
            "Completed evaluation fold",
            fold=fold,
            best_iter=best_iter,
            rmse_val=float(val_rmse),
            rmse_test=float(lss_rmse[-1]),
            nll_test=float(lss_nll[-1]),
            crps_test=float(lss_crps[-1]),
            time_pred=float(runtime_pred),
        )

        print(
                "[%d/%d] BestIter=%d RMSE: Val=%.4f Test=%.4f NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    fold,
                    n_folds,
                    best_iter,
                    val_rmse,
                    np.sqrt(mean_squared_error(forecast['loc'], y_test)),
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time_HP,
                )
            )
    log_dir = _output_dir("OPENML_FOLD_METRICS_DIR", "logs/openml/fold_metrics")
    fold_log_path = os.path.join(log_dir, f"{method_name}_task{task_id}_seed{run_seed}.csv")
    pd.DataFrame(fold_metrics).to_csv(fold_log_path, index=False)
    print(f"Saved per-fold metrics to {fold_log_path}")
    log_event("Saved fold metrics", fold_log_path=fold_log_path, rows=len(fold_metrics))

    print(task_id)
    print(dataset.name)
    print(
            "== RMSE XGBoostLSS=%.4f ± %.4f, NLL XGBoostLSS=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_cal =  %.4f +/- %.4f, CRPS_sha =  %.4f +/- %.4f,  TIME = %.4f"
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
    return dataset.name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), elapsed_time_HP, np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 

if __name__ == "__main__":
    try:
        print("XGBOOSTLSS")
        print("______________________")
        task_idx = int(sys.argv[1])
        task_number = benchmark_suite.tasks[task_idx]
        log_event("Resolved benchmark task", task_idx=task_idx, task_number=task_number)
        results = run_single_argument(task_number)
        batch_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "11697976"
        results_dir = _output_dir("OPENML_RESULTS_DIR", "results/openml")
        file_path = os.path.join(results_dir, f"openml_{method_name}_job_{batch_job_id}.csv")
        header = ["dset","RMSE-mean","RMSE-std","NLL-mean","NLL-std","CRPS-mean","CRPS-std","CRPS-calibration-mean","CRPS-calibration-std","CRPS-sharpness-mean","CRPS-sharpness-std","time_run","time_HP","WQL01-mean", "WQL01-std","WQL05-mean", "WQL05-std","WQL09-mean", "WQL09-std", "WQL_avg-mean", "WQL_avg-std"]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a+', newline='') as file:
            writer = csv.writer(file)

            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writerow(header)

            row_to_write = [results[0], results[1], results[2], results[3], results[4],
                            results[5], results[6], results[7], results[8], results[9],
                            results[10], results[11], results[12], results[13],
                            results[14], results[15], results[16], results[17],
                            results[18], results[19], results[20]]

            writer.writerow(row_to_write)
        log_event("Saved batch result row", file_path=file_path, dataset=results[0])
    except Exception as exc:
        log_event("Fatal exception", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        raise

import openml
import torch
import os
import sys
import json
import csv
import random
import numpy as np
import pandas as pd
import time
import torch
import gpboost as gpb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import optuna
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.logging import log_predictions
from utils.mem_usage import reduce_mem_usage

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

# Define constants and parameters
SUITE_ID = 336 # Regression on numerical features
print("Usage: python openml_gpboost_HP_single_run.py <task_idx> [run_seed]")
run_seed = 123 if len(sys.argv) <= 2 else int(sys.argv[2])
seed_everything(run_seed)
n_forecasts = 100
NUM_ROUNDS = 40

# Hardcoded parameters for testing
args = {
    "dataset": "Concrete Compression Strength",
    "n_splits": 20,
    "distn": "Normal",
    "verbose": True,
    "verbose_eval":1,
    "random_state": run_seed
}

method_name = 'GPboost'

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(SUITE_ID) 

def encode_categorical_series(y):
    # Check if the series is of type 'category' or 'object' (strings)
    if y.dtype.name == 'category' or y.dtype == 'object':
        y = y.astype('category').cat.codes  # Convert to category first, then encode
    return y

def encode_categorical_columns(df):
    # Iterate over columns that are either categorical or contain strings (objects)
    for col in df.select_dtypes(include=['category', 'object']).columns:
        df[col] = df[col].astype('category').cat.codes  # Convert to category first, then encode
    return df

# Define the Optuna objective class for hyperparameter tuning
class Objective(object):
    def __init__(self, X_train, y_train, coords_train, approx, seed):
        self.X_train = X_train
        self.y_train = y_train
        self.coords_train = coords_train
        self.approx = approx
        self.seed = seed
        
    def __call__(self, trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
            'max_depth': trial.suggest_categorical('max_depth', [-1]),  # Only -1 here, but can add more if desired            'num_leaves': trial.suggest_categorical('num_leaves', [2**i for i in range(1, 10)]),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 1),
            'seed': self.seed,
            'bagging_seed': self.seed,
            'feature_fraction_seed': self.seed,
            'data_random_seed': self.seed,
            'verbose': -1
            }
        delta_conv = trial.suggest_float('delta_rel_conv', 1e-4, 0.1)
        dtrain = gpb.Dataset(self.X_train, self.y_train)
        try:
            if self.approx:
                # Use gp_coords instead of group_data
                gp_model = gpb.GPModel(gp_coords=self.coords_train, likelihood="gaussian", gp_approx = "vecchia")
                gp_model.set_optim_params(params={"optimizer_cov": "nelder_mead"})
                params['num_neighbors'] = trial.suggest_int('num_neighbours', 10, 50, step=10)
                gp_model.set_optim_params(params={"delta_rel_conv": delta_conv})
                cv_results = gpb.cv(params, dtrain, gp_model=gp_model, num_boost_round=NUM_ROUNDS, nfold=5, early_stopping_rounds=10,  train_gp_model_cov_pars=False)
            else:
                # Use gp_coords instead of group_data
                gp_model = gpb.GPModel(gp_coords=self.coords_train, likelihood="gaussian")
                cv_results = gpb.cv(params, dtrain, gp_model=gp_model, num_boost_round=NUM_ROUNDS, nfold=5, early_stopping_rounds=10)

            return np.mean(cv_results['test_neg_log_likelihood-mean'])
        except Exception as e:
            print(f"Trial failed due to error: {e}")
            return float('inf')  # Return a high loss if an error is encountered

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    # Optimize memory usage
    X = reduce_mem_usage(X)
    print(f'Processing the dataset: {dataset.name}')
    
    # Encode categorical columns
    X = encode_categorical_columns(X)
    y = encode_categorical_series(y)

    lss_rmse, lss_nll, times, times_HP = [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    dset_name = dataset.name
    print(f"== Task ID={task_id} Dataset={dset_name} X.shape={str(X.shape)} {args['distn']}")

    if len(y) > 1000:
        approx_status = True
    else:
        approx_status = False

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X.iloc[train_indices], X.iloc[test_indices]
    y_train_opt, y_test_opt = y.iloc[train_indices], y.iloc[test_indices]

    # Standardize the input features (S) for Gaussian Process only on training data
    scaler = StandardScaler()
    coords_train_opt_scaled = scaler.fit_transform(X_train_opt)  # Fit and transform on training data

    train_opt_data = (X_train_opt.values, y_train_opt.values)

    # Hyperparameter optimization with Optuna
    start_time = time.time()
    print('Hyperparameter tuning...')
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=run_seed),
    )
    objective_tuning = Objective(X_train_opt, y_train_opt, coords_train_opt_scaled, approx_status, run_seed)
    study.optimize(objective_tuning, n_trials=20, timeout=86400)
    end_time = time.time()  # End time measurement
    elapsed_time_HP = end_time - start_time  # Calculate elapsed time
    print(f'Hyperparameter tuning time: {elapsed_time_HP:.2f} seconds')

    # Set the best parameters and number of estimators from hyperparameter tuning
    best_params = study.best_params
    print(f'Best hyperparameters for fold 0: {best_params}')

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        indices = np.arange(len(y_train))
        X_train_val, X_val, y_train_val, y_val, train_val_ind, val_ind = train_test_split(
            X_train,
            y_train,
            indices,
            test_size=0.2,
            random_state=run_seed + fold,
            shuffle=True,
        )
        # Standardize the input features (S) for Gaussian Process only on training data
        coords_train = scaler.transform(X_train_val)
        coords_val = scaler.transform(X_val)
        coords_test = scaler.transform(X_test)

        train_data = gpb.Dataset(X_train.values, y_train.values)
        train_val_data = gpb.Dataset(X_train_val.values, y_train_val.values)
        valid_data = gpb.Dataset(X_val.values, y_val.values)

        # Train the final model on the full training set (including validation)
        print('Training validation model...')
        fold_params = best_params.copy()
        fold_params.update({
            "seed": run_seed,
            "bagging_seed": run_seed,
            "feature_fraction_seed": run_seed,
            "data_random_seed": run_seed,
        })

        if approx_status:
            gp_model = gpb.GPModel(gp_coords=coords_train, likelihood="gaussian", gp_approx = "vecchia")
            gp_model.set_optim_params(params={"optimizer_cov": "nelder_mead"})
            gp_model.set_optim_params(params={"delta_rel_conv": fold_params['delta_rel_conv']})
        else:
            gp_model = gpb.GPModel(gp_coords=coords_train, likelihood="gaussian")
        eval_ind = val_ind
        # Use a valiation set for finding the optimal number of iterations
        gp_model.set_prediction_data(gp_coords_pred=coords_val)
        evals_result = {}  # record eval results for plotting
        if approx_status:
            st = gpb.train(params=fold_params, train_set=train_val_data, num_boost_round=NUM_ROUNDS,
                    gp_model=gp_model, valid_sets=valid_data, 
                    early_stopping_rounds=20, use_gp_model_for_validation=True,
                    evals_result=evals_result, train_gp_model_cov_pars=False)
        else:
            st = gpb.train(params=fold_params, train_set=train_val_data, num_boost_round=NUM_ROUNDS,
                    gp_model=gp_model, valid_sets=valid_data, 
                    early_stopping_rounds=20, use_gp_model_for_validation=True,
                    evals_result=evals_result)
        # Step 1: Extract the test_neg_log_likelihood list
        neg_log_likelihood_list = evals_result['valid_0']['test_neg_log_likelihood']

        # Step 2: Find the index of the minimum value in the list
        min_index = neg_log_likelihood_list.index(min(neg_log_likelihood_list))
        print("Best number of iterations: " + str(min_index + 1))
        best_iter = min_index + 1

        print('Training final model...')
        start_time = time.time()
        st_final = gpb.train(params=fold_params, train_set=train_val_data, num_boost_round=best_iter,
        gp_model=gp_model, use_gp_model_for_validation=False)
        training_time = time.time() - start_time
        print(f'Training time for fold {fold + 1}: {training_time:.2f} seconds')
        
        # Make predictions
        print('Prediction...')
        group_test = np.arange(len(y_test))
        pred = st_final.predict(X_test.values, gp_coords_pred=coords_test, predict_var=True, pred_latent=False)
        mu = pred['response_mean']
        var = pred['response_var']
        std = np.sqrt(var)

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(mu, y_test))
        nll_test = -norm(mu, std).logpdf(y_test).mean()

        rng = np.random.default_rng(run_seed + task_id * 1000 + fold)
        samples = np.array([[rng.normal(loc=loc, scale=np.std(scale), size=100) for loc, scale in zip(mu, std)]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test, samples)
        crps_test = crps_comps[0]
        crps_cal, crps_sha = crps_comps[1], crps_comps[2]

        # Store results
        lss_rmse.append(rmse)
        lss_nll.append(nll_test)
        lss_crps.append(crps_test)
        lss_crps_cal.append(crps_cal)
        lss_crps_sha.append(crps_sha)
        times += [training_time]

        # Define the quantiles to evaluate
        quantiles = [0.1, 0.5, 0.9]

        # Compute the quantiles for each observation
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = norm.ppf(q, loc=mu, scale=std)
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)

        # Log predictions for each fold
        log_predictions(fold, dset_name, y_test.values, mu, std, quantile_preds, f"logs/openml/predictions/{method_name}.csv")

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]

    print(f'Completed dataset: {dset_name}')
    # return a dictonary of val
    return  dset_name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), elapsed_time_HP, np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 



if __name__ == "__main__":
    print("gpboost")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    print("Task number: " + str(task_number))
    vsc_data = os.environ['VSC_DATA']
    results = run_single_argument(task_number)
    file_path = f"results/openml/openml_{method_name}.csv"
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

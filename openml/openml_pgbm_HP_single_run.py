import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from pgbm.torch import PGBMRegressor
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path
import optuna
from pgbm.torch import PGBM
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import norm
from properscoring._mean_crps import _mean_crps_hersbach

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = '0fc137c28db32cdfecb6347178c7be68'

# Define constants and parameters
SUITE_ID = 336 # Regression on numerical features
np.random.seed(1)
mode = 'exp'
natural_flag = False

# Hardcoded parameters for testing
args = {
    "dataset": "Concrete Compression Strength",
    "n_splits": 20,
    "distn": "Normal",
    "verbose": True,
    "verbose_eval":1,
    "random_state":1
}

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(SUITE_ID) 

def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
    return df

def objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()
    return loss

# Define the Optuna objective class for hyperparameter tuning
class Objective(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def __call__(self, trial):
        params = {
            'n_estimators': 200,
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.4),
            'max_leaves': trial.suggest_int('max_leaves', 20, 200),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # Constant for this example
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'device': 'gpu'
        }
        model = PGBMRegressor()
        model.set_params(**params)
        score = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=5, n_jobs=5, scoring='neg_root_mean_squared_error'))
        return score

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    # Encode categorical columns
    X = encode_categorical_columns(X)

    lss_rmse, lss_nll, times = [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['distn']}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X.iloc[train_indices], X.iloc[test_indices]
    y_train_opt, y_test_opt = y.iloc[train_indices], y.iloc[test_indices]

    train_opt_data = (X_train_opt, y_train_opt)

    # Hyperparameter optimization with Optuna
    start_time = time.time()
    print('Hyperparameter tuning...')
    study = optuna.create_study(direction='maximize')
    objective_tuning = Objective(X_train_opt, y_train_opt)
    study.optimize(objective_tuning, n_trials=20)
    end_time = time.time()  # End time measurement
    elapsed_time_HP = end_time - start_time  # Calculate elapsed time

    # Set the best parameters and number of estimators from hyperparameter tuning
    best_params = study.best_params
    print(f'Best hyperparameters for fold 0: {best_params}')

    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_trainall, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_trainall, y_test = y.iloc[train_indices], y.iloc[test_indices]

        X_train, X_val, y_train, y_val = train_test_split(X_trainall, y_trainall, test_size=0.2)

        train_data = (X_trainall.values, y_trainall.values)
        train_val_data = (X_train.values, y_train.values)
        valid_data = (X_val.values, y_val.values)

        # Train the final model on the full training set (including validation)
        print('Training validation model...')
        model = PGBM()
        model.train(train_val_data, objective=objective, metric=rmseloss_metric, valid_set=(X_val, y_val), params=best_params)
        best_params['n_estimators'] = model.best_iteration

        print('Training final model...')
        start_time = time.time()
        model = PGBM()
        model.train(train_data,  objective=objective, metric=rmseloss_metric, params=best_params)
        training_time = time.time() - start_time
        
        # Make predictions
        print('Prediction...')
        yhat_point, yhat_test_std = model.predict(X_test)
        yhat_dist, mu, var = model.predict_dist(X_test, n_forecasts=n_forecasts, parallel=False, output_sample_statistics=True)

        # Compute metrics
        rmse = rmseloss_metric(yhat_point.cpu(), y_test).numpy()
        crps_test = model.crps_ensemble(yhat_dist, y_test).mean().numpy()
        nll_test = -norm(mu, var).logpdf(y_test.flatten()).mean()
    
        yhat_dist = yhat_dist.reshape(yhat_dist.shape[1], yhat_dist.shape[0])
        crps_comps = crps(y_test.flatten(), yhat_dist)
        crps_cal, crps_sha = crps_comps[1], crps_comps[2]

        # Store results
        lss_rmse.append(rmse)
        lss_nll.append(nll_test)
        lss_crps.append(crps_test)
        lss_crps_cal.append(crps_cal)
        lss_crps_sha.append(crps_sha)
        times += [training_time]
        times_HP += [elapsed_time_HP]

    print(f'Completed dataset: {dataset.name}')
    # return a dictonary of val
    return  dataset.name, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP)



if __name__ == "__main__":
    print("PGBM")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    result = run_single_argument(task_number)
    vsc_data = os.environ['VSC_DATA']
    results = run_single_argument(sys.argv[1])
    file = open("logs/openml/PBGM_no_natural.csv", "a+")
    file.write(f"\n{results[0]}, {results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}, {results[6]}, {results[7]}, {results[8]}, {results[9]}, {results[10]}, {results[11]}")
    file.close()
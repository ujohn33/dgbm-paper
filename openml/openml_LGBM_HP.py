import openml
import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from optuna.integration import OptunaSearchCV
from sklearn.metrics import mean_pinball_loss
import optuna


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import quantile_loss

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = os.environ.get("OPENML_APIKEY", "")

# Define constants and parameters
MODEl_PATH = os.environ['VSC_SCRATCH'] + '/LSSboost/'
np.random.seed(1)
natural_flag = False

# Define constants and parameters
args = {
    "quantiles": [0.1, 0.5, 0.9],
    "SUITE_ID": 336, # Regression on numerical features
    "n_splits": 5,
    "n_trials": 100,
    "n_est": 2000,
}

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(args["SUITE_ID"])  # obtain the benchmark suite

# Define hyperparameter search space
param_distributions = {
    "eta": optuna.distributions.FloatDistribution(1e-5, 0.4, log=True),
    "max_depth": optuna.distributions.IntDistribution(2, 10),
    "num_leaves": optuna.distributions.IntDistribution(20, 200),
    "min_data_in_leaf": optuna.distributions.IntDistribution(20, 100),
    "bagging_fraction": optuna.distributions.FloatDistribution(0.5, 1, log=False),
    "feature_pre_filter": optuna.distributions.CategoricalDistribution([False])
}

def encode_categorical_series(y):
    # Check if the series is of type 'category' or 'object' (strings)
    if y.dtype.name == 'category' or y.dtype == 'object':
        y = y.astype('category').cat.codes  # Convert to category first, then encode
    return y

def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
    return df

def load_data_from_openml(task_id):
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    # Encode categorical columns
    X = encode_categorical_columns(X)
    
    return X, y, dataset.name

def run_single_argument(task_id, quantiles=[0.1, 0.5, 0.9]):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    dset_name = dataset.name

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    print(f'Processing the dataset: {dataset.name}')
    
    times = []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")

    # Perform hyperparameter optimization on the first fold
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train_opt, X_test_opt = X.iloc[train_indices], X.iloc[test_indices]
    y_train_opt, y_test_opt = y.iloc[train_indices], y.iloc[test_indices]

    dtrain = lgb.Dataset(X_train_opt, y_train_opt)

    start_time = time.time()  # Start time measurement

    # Define the LightGBM regressor
    estimator = lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=1, n_estimators=args["n_est"])

    # Define cross-validation
    cv = KFold(n_splits=args['n_splits'], shuffle=True, random_state=1)

    # Use OptunaSearchCV for hyperparameter optimization
    optuna_search = OptunaSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        cv=cv,
        n_trials=args['n_trials'],
        refit=True,
        random_state=1,
        verbose=1
    )

    # Fit OptunaSearchCV with cross-validation
    optuna_search.fit(X_train_opt, y_train_opt)

    time_HP = time.time() - start_time

    # Get the best parameters
    best_params = optuna_search.best_params_
    print(f"Best parameters: {best_params}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")
    
    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_train_fold, X_test_fold = X.iloc[train_indices], X.iloc[test_indices]
        y_train_fold, y_test_fold = y.iloc[train_indices], y.iloc[test_indices]

        quantile_models = {}
        quantile_preds = {}
        quantile_losses = []
        
        runtime_start = time.time()

        # Train separate models for each quantile (0.1, 0.5, 0.9)
        for count, q in enumerate(args["quantiles"]):
            model = lgb.LGBMRegressor(**best_params, objective="quantile", alpha=q, random_state=1)
            model.fit(X_train_fold, y_train_fold)
            quantile_models[q] = model

        runtime_pred = time.time() - runtime_start

        for q in args["quantiles"]:
            q_pred = quantile_models[q].predict(X_test_fold)
            q_loss = quantile_loss(q, y_test_fold, q_pred).mean()
            if q == 0.1:
                quantile_losses.append(q_loss)
            elif q == 0.5:
                quantile_losses.append(q_loss)
            else:
                quantile_losses.append(q_loss)

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)
        print(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]
        times += [runtime_pred]

        # wql_01 = np.asarray(wql_01)
        # wql_05 = np.asarray(wql_05)
        # wql_09 = np.asarray(wql_09)
        # wql_avg = np.asarray(wql_avg)
        print(wql_01)
    
    print(task_id)
    print(dataset.name)
    
    return dset_name, np.mean(times), np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 


if __name__ == "__main__":
    print("LGBM quantiles")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    results = run_single_argument(task_number)

    file_path = "logs/openml/openml_lgbm.csv"
    header = ["dset","time_run","WQL01-mean", "WQL01-std","WQL05-mean", "WQL05-std","WQL09-mean", "WQL09-std", "WQL_avg-mean", "WQL_avg-std"]
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
                        results[5], results[6], results[7], results[8], results[9]]

        writer.writerow(row_to_write)
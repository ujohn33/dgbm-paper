import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from scipy.stats import norm

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = '0fc137c28db32cdfecb6347178c7be68'

# Define constants and parameters
SUITE_ID = 336 # Regression on numerical features
np.random.seed(1)
mode = 'exp'
natural_flag = True
args = {
    "reps": 3,
    "n_est": 2000,
    "n_splits": 20,
    "score": "MLE",
    "distn": "Normal",
    "base": "tree",
    "verbose": True,
}

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

# Define your hyperparameter space
param_dict = {
    "eta": ["float", {"low": 0.001, "high": 0.4, "log": True}],
    "max_depth": ["int", {"low": 2, "high": 15, "log": False}],
    "min_child_weight": ["int", {"low": 1, "high": 15, "log": False}],
}

def encode_categorical_columns(df):
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
    return df

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    # Encode categorical columns
    X = encode_categorical_columns(X)

    y_true, lss_rmse, lss_nll, times = [], [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    start_time = time.time()  # Start time measurement

    xgblss = XGBoostLSS(Gaussian(stabilization="None", response_fn="exp", loss_fn="nll", natural_gradient=True))
    # Modify start values     
    xgblss.start_values = np.array([np.array(0.5) for _ in range(xgblss.dist.n_dist_param)])

    # Perform hyperparameter optimization
    np.random.seed(123)
    dtrain = xgb.DMatrix(X, label=y)

    opt_param = xgblss.hyper_opt(param_dict, dtrain, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=20, max_minutes=3000, n_trials=80,
                                    silence=True, seed=1, hp_seed=1)

    print(opt_param)
    opt_params = opt_param.copy()
    if natural_flag == True:
        # Save the optimized parameters
        with open(f'logs/openml/xgboost/natural/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    else:
        # Save the optimized parameters
        with open(f'logs/openml/xgboost/normal/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    
    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]

    # Time measurement
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    
    print(f"Optimization completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    results = []
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    result = run_single_argument(task_number)

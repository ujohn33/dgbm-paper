import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
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
    "eta": ["float", {"low": 0.025, "high": 0.025, "log": False}],
    "max_depth": ["int", {"low": 2, "high": 3, "log": False}],
    "num_leaves": ["int", {"low": 20, "high": 200, "log": False}],  # Constant for this example
    "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "feature_pre_filter": ["categorical", [False]]
}

def run_single_argument(task_id):
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    y_true, lss_rmse, lss_nll, times = [], [], [], []

    print(f"== Task ID={task_id} Dataset={dataset.name} X.shape={str(X.shape)} {args['score']}/{args['distn']}")

    start_time = time.time()  # Start time measurement

    lgblss = LightGBMLSS(Gaussian(stabilization="None", response_fn="exp", loss_fn="nll", natural_gradient=True))
    # Modify start values     
    lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])

    # Perform hyperparameter optimization
    np.random.seed(123)
    dtrain = lgb.Dataset(X, y)

    opt_param = lgblss.hyper_opt(param_dict, dtrain, num_boost_round=args["n_est"],
                                    nfold=args['n_splits'], early_stopping_rounds=20, max_minutes=3000, n_trials=80,
                                    silence=True, seed=1, hp_seed=1)

    print(opt_param)
    opt_params = opt_param.copy()
    if natural_flag == True:
        # Save the optimized parameters
        with open(f'logs/openml/natural/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    else:
        # Save the optimized parameters
        with open(f'logs/openml/normal/exp/{dataset.name}_opt_params.json', 'w') as f:
            json.dump(opt_params, f)
    
    n_rounds = opt_params["opt_rounds"]
    del opt_params["opt_rounds"]


if __name__ == "__main__":
    results = []
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    result = run_single_argument(task_number)

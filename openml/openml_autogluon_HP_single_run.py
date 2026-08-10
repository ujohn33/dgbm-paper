import openml
import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import time
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import quantile_loss

np.random.seed(123)

# Set OpenML API key
openml.config.apikey = os.environ.get("OPENML_APIKEY", "")

# Define constants and parameters
SUITE_ID = 336 # Regression on numerical features
MODEl_PATH = os.environ['VSC_SCRATCH'] + '/LSSboost/'
np.random.seed(1)
mode = 'exp'
natural_flag = False
n_forecasts = 100
distn = "Normal"

# Obtain the benchmark suite from OpenML
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

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
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    print(f'Processing the dataset: {dataset.name}')
    
    # Encode categorical columns
    X = encode_categorical_columns(X)
    y = encode_categorical_series(y)

    # Merge X and y into a single DataFrame, required for AutoGluon
    X['target'] = y  # Ensure 'target' is the string label name
    
    times, times_HP = [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []

    dset_name = dataset.name
    print(f"== Task ID={task_id} Dataset={dset_name} X.shape={str(X.shape)}")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"Task {task_id}: number of repeats: {n_repeats}, number of folds: {n_folds}, number of samples {n_samples}.")
    
    # Evaluate the optimized parameters on the remaining folds
    for fold in range(1, n_folds):
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
        runtime_start = time.time()
        # Convert X_train to a TabularDataset
        train_data = TabularDataset(X_train)
        test_data = TabularDataset(X_test)
        test_data_nolabel = test_data.drop(columns=['target'])
        
        # Use TabularPredictor to fit the model
        predictor = TabularPredictor(label='target', problem_type='quantile', eval_metric='pinball', quantile_levels=quantiles).fit(train_data, presets='high_quality', save_bag_folds=False, save_space=True)

        runtime_pred = time.time() - runtime_start

        predictions = predictor.predict(test_data_nolabel)
        predictions = predictions.to_numpy()

        # Compute the quantiles for each observation
        quantile_preds = {}
        quantile_losses = []

        for idx, q in enumerate(quantiles):
            quantile_preds[str(q)] = predictions[:, idx]
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]
        times += [runtime_pred]
    
    print(task_id)
    print(dataset.name)
    
    return dset_name, np.mean(times), np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 


if __name__ == "__main__":
    print("AUTOGLUON")
    print("______________________")
    task_number = benchmark_suite.tasks[int(sys.argv[1])]
    results = run_single_argument(task_number)

    file_path = "results/openml/openml_autogluon.csv"
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
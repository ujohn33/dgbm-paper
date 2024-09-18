import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import norm
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss


np.random.seed(123)

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
    "n_splits": 20,
    "distn": "Normal",
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

def run_single_argument(run_seed, quantiles=[0.1, 0.5, 0.9]):
    dset = dataset_list[int(run_seed)]
    args["dataset"] = dset
    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
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

    if args["dataset"] == "Year Prediciton MSD":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
    elif args["dataset"] == "Protein Structure":
        kf = KFold(n_splits=5)
        folds = kf.split(X)
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(1)
        folds = []
        for i in range(5):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))        
    else:
        kf = KFold(n_splits=args["n_splits"])
        folds = kf.split(X)
        # Follow https://github.com/yaringal/DropoutUncertaintyExps/blob/master/UCI_Datasets/concrete/data/split_data_train_test.py
        n = X.shape[0]
        np.random.seed(1)
        folds = []
        for i in range(args['n_splits']):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))

    for itr, (train_index, test_index) in enumerate(folds):
        print(f'{dset}: fold {itr + 1}/{len(folds)}')
        #X_train, X_test, y_train, y_test = get_fold(dataset_name, data, fold)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        runtime_start = time.time()
        # Convert X_train to a TabularDataset
        train_data = TabularDataset(X_train)
        test_data = TabularDataset(X_test)
        test_data_nolabel = test_data.drop(columns=[label])
        
        # Use TabularPredictor to fit the model
        predictor = TabularPredictor(label='target', problem_type='quantile', eval_metric='pinball', quantile_levels=quantiles).fit(train_data, presets='high_quality')

        runtime_pred = time.time() - runtime_start

        predictions = predictor.predict(test_data_nolabel)
        predictions = predictions.to_numpy()

        # Compute the quantiles for each observation
        quantile_preds = {}

        for idx, q in enumerate(quantiles):
            quantile_preds[q] = predictions[:, idx]
            q_loss = quantile_loss(q, y_test, quantile_preds[q]).mean()
            quantile_losses.append(q_loss)

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]
    
    print(task_id)
    print(dataset.name)
    
    return dset_name, np.mean(times), p.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 


if __name__ == "__main__":
    print("AUTOGLUON")
    print("______________________")
    results = run_single_argument(sys.argv[1])

    file_path = "logs/openml/openml_autogluon.csv"
    header = ["dset","time_run","WQL01-mean", "WQL01-std","WQL05-mean", "WQL05-std","WQL09-mean", "WQL09-std", "WQL_avg-mean", "WQL_avg-std"]
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    # Open the file in append mode ('a+')
    with open(file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)

        # If the file does not exist or is empty, write the header
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(header)  # Write header

        row_to_write = f"\n{results[0]}, {results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}, {results[6]}, {results[7]}, {results[8]}, {results[9]}"
        # Write the results to the file
        writer.writerow(row_to_write)

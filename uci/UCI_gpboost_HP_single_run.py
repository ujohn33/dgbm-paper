import openml
import os
import sys
import json
import numpy as np
import pandas as pd
import time
import torch
import gpboost as gpb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from pathlib import Path
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import norm
from properscoring._mean_crps import _mean_crps_hersbach

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps

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
n_forecasts = 1000

# Hardcoded parameters for testing
args = {
    "dataset": "Concrete Compression Strength",
    "n_splits": 20,
    "distn": "Normal",
    "verbose": True,
    "random_state":1
}

# Define objective function for GPBoost
def gpboost_objective(X_train, y_train, trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
        'max_depth': trial.suggest_int('max_depth', 1, 6),
        'num_leaves': trial.suggest_int('num_leaves', 2, 64),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 1),
        'verbose': -1
    }
    
    gp_model = gpb.GPModel(group_data=np.arange(len(y_train)), likelihood="gaussian")
    dtrain = gpb.Dataset(X_train, y_train)
    
    cv_results = gpb.cv(params, dtrain, gp_model=gp_model, num_boost_round=2000, nfold=5, early_stopping_rounds=10)
    #print(cv_results)
    return np.mean(cv_results['test_neg_log_likelihood-mean'])

def run_single_argument(run_seed):
    dset = dataset_list[int(run_seed)]
    args["dataset"] = dset
    y_true, lss_rmse, lss_nll, times, times_HP = [], [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={args['dataset']} X.shape={str(X.shape)} {args['distn']}")
    lgbm_rmse = []
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
        # default_params = {
        #     "max_depth":                9,
        #     "num_leaves":               110,
        #     "min_data_in_leaf":         22,
        #     "subsample":                1,
        # }
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
        indices = np.arange(len(y_train))
        X_train_val, X_val, y_train_val, y_val, train_val_ind, val_ind = train_test_split(X_train, y_train, indices, test_size=0.2)
        group = np.arange(len(y_train))

        # Hyperparameter optimization with Optuna
        start_time = time.time()
        print('Hyperparameter tuning...')
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: gpboost_objective(X_train, y_train, trial), n_trials=20)
        end_time = time.time()  # End time measurement
        elapsed_time_HP = end_time - start_time  # Calculate elapsed time

        # Set the best parameters and number of estimators from hyperparameter tuning
        best_params = study.best_params
        print(f'Best hyperparameters for fold {itr + 1}: {best_params}')

        gp_model = gpb.GPModel(group_data=group[train_val_ind], likelihood="gaussian")
        dtrain = gpb.Dataset(X_train, y_train)
        dtrain_val = gpb.Dataset(X_train_val, y_train_val)
        deval = gpb.Dataset(X_val, y_val, reference=dtrain_val)

        eval_ind = val_ind
        # Use a valiation set for finding the optimal number of iterations
        gp_model.set_prediction_data(group_data_pred=group[eval_ind])
        evals_result = {}  # record eval results for plotting
        st = gpb.train(params=best_params, train_set=dtrain_val, num_boost_round=2000,
                gp_model=gp_model, valid_sets=deval, 
                early_stopping_rounds=20, use_gp_model_for_validation=True,
                evals_result=evals_result)
        #print(evals_result)
        # Step 1: Extract the test_neg_log_likelihood list
        neg_log_likelihood_list = evals_result['valid_0']['test_neg_log_likelihood']

        # Step 2: Find the index of the minimum value in the list
        min_index = neg_log_likelihood_list.index(min(neg_log_likelihood_list))
        print("Best number of iterations: " + str(min_index + 1))
        best_iter = min_index + 1

        # Train the final model configuration
        print('Training final model...')
        start_time = time.time()
        st_final = gpb.train(params=best_params, train_set=dtrain_val, num_boost_round=best_iter,
            gp_model=gp_model, use_gp_model_for_validation=True)
        training_time = time.time() - start_time
        print(f'Training time for fold {itr + 1}: {training_time:.2f} seconds')

        # Make predictions
        print('Prediction...')
        group_test = np.arange(len(y_test))
        pred = st_final.predict(X_test, group_data_pred=group_test, predict_var=True, pred_latent=False)
        mu = pred['response_mean']
        var = pred['response_var']

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(mu, y_test))
        nll_test = -norm(mu, var).logpdf(y_test.flatten()).mean()

        samples = np.array([[np.random.normal(loc=loc, scale=scale, size=100) for loc, scale in zip(mu, var)]])
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test.flatten(), samples)

        # Store results
        lss_rmse.append(rmse)
        lss_nll.append(nll_test)
        lss_crps.append(crps_comps[0])
        lss_crps_cal.append(crps_comps[1])
        lss_crps_sha.append(crps_comps[2])
        times += [training_time]
        times_HP += [elapsed_time_HP]

    print(f'Completed dataset: {dset}')
    # return a dictonary of val
    return  dset, np.mean(lss_rmse), np.std(lss_rmse), np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP)



if __name__ == "__main__":
    vsc_data = os.environ['VSC_DATA']
    results = run_single_argument(sys.argv[1])
    file = open("logs/uci/gpboost.csv", "a+")
    file.write(f"\n{results[0]}, {results[1]}, {results[2]}, {results[3]}, {results[4]}, {results[5]}, {results[6]}, {results[7]}, {results[8]}, {results[9]}, {results[10]}, {results[11]}, {results[12]}")
    file.close()


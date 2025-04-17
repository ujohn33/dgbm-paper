import os
import sys
import json
import numpy as np
import pandas as pd
import time
import csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
from lightgbmlss.model import *
from lightgbmlss.distributions.distribution_utils import DistributionClass
from lightgbmlss.distributions import *
# Import specific distribution classes
from xgboostlss.distributions.Gaussian import Gaussian
from xgboostlss.distributions.StudentT import StudentT
from xgboostlss.distributions.Gamma import Gamma
from xgboostlss.distributions.LogNormal import LogNormal
from xgboostlss.distributions.Weibull import Weibull
from xgboostlss.distributions.Gumbel import Gumbel
from xgboostlss.distributions.Laplace import Laplace

from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.logging import log_predictions
from utils.safety import apply_safety_net 

np.random.seed(123)

print("Usage: python UCI_lssboost_autoselect_conditional_HP_single_run.py <seed_id> <mode> <natural_grad> <stabilization> <clip_value> <standardize>")

mode = sys.argv[2]  # e.g., 'exp'
natural_grad = sys.argv[3].lower() == 'true'  # Convert 'True' or 'False' to boolean
stabilization = sys.argv[4]  # e.g., 'L2', 'MAD', or 'None'
clip_value = None if len(sys.argv) <= 5 or sys.argv[5] == 'None' else float(sys.argv[5])
# If standardize not provided, default to False
standardize = False if len(sys.argv) <= 6 else sys.argv[6].lower() == 'true'
    
if natural_grad:
    method_name = f'LSSboost_conditional_natural_{mode}_{stabilization}_clip_{clip_value}_std_{standardize}'
else:
    method_name = f'LSSboost_conditional_no_natural_{mode}_{stabilization}_clip_{clip_value}_std_{standardize}'

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
    "mode": mode,
    "natural_grad": natural_grad,
    "stabilization": stabilization, # None, 'L2', "MAD"  
    "clip_value": clip_value,
    "n_est": 200,
    "n_splits": 20,
    "score": "MLE",
    "distn": "conditional",  # Using conditional for the feature-based selection
    "standardize": standardize,
    "random_state": 1,
}

# Define your hyperparameter space
param_dict = {
    "eta": ["float", {"low": 1e-5, "high": 0.4, "log": True}],
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "num_leaves": ["int", {"low": 20, "high": 100, "log": False}],
    "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],
    "feature_pre_filter": ["categorical", [False]],
}

def feature_conditional_dist_select(X_train, y_train, candidate_distributions, cv=5, **kwargs):
    """
    Select best distribution based on predictive performance with features.
    
    Args:
        X_train: Feature matrix
        y_train: Target vector
        candidate_distributions: List of distribution classes to evaluate
        cv: Number of cross-validation folds
        **kwargs: Additional parameters for LightGBMLSS model
    
    Returns:
        best_dist: Best performing distribution class
        cv_results: DataFrame with cross-validation results
    """
    # Default parameters for quick training
    default_params = {
        'num_boost_round': 50,  # Reduced for speed
        'early_stopping_rounds': 10,
        'learning_rate': 0.1,
        'max_depth': 3
    }
    
    # Update with any provided kwargs
    model_params = {**default_params, **kwargs}
    
    results = []
    
    # Setup cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for dist_class in candidate_distributions:
        dist_name = dist_class.__name__.split(".")[-1]
        print(f"Evaluating {dist_name}...")
        
        fold_scores = []
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Create distribution instance
            dist = dist_class(stabilization="None", response_fn="exp", loss_fn="nll")
            
            try:
                # Create validation data
                dtrain = lgb.Dataset(X_fold_train, y_fold_train)
                deval = lgb.Dataset(X_fold_val, y_fold_val)
                
                # Create model
                model = LightGBMLSS(dist)
                
                # Set start values
                model.start_values = np.array([np.array(0.5) for _ in range(model.dist.n_dist_param)])
                
                # Training with early stopping
                gbm = model.train(default_params, dtrain, 
                                  num_boost_round=default_params["num_boost_round"],
                                  valid_sets=[dtrain, deval],
                                  early_stopping_rounds=default_params["early_stopping_rounds"])
                
                # Predict
                pred = model.predict(X_fold_val)
                
                # Calculate NLL
                nll = model.dist.logpdf(pred, y_fold_val)
                
                fold_scores.append({
                    'fold': fold,
                    'nll': nll,
                })
            except Exception as e:
                print(f"Error fitting {dist_name} on fold {fold}: {str(e)}")
                fold_scores.append({
                    'fold': fold,
                    'nll': float('inf'), 
                })
        
        # Average scores across folds
        avg_nll = np.mean([s['nll'] for s in fold_scores if s['nll'] != float('inf')])
        
        results.append({
            'distribution': dist_name,
            'avg_nll': avg_nll if not np.isnan(avg_nll) else float('inf'),
            'fold_scores': fold_scores
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by NLL (lower is better)
    results_df = results_df.sort_values('avg_nll')
    
    # Select best distribution
    best_dist_name = results_df.iloc[0]['distribution'] if not results_df.empty else candidate_distributions[0].__name__
    best_dist = next(d for d in candidate_distributions if d.__name__ == best_dist_name)
    
    return best_dist, results_df

def run_single_arguement(run_seed):
    dset = dataset_list[int(run_seed)]
    args["dataset"] = dset
    y_true, lss_nll, times, times_HP = [], [], [], []
    lss_crps, lss_crps_cal, lss_crps_sha = [], [], []
    wql_01, wql_05, wql_09, wql_avg = [], [], [], []
    selected_distributions = []  # Add this line to track selected distributions

    # Load dataset -- use last column as labela
    data = dataset_name_to_loader[args['dataset']]()
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    print(f"== Dataset={dset} X.shape={str(X.shape)} {args['score']}/{args['distn']} Standardize={args['standardize']}")
    if dset == "Year Prediciton MSD":
        folds = [(np.arange(463715), np.arange(463715, len(X)))]
    elif dset == "Protein Structure":
        n = X.shape[0]
        np.random.seed(args['random_state'])
        folds = []
        for i in range(5):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))      
    else:
        n = X.shape[0]
        np.random.seed(args['random_state'])
        folds = []
        for i in range(args['n_splits']):
            permutation = np.random.choice(range(n), n, replace=False)
            end_train = round(n * 9.0 / 10)
            end_test = n

            train_index = permutation[0:end_train]
            test_index = permutation[end_train:n]
            folds.append((train_index, test_index))

    for itr, (train_index, test_index) in enumerate(folds):
        start_time = time.time()
        X_trainall, X_test = X[train_index], X[test_index]
        y_trainall, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainall, y_trainall, test_size=0.2, random_state=args['random_state']
        )

        # Apply standardization based on the parameter
        if args['standardize'] or dset == "Year Prediciton MSD":
            y_mean = np.mean(y_trainall)
            y_std = np.std(y_trainall)
            y_trainall = (y_trainall - y_mean) / y_std
            y_train = (y_train - y_mean) / y_std
            y_val = (y_val - y_mean) / y_std
            y_test = (y_test - y_mean) / y_std
        else:
            pass

        # Conditional distribution selection
        if args['distn'] == "conditional":
            print(f"Selecting best distribution for fold {itr+1} using feature-based approach...")
            candidate_distributions = [Gaussian, StudentT, Gamma, LogNormal, Weibull, Gumbel, Laplace]
            
            best_dist, dist_fit_df = feature_conditional_dist_select(
                X_train, y_train, 
                candidate_distributions,
                cv=3,  # Use 3-fold CV for speed
                num_boost_round=50,
                early_stopping_rounds=10
            )
            
            best_dist_name = best_dist.__name__
            selected_distributions.append(best_dist_name)
            print(f"Selected distribution: {best_dist_name} with NLL: {dist_fit_df.iloc[0]['avg_nll']}")
            
            # Create distribution instance with the desired parameters
            distribution = best_dist(
                stabilization=args['stabilization'],
                response_fn=args['mode'],
                loss_fn="nll",
                natural_gradient=args['natural_grad'],
                clip_value=args['clip_value']
            )
        else:
            # Default to Gaussian if not using conditional selection
            distribution = Gaussian(
                stabilization=args['stabilization'],
                response_fn=args['mode'],
                loss_fn="nll",
                natural_gradient=args['natural_grad'],
                clip_value=args['clip_value']
            )
            selected_distributions.append("Gaussian")

        full_train_data = lgb.Dataset(X_trainall, y_trainall)

        start_time = time.time()
        lgblss = LightGBMLSS(distribution)
        
        # Modify start values     
        lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])
        
        # Modify parameter dictionary for Year Prediciton MSD dataset
        current_param_dict = param_dict.copy()
        if dset == "Year Prediciton MSD":
            current_param_dict["bagging_fraction"] = ["categorical", [0.1]]
        else:
            current_param_dict = param_dict

        opt_param = lgblss.hyper_opt(current_param_dict, full_train_data, num_boost_round=args["n_est"],
                                    nfold=5, early_stopping_rounds=20, max_minutes=80, n_trials=20,
                                    silence=True, seed=args['random_state'], hp_seed=args['random_state'])
        opt_params = opt_param.copy()

        end_time = time.time()
        elapsed_time_HP = end_time - start_time

        dtrain = lgb.Dataset(X_train, y_train)
        deval = lgb.Dataset(X_val, y_val)
        
        # Training with early stopping
        evals_result = {}
        opt_params['early_stopping'] = 20
        
        # Train Model with optimized hyperparameters
        gbm = lgblss.train(opt_params, dtrain, 
                           num_boost_round=args["n_est"],
                           valid_sets=[dtrain, deval])

        # Best iteration
        print(f"Best iteration: {lgblss.booster.best_iteration}")

        opt_params['early_stopping'] = None
        best_iter = lgblss.booster.best_iteration

        # Train final model on all training data
        start_time = time.time()
        final_gbm = lgblss.train(opt_params, full_train_data, 
                                num_boost_round=best_iter)
        
        # Make predictions
        forecast = lgblss.predict(X_test)

        print(f"Raw Predictions - min: {forecast[lgblss.dist.params[0]].min()}, max: {forecast[lgblss.dist.params[0]].max()}, mean: {forecast[lgblss.dist.params[0]].mean()}")

        # Handle rescaling for standardized data
        if args['standardize'] or dset == "Year Prediciton MSD":
            for param in lgblss.dist.params:
                if param == lgblss.dist.params[0]:  # location parameter
                    forecast[param] = forecast[param] * y_std + y_mean
                if param == lgblss.dist.params[1]:  # scale parameter
                    forecast[param] = forecast[param] * y_std
            y_test = y_test * y_std + y_mean
            y_trainall = y_trainall * y_std + y_mean
        else:
            pass

        print(f"Predictions after rescaling - min: {forecast[lgblss.dist.params[0]].min()}, max: {forecast[lgblss.dist.params[0]].max()}, mean: {forecast[lgblss.dist.params[0]].mean()}")

        forecast_val = lgblss.predict(X_val)
        
        # Time the duration for forecast deployment
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate metrics - adapt for different distributions
        loc_param = lgblss.dist.params[0]
        scale_param = lgblss.dist.params[1] if len(lgblss.dist.params) > 1 else None
        
        # Calculate NLL using the distribution's logpdf method
        lss_nll += [lgblss.dist.logpdf(forecast, y_test)]
        
        # Generate samples and calculate CRPS
        samples = lgblss.dist.sample(forecast, 100)
        samples = samples.reshape(samples.shape[1], samples.shape[2])
        crps_comps = crps(y_test.flatten(), samples)
        lss_crps += [crps_comps[0]]
        lss_crps_cal += [crps_comps[1]]
        lss_crps_sha += [crps_comps[2]]
        times += [elapsed_time]
        times_HP += [elapsed_time_HP]

        # Define the quantiles to evaluate
        quantiles = [0.1, 0.5, 0.9]

        # Compute the quantiles for each observation using the distribution's ppf
        quantile_preds = {}
        quantile_losses = []
        for q in quantiles:
            quantile_preds[str(q)] = lgblss.dist.ppf(q, forecast)
            q_loss = quantile_loss(q, y_test, quantile_preds[str(q)]).mean()
            quantile_losses.append(q_loss)

        # Log predictions for each fold
        log_predictions(itr, dset, y_test, forecast[loc_param], 
                        forecast[scale_param] if scale_param else None, 
                        quantile_preds, f"logs/uci/predictions/{method_name}.csv")

        # Compute the average of the quantile losses (WQL as an average)
        wql_avg_fold = np.mean(quantile_losses)

        wql_01 += [quantile_losses[0]]
        wql_05 += [quantile_losses[1]]
        wql_09 += [quantile_losses[2]]
        wql_avg += [wql_avg_fold]

        print(
                "[%d/%d] BestIter=%d NLL: Test=%.4f CRPS=%.4f CRPS_CAL=%.4f CRPS_SHA=%.4f TIME=%.4f"
                % (
                    itr + 1,
                    args['n_splits'],
                    best_iter,
                    lss_nll[-1],
                    lss_crps[-1],
                    lss_crps_cal[-1],
                    lss_crps_sha[-1],
                    elapsed_time,
                )
            )

    # Print summary of selected distributions
    dist_counts = {}
    for dist in selected_distributions:
        dist_counts[dist] = dist_counts.get(dist, 0) + 1
    print("Selected distributions across folds:")
    for dist, count in dist_counts.items():
        print(f"- {dist}: {count} folds ({count/len(selected_distributions)*100:.1f}%)")

    print(dset)
    print(
            "== NLL GBMLSS=%.4f ± %.4f, CRPS = %.4f  +/- %.4f, CRPS_cal =  %.4f +/- %.4f, CRPS_sha =  %.4f +/- %.4f,  TIME = %.4f"
            % (
                np.mean(lss_nll),
                np.std(lss_nll),
                np.mean(lss_crps),
                np.std(lss_crps),
                np.mean(lss_crps_cal),
                np.std(lss_crps_cal),
                np.mean(lss_crps_sha),
                np.std(lss_crps_sha),
                np.mean(times)
            )
        )
    # return result values
    return  dset, np.mean(lss_nll), np.std(lss_nll), np.mean(lss_crps), np.std(lss_crps), np.mean(lss_crps_cal), np.std(lss_crps_cal), np.mean(lss_crps_sha), np.std(lss_crps_sha), np.mean(times), np.mean(times_HP), np.mean(wql_01), np.std(wql_01), np.mean(wql_05), np.std(wql_05), np.mean(wql_09), np.std(wql_09), np.mean(wql_avg), np.std(wql_avg) 

if __name__ == "__main__":
    vsc_data = os.environ['VSC_DATA']
    results = run_single_arguement(sys.argv[1])
    method_name = f"{method_name}_n_est_{args['n_est']}"  # Assuming args['n_est'] exists
    file_path = f"results/uci/uci_{method_name}.csv"
    header = ["dset","NLL-mean","NLL-std","CRPS-mean","CRPS-std","CRPS-calibration-mean","CRPS-calibration-std","CRPS-sharpness-mean","CRPS-sharpness-std","time_run","time_HP","WQL01-mean", "WQL01-std","WQL05-mean", "WQL05-std","WQL09-mean", "WQL09-std", "WQL_avg-mean", "WQL_avg-std"]
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    # saving the results
    print(f"Saving results to {file_path}")
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
                        results[14], results[15], results[16], results[17]]

        writer.writerow(row_to_write)

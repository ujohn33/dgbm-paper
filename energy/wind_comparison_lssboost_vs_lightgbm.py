import os, sys, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import torch
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from lightgbmlss.distributions.NegativeBinomial import *
from lightgbmlss.distributions.Poisson import *
from lightgbmlss.distributions.ZINB import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.safety import apply_safety_net

np.random.seed(123)

NUM_BOOST_ROUNDS = 200

# Get parameters from command line
if len(sys.argv) < 2:
    print("Usage: python wind_comparison_lssboost_vs_lightgbm.py <model_nwp> [mode] [stabilization] [dist]")
    sys.exit(1)

def detect_categorical_features(df, threshold_unique=20, threshold_ratio=0.05):
    """
    Detect likely categorical features in a DataFrame.
    Parameters:
    - df: pandas DataFrame
    - threshold_unique: maximum number of unique values to consider a feature categorical
    - threshold_ratio: maximum ratio of unique values to total samples to consider categorical
    Returns:
    - List of column names likely to be categorical
    """
    categorical_cols = []
    for col in df.columns:
        num_unique = df[col].nunique()
        total_samples = len(df[col])
        if pd.api.types.is_integer_dtype(df[col]):
            if (num_unique <= threshold_unique) or (num_unique / total_samples <= threshold_ratio):
                categorical_cols.append(col)
        elif (pd.api.types.is_object_dtype(df[col]) or 
              pd.api.types.is_categorical_dtype(df[col]) or 
              pd.api.types.is_string_dtype(df[col])):
            categorical_cols.append(col)
    return categorical_cols

# Define training periods in hours
training_periods = {
    "1_day": 24,
    "3_days": 72,
    "1_week": 168,
    "2_weeks": 336,
    "1_month": 720,
    "3_months": 2160,
    "6_months": 4320,
    "1_year": 8760,
}

# Parse arguments
model_nwp = sys.argv[1] if len(sys.argv) > 1 else "icon_eu"
mode = sys.argv[2] if len(sys.argv) > 2 else "exp"
stabilization = sys.argv[3] if len(sys.argv) > 3 else 'None'
dist = sys.argv[4] if len(sys.argv) > 4 else "Gaussian"

# today's date
today = pd.to_datetime("today").strftime("%Y-%m-%d")
method_name = f'{mode}_{stabilization}_{dist}_{today}'

# Log received parameters
print(f"Parameters: weather model={model_nwp}, mode={mode}, stabilization={stabilization}, dist={dist}")

# Path to data
data_path = f"/scratch/brussel/105/vsc10528/WP_predico_project/data/training_data_wind_iconeu_ecmwfifs025_gfsseamless_2clusters_with_averages_30may2025.csv"

# Load data
print(f"Loading data from {data_path}")
df = pd.read_csv(data_path)

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort by datetime to ensure temporal order
df = df.sort_values('datetime')

# Extract time-based features
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Check for categorical features
detected_cat_features = detect_categorical_features(df, threshold_unique=20, threshold_ratio=0.05)
cat_features = ['hour', 'day_of_week', 'is_weekend'] + [col for col in detected_cat_features 
                                               if col not in ['hour', 'day_of_week', 'is_weekend']]

# Log detected categorical features
print(f"Using categorical features: {cat_features}")

# Define the target column
target_col = 'measured'

# Define quantiles for evaluation
quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

# Create directories for logs and results
os.makedirs("logs/wind/comparison", exist_ok=True)
os.makedirs("results/wind/comparison", exist_ok=True)

# Initialize results dataframes
results_df = pd.DataFrame()
detailed_scores = pd.DataFrame()

# Get the latest timestamp for testing
max_datetime = df['datetime'].max()

# Create test set (last timestamp in the data)
test_mask = df['datetime'] == max_datetime
test_df = df[test_mask].copy()
y_test = test_df[target_col]

# Remove datetime column and target from features
X_test = test_df.drop(columns=[target_col, 'datetime'])

# For each training period
for period_name, hours in training_periods.items():
    print(f"\n{'='*50}")
    print(f"Running with training period: {period_name} ({hours} hours)")
    print(f"{'='*50}")
    
    # Calculate the minimum datetime to include in training
    min_train_datetime = max_datetime - pd.Timedelta(hours=hours)
    train_mask = (df['datetime'] < max_datetime) & (df['datetime'] >= min_train_datetime)
    train_df = df[train_mask].copy()
    
    # Skip this period if not enough data
    if len(train_df) < 100:
        print(f"Not enough data for period {period_name}, skipping...")
        continue
    
    # Separate features and target for training
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col, 'datetime'])
    
    # Split into training and validation sets
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42)
    
    # Training time tracking
    start_time_lgbmlss = time.time()
    
    #-------------------------------------------------------------------------
    # PART 1: Train LightGBMLSS model
    #-------------------------------------------------------------------------
    print("Training LightGBMLSS model...")
    
    # Start tracking total time including HP optimization
    start_time_lgbmlss_total = time.time()
    
    # Define the model
    if dist == "Gaussian":
        lgblss = LightGBMLSS(
            Gaussian(
                stabilization=stabilization,
                response_fn=mode,
                loss_fn="nll"
            )
        )
    elif dist == "NegativeBinomial":
        if len(X_train) > 20000:  # Adjust threshold as needed
            lgblss = LightGBMLSS(
                NegativeBinomial(
                    stabilization=stabilization,
                    response_fn_total_count='softplus',
                )
            )
        else:
            lgblss = LightGBMLSS(
                NegativeBinomial(
                    stabilization=stabilization,
                    response_fn_total_count='exp',
                )
            )
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    # Initialize start values for wind generation data
    # These are typically non-negative continuous values
    mu_init = np.maximum(np.mean(y_train), 1e-2)    
    var_init = np.maximum(np.var(y_train), 1e-2)
    disp_init = max(var_init / mu_init - 1, 1e-2) if dist == "NegativeBinomial" else None

    if dist == "Gaussian":
        lgblss.start_values = np.array([np.log(mu_init), np.log(var_init)])
    else:  # NegativeBinomial
        lgblss.start_values = np.array([np.log(mu_init), np.log(disp_init)])
    
    # Create LightGBM datasets (keeping data as DataFrames)
    dtrain = lgb.Dataset(X_train_fit, y_train_fit, categorical_feature=cat_features, free_raw_data=False)
    dval = lgb.Dataset(X_val, y_val, categorical_feature=cat_features, free_raw_data=False)
    dtrain_full = lgb.Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
    
    # Define hyperparameter space
    param_dict = {
        "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
        "num_leaves": ["int", {"low": 20, "high": 100, "log": False}],
        "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],
        "lambda_l1": ["float", {"low": 1e-8, "high": 10, "log": True}],
        "histogram_pool_size": ["int", {"low": 1e3, "high": 5e3, "log": True}],
        "feature_pre_filter": ["categorical", [False]],
    }

    if len(X_train) > 20000:  # Adjust threshold as needed
        param_dict["eta"] = ["float", {"low": 1e-6, "high": 5e-3, "log": True}]
    else:
        param_dict["eta"] = ["float", {"low": 1e-5, "high": 1e-1, "log": True}]

    # Add device parameter if CUDA is available
    param_dict["device"] = ["categorical", ['cuda']] if torch.cuda.is_available() else ["categorical", ['cpu']]
    
    # HP tuning with fewer trials for comparison experiment
    opt_params = lgblss.hyper_opt(
        param_dict, dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        nfold=5,
        early_stopping_rounds=20,
        max_minutes=600,  # Reduced from 1440 to make experiment faster
        n_trials=10,     # Reduced from 80 to make experiment faster
        silence=True,
        seed=1,
        hp_seed=1,
    )
    
    n_rounds = opt_params.pop("opt_rounds")
    print(f"LightGBMLSS best number of boosters: {n_rounds}")
    
    # Store HP optimization time
    hp_opt_time_lss = time.time() - start_time_lgbmlss_total
    
    # Continue with actual model training time
    start_time_lgbmlss = time.time()
    
    # Train final model
    model_lgbmlss = lgblss.train(opt_params, dtrain_full, num_boost_round=n_rounds)
    
    lss_time = time.time() - start_time_lgbmlss
    lss_total_time = time.time() - start_time_lgbmlss_total
    
    # Predict quantiles with LightGBMLSS
    q_preds_lss = lgblss.predict(X_test, pred_type="quantiles",
                                 n_samples=200,
                                 quantiles=quantiles)
    
    #-------------------------------------------------------------------------
    # PART 2: Train regular LightGBM with pinball loss for quantile regression
    #-------------------------------------------------------------------------
    print("Training regular LightGBM models for quantile regression...")
    start_time_lgbm_total = time.time()
    
    # For storing quantile predictions from standard LightGBM
    q_preds_lgbm = pd.DataFrame(index=X_test.index)
    
    # Base parameters for LightGBM
    base_params = {
        'objective': 'quantile',
        'boosting_type': 'gbdt',
        'metric': 'quantile',
        'verbosity': -1,
        'feature_pre_filter': False,
    }
    
    if torch.cuda.is_available():
        base_params['device'] = 'cuda'
    
    # Define hyperparameter optimization for regular LightGBM
    def objective(trial, quantile):
        # Parameters to optimize
        params = {
            'objective': 'quantile',
            'alpha': quantile,  # Set quantile for the pinball loss
            'boosting_type': 'gbdt',
            'metric': 'quantile',
            'verbosity': -1,
            'feature_pre_filter': False,
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10, log=True),
        }
        
        if torch.cuda.is_available():
            params['device'] = 'cuda'
        
        # Create dataset
        dtrain_lgb = lgb.Dataset(X_train_fit, y_train_fit, categorical_feature=cat_features)
        dval_lgb = lgb.Dataset(X_val, y_val, categorical_feature=cat_features, reference=dtrain_lgb)
        
        # Train with early stopping
        gbm = lgb.train(
            params,
            dtrain_lgb,
            num_boost_round=NUM_BOOST_ROUNDS,
            valid_sets=[dval_lgb],
            callbacks=[lgb.early_stopping(stopping_rounds=20)],
        )
        
        # Extract best score (quantile loss)
        return gbm.best_score['valid_0']['quantile']
    
    # Store optimized parameters and best iterations for each quantile
    lgbm_opt_params = {}
    lgbm_best_rounds = {}
    
    # Time spent on HP optimization for LightGBM
    hp_opt_time_lgbm = 0
    
    # Perform HP optimization for each quantile
    for q in quantiles:
        print(f"Optimizing LightGBM for quantile {q}")
        start_time_q_opt = time.time()
        
        # Create an Optuna study for this quantile
        sampler = TPESampler(seed=1)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(lambda trial: objective(trial, q), n_trials=10, timeout=60*60)  # 10 trials, 1 hour max
        
        # Store results
        best_params = study.best_params
        best_params['objective'] = 'quantile'
        best_params['alpha'] = q
        best_params['verbosity'] = -1
        best_params['feature_pre_filter'] = False
        
        if torch.cuda.is_available():
            best_params['device'] = 'cuda'
            
        lgbm_opt_params[q] = best_params
        lgbm_best_rounds[q] = study.best_trial.user_attrs.get('best_iteration', NUM_BOOST_ROUNDS)
        
        hp_opt_time_lgbm += time.time() - start_time_q_opt
    
    # Now train final models with optimized parameters
    start_time_lgbm = time.time()
    
    # Train one model for each quantile using optimized parameters
    for q in quantiles:
        print(f"Training LightGBM for quantile {q} with optimized parameters")
        
        # Get optimized parameters
        params = lgbm_opt_params[q]
        n_rounds = lgbm_best_rounds[q]
        
        # Create dataset for full training
        dtrain_lgb = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)
        
        # Train final model
        gbm = lgb.train(
            params,
            dtrain_lgb,
            num_boost_round=n_rounds,
            categorical_feature=cat_features
        )
        
        # Predict for this quantile
        preds = gbm.predict(X_test)
        # For wind generation, we don't need to round to integers
        q_preds_lgbm[f"quant_{q}"] = preds
    
    lgbm_time = time.time() - start_time_lgbm
    lgbm_total_time = time.time() - start_time_lgbm_total
    
    #-------------------------------------------------------------------------
    # Evaluate and compare both approaches
    #-------------------------------------------------------------------------
    print("Evaluating and comparing models...")
    
    # Compute quantile losses for both approaches
    ql_lss = {}
    ql_lgbm = {}
    
    # Add this debug line before the loop
    print("Available columns in q_preds_lss:", q_preds_lss.columns.tolist())
    
    for i, q in enumerate(quantiles):
        col_name = f"quant_{q}"
        # LightGBMLSS
        ql_lss[q] = np.mean(quantile_loss(y_test.values, q_preds_lss[col_name].values, q))
        # Regular LightGBM
        ql_lgbm[q] = np.mean(quantile_loss(y_test.values, q_preds_lgbm[col_name].values, q))
    
    # Average pinball loss across all quantiles
    avg_ql_lss = np.mean(list(ql_lss.values()))
    avg_ql_lgbm = np.mean(list(ql_lgbm.values()))
    
    # Compute RMSE for median predictions (q=0.5)
    rmse_lss = np.sqrt(mean_squared_error(y_test.values, q_preds_lss["quant_0.5"].values))
    rmse_lgbm = np.sqrt(mean_squared_error(y_test.values, q_preds_lgbm["quant_0.5"].values))
    
    # Calculate CRPS if necessary
    crps_lss = crps(y_test.values, q_preds_lss)
    crps_lgbm = crps(y_test.values, q_preds_lgbm)
    
    # Store results for this training period
    period_results = {
        "model_weather": model_nwp,
        "training_period": period_name,
        "training_hours": hours,
        "training_samples": len(X_train),
        "mode": mode,
        "stabilization": stabilization,
        "dist": dist,
        "lss_train_time": lss_time,
        "lgbm_train_time": lgbm_time,
        "lss_hp_opt_time": hp_opt_time_lss,
        "lgbm_hp_opt_time": hp_opt_time_lgbm,
        "lss_total_time": lss_total_time,
        "lgbm_total_time": lgbm_total_time,
        "lss_avg_pinball_loss": avg_ql_lss,
        "lgbm_avg_pinball_loss": avg_ql_lgbm,
        "lss_rmse": rmse_lss,
        "lgbm_rmse": rmse_lgbm,
        "lss_crps": crps_lss,
        "lgbm_crps": crps_lgbm,
    }
    
    # Add individual quantile losses
    for q in quantiles:
        period_results[f"lss_ql_{q}"] = ql_lss[q]
        period_results[f"lgbm_ql_{q}"] = ql_lgbm[q]
    
    # Append to results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([period_results])], ignore_index=True)
    
    # Create detailed per-sample metrics for this period
    detailed_period = pd.DataFrame({
        "test_index": np.arange(len(y_test)),
        "actual": y_test.values,
        "training_period": period_name,
        "training_hours": hours,
        "model_weather": model_nwp,
        "datetime": test_df['datetime'].values
    })
    
    # Add LightGBMLSS predictions
    for col in q_preds_lss.columns:
        detailed_period[f"lss_{col}"] = q_preds_lss[col].values
    
    # Add regular LightGBM predictions
    for col in q_preds_lgbm.columns:
        detailed_period[f"lgbm_{col}"] = q_preds_lgbm[col].values
    
    # Append to detailed scores
    detailed_scores = pd.concat([detailed_scores, detailed_period], ignore_index=True)
    
    print(f"Results for {period_name}:")
    print(f"- LightGBMLSS Avg Pinball Loss: {avg_ql_lss:.4f}, RMSE: {rmse_lss:.4f}, CRPS: {crps_lss:.4f}, Time: {lss_time:.1f}s")
    print(f"- LightGBM Avg Pinball Loss: {avg_ql_lgbm:.4f}, RMSE: {rmse_lgbm:.4f}, CRPS: {crps_lgbm:.4f}, Time: {lgbm_time:.1f}s")

# Change the file paths to be shared across models
results_output_path = f"results/wind/comparison/all_models_comparison_results_{method_name}.csv"
detailed_output_path = f"logs/wind/comparison/all_models_detailed_comparison_{method_name}.csv"

# Check if the summary results file already exists and append to it if it does
if os.path.exists(results_output_path):
    existing_results = pd.read_csv(results_output_path)
    # Append new results
    combined_results = pd.concat([existing_results, results_df], ignore_index=True)
    combined_results.to_csv(results_output_path, index=False)
    print(f"✅ Summary results appended to {results_output_path}")
else:
    # Create new file
    results_df.to_csv(results_output_path, index=False)
    print(f"✅ Summary results saved to {results_output_path}")

# Check if the detailed results file already exists and append to it if it does
if os.path.exists(detailed_output_path):
    existing_detailed = pd.read_csv(detailed_output_path)
    # Append new detailed results
    combined_detailed = pd.concat([existing_detailed, detailed_scores], ignore_index=True)
    combined_detailed.to_csv(detailed_output_path, index=False)
    print(f"✅ Detailed results appended to {detailed_output_path}")
else:
    # Create new file
    detailed_scores.to_csv(detailed_output_path, index=False)
    print(f"✅ Detailed results saved to {detailed_output_path}")


# Print final comparison
print("\n" + "="*80)
print("FINAL COMPARISON ACROSS ALL TRAINING PERIODS")
print("="*80)

# Compare average performance across all periods
for metric in ["avg_pinball_loss", "rmse", "crps"]:
    lss_avg = results_df[f"lss_{metric}"].mean()
    lgbm_avg = results_df[f"lgbm_{metric}"].mean()
    diff_pct = ((lss_avg - lgbm_avg) / lgbm_avg) * 100
    
    print(f"{metric.upper()}: LightGBMLSS: {lss_avg:.4f}, LightGBM: {lgbm_avg:.4f}, Diff: {diff_pct:.2f}%")

# Average training time comparison (including HP optimization)
lss_time_avg = results_df["lss_train_time"].mean()
lgbm_time_avg = results_df["lgbm_train_time"].mean()
lss_total_avg = results_df["lss_total_time"].mean()
lgbm_total_avg = results_df["lgbm_total_time"].mean()
time_ratio = lss_time_avg / lgbm_time_avg
total_time_ratio = lss_total_avg / lgbm_total_avg

print(f"TRAINING TIME (model only): LightGBMLSS: {lss_time_avg:.1f}s, LightGBM: {lgbm_time_avg:.1f}s, Ratio: {time_ratio:.2f}x")
print(f"TOTAL TIME (incl. HP opt): LightGBMLSS: {lss_total_avg:.1f}s, LightGBM: {lgbm_total_avg:.1f}s, Ratio: {total_time_ratio:.2f}x")
print("="*80)
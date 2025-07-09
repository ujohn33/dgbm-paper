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
SPLIT_DATE = "2025-03-01"  # Global split date for train/test divide
SCALE = True  # Whether to scale the target by monitored capacity

# Get parameters from command line
if len(sys.argv) < 2:
    print("Usage: python solar_comparison_lssboost_vs_lightgbm.py <model_nwp> [mode] [stabilization] [dist]")
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
    "2_weeks": 336,
    "1_month": 720,
    "3_months": 2160,
    "6_months": 4320,
    "1_year": 8760,
    "1.5_years": 13140,
    "2_years": 17520,
}

# Parse arguments
model_nwp = sys.argv[1] if len(sys.argv) > 1 else "icon_eu"
mode = sys.argv[2] if len(sys.argv) > 2 else "softplus"
stabilization = sys.argv[3] if len(sys.argv) > 3 else 'MAD'
dist = sys.argv[4] if len(sys.argv) > 4 else "Gaussian"

# today's date
today = pd.to_datetime("today").strftime("%Y-%m-%d")
method_name = f'{model_nwp}_{mode}_{stabilization}_{dist}_{today}'

# Log received parameters
print(f"Parameters: weather model={model_nwp}, mode={mode}, stabilization={stabilization}, dist={dist}")

# Path to data
data_path = f"/scratch/brussel/105/vsc10528/WP_predico_project/data/training_solar_data_iconeu_gfsseamless_ecmwfifs025_17jun2025.csv"

# Load data
print(f"Loading data from {data_path}")
df = pd.read_csv(data_path, parse_dates=['datetime'])

# List of models to exclude or include based on the selected model_nwp
all_models = ['icon', 'ecmwf', 'gfs', 'metno', 'best', 'aifs025']
always_keep_cols = ['datetime', 'measured', 'monitoredcapacity', 'time_of_day', 
                   'day_of_year', 'year_month']

# Determine which model to keep
if 'icon_eu' in model_nwp:
    models_to_exclude = [m for m in all_models if m != 'icon']
elif 'ecmwf_aifs025_single' in model_nwp:
    models_to_exclude = [m for m in all_models if m not in ['ecmwf', 'aifs025']]
elif 'ecmwf_ifs025' in model_nwp:
    models_to_exclude = [m for m in all_models if m not in ['ecmwf', 'ifs025']]
    # Special case for ecmwf_ifs025: exclude aifs025
    if 'ifs025' in model_nwp:
        models_to_exclude.append('aifs025')
else:
    # Default case - don't exclude any model data
    models_to_exclude = []

# Start with all columns
columns_to_keep = list(df.columns)

# Filter columns based on excluded models
for col in df.columns:
    # Remove time-related columns that we don't want
    if any(x in col.lower() for x in ['month', 'hour', 'is_day']):
        if col in columns_to_keep:
            columns_to_keep.remove(col)
            
    # Remove columns from excluded models
    if any(model in col.lower() for model in models_to_exclude) and col not in always_keep_cols:
        if col in columns_to_keep:
            columns_to_keep.remove(col)

# Filter the dataframe to keep only relevant columns
df = df[columns_to_keep]

# Sort by datetime to ensure temporal order
df = df.sort_values('datetime')

# Store original measured values before any normalization
df['measured_raw'] = df['measured'].copy()

# Scale target if required (by total capacity)
if SCALE:
    df['measured_capacity_normalized'] = df['measured'] / df['monitoredcapacity']
    target_col = 'measured_capacity_normalized'
else:
    target_col = 'measured'
    df = df.drop(columns=["monitoredcapacity"])
    df['measured_capacity_normalized'] = df['measured'].copy()

# Define quantiles for evaluation
quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

# Create directories for logs and results
os.makedirs("logs/solar/comparison", exist_ok=True)
os.makedirs("results/solar/comparison", exist_ok=True)

# Initialize results dataframes
results_df = pd.DataFrame()
detailed_scores = pd.DataFrame()

# Split data by date
split_date = pd.to_datetime(SPLIT_DATE)
# Fix timezone mismatch - make split_date timezone-aware to match the datetime column
if df['datetime'].dt.tz is not None:
    split_date = split_date.tz_localize(df['datetime'].dt.tz)

train_mask = df['datetime'] < split_date
test_mask = df['datetime'] >= split_date

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

# Check if we have enough data
if len(train_df) < 100:
    print(f"Not enough training data (only {len(train_df)} samples). Exiting.")
    sys.exit(1)

if len(test_df) < 10:
    print(f"Not enough test data (only {len(test_df)} samples). Exiting.")
    sys.exit(1)

print(f"Data split: {len(train_df)} training samples, {len(test_df)} test samples")

# Setup test data
y_test = test_df[target_col]
X_test = test_df.drop(columns=[target_col, 'measured_raw', 'measured', 'datetime'])

# Log feature columns
print(f"Number of features: {len(X_test.columns)}")
print(f"First few features: {list(X_test.columns[:10])}")

# For each training period
for period_name, hours in training_periods.items():
    print(f"\n{'='*50}")
    print(f"Running with training period: {period_name} ({hours} hours)")
    print(f"{'='*50}")
    
    # Calculate the minimum datetime to include in training
    min_train_datetime = split_date - pd.Timedelta(hours=hours)
    # Ensure min_train_datetime has the same timezone as split_date
    if df['datetime'].dt.tz is not None and min_train_datetime.tz is None:
        min_train_datetime = min_train_datetime.tz_localize(df['datetime'].dt.tz)
    
    period_mask = (train_df['datetime'] >= min_train_datetime)
    period_train_df = train_df[period_mask].copy()
    
    # Skip this period if not enough data
    if len(period_train_df) < 100:
        print(f"Not enough data for period {period_name} (only {len(period_train_df)} samples), skipping...")
        continue
    
    # Separate features and target for training
    y_train = period_train_df[target_col]
    X_train = period_train_df.drop(columns=[target_col, 'measured_raw', 'measured', 'datetime'])
    
    # Split into training and validation sets
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, shuffle=False)
    
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
                response_fn='softplus', 
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

    # Initialize start values for solar generation data
    # These are typically non-negative continuous values
    mu_init = np.maximum(np.mean(y_train), 1e-2)    
    var_init = np.maximum(np.var(y_train), 1e-2)
    disp_init = max(var_init / mu_init - 1, 1e-2) if dist == "NegativeBinomial" else None

    if dist == "Gaussian":
        lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])
    else:  # NegativeBinomial
        lgblss.start_values = np.array([np.log(mu_init), np.log(disp_init)])
    
    # Create LightGBM datasets (keeping data as DataFrames)
    dtrain = lgb.Dataset(X_train_fit, y_train_fit, free_raw_data=False)
    dval = lgb.Dataset(X_val, y_val, free_raw_data=False)
    dtrain_full = lgb.Dataset(X_train, y_train, free_raw_data=False)

    # Define hyperparameter space (identical for both models)
    def create_param_space(trial, is_lgbmlss=False):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
            "feature_pre_filter": False,
        }
        
        if len(X_train) > 20000:
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 5e-3, log=True)
        
        # Add device parameter if CUDA is available
        if torch.cuda.is_available():
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"
            
        # Add LightGBMLSS specific parameters
        if is_lgbmlss:
            params["eta"] = params.pop("learning_rate")  # LightGBMLSS uses 'eta' instead of 'learning_rate'
            
        return params
    
    # Time series cross-validation function
    def time_series_cv_score(params, X_train_data, y_train_data, lgblss_model=None, n_splits=3):
        """
        Perform time series cross-validation without shuffling data.
        """
        n_samples = len(X_train_data)
        scores = []
        
        for i in range(n_splits):
            # Calculate split points maintaining temporal order
            test_start = int(n_samples * (i + 1) / (n_splits + 1))
            test_end = int(n_samples * (i + 2) / (n_splits + 1))
            
            # Training data: everything before test period
            train_indices = list(range(test_start))
            test_indices = list(range(test_start, test_end))
            
            if len(train_indices) < 50 or len(test_indices) < 20:
                continue
                
            X_fold_train = X_train_data.iloc[train_indices]
            y_fold_train = y_train_data.iloc[train_indices]
            X_fold_test = X_train_data.iloc[test_indices]
            y_fold_test = y_train_data.iloc[test_indices]
            
            try:
                if lgblss_model is not None:
                    # LightGBMLSS training
                    dtrain_fold = lgb.Dataset(X_fold_train, y_fold_train, free_raw_data=False)
                    dtest_fold = lgb.Dataset(X_fold_test, y_fold_test, free_raw_data=False)

                    model = lgblss_model.train(params, dtrain_fold, num_boost_round=NUM_BOOST_ROUNDS,
                                             valid_sets=[dtest_fold], early_stopping_rounds=20)
                    
                    # Get validation score from the model
                    if hasattr(model, 'best_score') and 'valid_0' in model.best_score:
                        score = model.best_score['valid_0']['l2']  # Use L2 loss for LightGBMLSS
                    else:
                        # Fallback: make predictions and calculate MSE
                        preds = lgblss_model.predict(X_fold_test, pred_type="parameters")
                        if dist == "Gaussian":
                            pred_mean = preds.iloc[:, 0]  # First parameter is mean
                        else:
                            pred_mean = preds.iloc[:, 0]  # Adjust based on distribution
                        score = mean_squared_error(y_fold_test, pred_mean)
                else:
                    # Regular LightGBM training
                    dtrain_fold = lgb.Dataset(X_fold_train, y_fold_train, free_raw_data=False)
                    dtest_fold = lgb.Dataset(X_fold_test, y_fold_test, reference=dtrain_fold)

                    model = lgb.train(params, dtrain_fold, num_boost_round=NUM_BOOST_ROUNDS,
                                    valid_sets=[dtest_fold], early_stopping_rounds=20, 
                                    callbacks=[lgb.early_stopping(stopping_rounds=20)])
                    
                    # Get validation score
                    if hasattr(model, 'best_score') and 'valid_0' in model.best_score:
                        if 'l2' in model.best_score['valid_0']:
                            score = model.best_score['valid_0']['l2']
                        elif 'quantile' in model.best_score['valid_0']:
                            score = model.best_score['valid_0']['quantile']
                        else:
                            # Fallback
                            preds = model.predict(X_fold_test)
                            score = mean_squared_error(y_fold_test, preds)
                    else:
                        # Fallback
                        preds = model.predict(X_fold_test)
                        score = mean_squared_error(y_fold_test, preds)
                
                scores.append(score)
                
            except Exception as e:
                print(f"Error in fold {i}: {e}")
                continue
        
        return np.mean(scores) if scores else float('inf')
    
    # Custom Optuna objective for LightGBMLSS
    def objective_lgbmlss(trial):
        params = create_param_space(trial, is_lgbmlss=True)
        score = time_series_cv_score(params, X_train, y_train, lgblss_model=lgblss)
        return score
    
    # HP tuning for LightGBMLSS with time series CV
    print("Starting hyperparameter optimization for LightGBMLSS with time series CV...")
    sampler = TPESampler(seed=1)
    study_lss = optuna.create_study(direction='minimize', sampler=sampler)
    study_lss.optimize(objective_lgbmlss, n_trials=10, timeout=600)  # 10 trials, 10 minutes max
    
    # Get best parameters for LightGBMLSS
    opt_params = study_lss.best_params
    opt_params["feature_pre_filter"] = False
    if torch.cuda.is_available():
        opt_params["device"] = "cuda"
    
    # Store HP optimization time
    hp_opt_time_lss = time.time() - start_time_lgbmlss_total
    
    # Continue with actual model training time
    start_time_lgbmlss = time.time()
    
    # Train final model with best parameters
    model_lgbmlss = lgblss.train(opt_params, dtrain_full, num_boost_round=NUM_BOOST_ROUNDS)
    
    lss_time = time.time() - start_time_lgbmlss
    lss_total_time = time.time() - start_time_lgbmlss_total
    
    # Predict quantiles with LightGBMLSS
    q_preds_lss = lgblss.predict(X_test, pred_type="quantiles",
                                 n_samples=1000,
                                 quantiles=quantiles)
    
    # Scale predictions back if needed
    if SCALE:
        for col in q_preds_lss.columns:
            q_preds_lss[col] = q_preds_lss[col] * test_df['monitoredcapacity'].values
        y_test_actual = y_test * test_df['monitoredcapacity'].values
    else:
        y_test_actual = y_test
    
    # Clip LightGBMLSS predictions to be non-negative
    for col in q_preds_lss.columns:
        q_preds_lss[col] = np.maximum(q_preds_lss[col], 0)
    
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
    def objective_lgbm(trial, quantile):
        # Use the same parameter space as LightGBMLSS
        params = create_param_space(trial, is_lgbmlss=False)
        
        # Add LightGBM specific parameters
        params['objective'] = 'quantile'
        params['alpha'] = quantile
        params['metric'] = 'quantile'
        params['verbosity'] = -1
        
        # Use time series CV for regular LightGBM too
        score = time_series_cv_score(params, X_train, y_train, lgblss_model=None)
        return score
    
    # Store optimized parameters and best iterations for each quantile
    lgbm_opt_params = {}
    lgbm_best_rounds = {}
    
    # Time spent on HP optimization for LightGBM
    hp_opt_time_lgbm = 0
    
    # Perform HP optimization for each quantile using time series CV
    for q in quantiles:
        print(f"Optimizing LightGBM for quantile {q} with time series CV...")
        start_time_q_opt = time.time()
        
        # Create an Optuna study for this quantile
        sampler = TPESampler(seed=1)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(lambda trial: objective_lgbm(trial, q), n_trials=10, timeout=600)  # 10 trials, 10 minutes max
        
        # Store results
        best_params = study.best_params
        best_params['objective'] = 'quantile'
        best_params['alpha'] = q
        best_params['metric'] = 'quantile'
        best_params['verbosity'] = -1
        best_params['feature_pre_filter'] = False
        
        if torch.cuda.is_available():
            best_params['device'] = 'cuda'
            
        lgbm_opt_params[q] = best_params
        lgbm_best_rounds[q] = NUM_BOOST_ROUNDS  # Use fixed number of rounds
        
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
        dtrain_lgb = lgb.Dataset(X_train, y_train)
        
        # Train final model
        gbm = lgb.train(
            params,
            dtrain_lgb,
            num_boost_round=n_rounds
        )
        
        # Predict for this quantile
        preds = gbm.predict(X_test)
        # Convert to physical values if capacity scaled
        if SCALE:
            preds = preds * test_df['monitoredcapacity'].values
        
        # Clip predictions to be non-negative
        preds = np.maximum(preds, 0)
        
        q_preds_lgbm[f"quant_{q}"] = preds
    
    lgbm_time = time.time() - start_time_lgbm
    lgbm_total_time = time.time() - start_time_lgbm_total
    
    #-------------------------------------------------------------------------
    # Evaluate and compare both approaches
    #-------------------------------------------------------------------------
    print("Evaluating and comparing models...")
    
    print(f"DEBUG SCALING CHECK:")
    print(f"- y_test (normalized) range: {y_test.min():.4f} to {y_test.max():.4f}")
    print(f"- y_test_actual (physical) range: {y_test_actual.min():.2f} to {y_test_actual.max():.2f}")
    print(f"- monitoredcapacity range: {test_df['monitoredcapacity'].min():.2f} to {test_df['monitoredcapacity'].max():.2f}")
    print(f"- LSS predictions (q=0.5) range: {q_preds_lss['quant_0.5'].min():.2f} to {q_preds_lss['quant_0.5'].max():.2f}")

    # Calculate a sample quantile loss to check scale
    sample_ql = quantile_loss(y_test_actual.values[:5], q_preds_lss['quant_0.5'].values[:5], 0.5)
    print(f"- Sample quantile losses (first 5): {sample_ql}")
    print(f"- Mean of sample QL: {np.mean(sample_ql):.2f}")


    # Compute quantile losses for both approaches
    ql_lss = {}
    ql_lgbm = {}
    
    for i, q in enumerate(quantiles):
        col_name = f"quant_{q}"
        
        if SCALE:
            # Convert predictions back to normalized scale for loss calculation
            pred_lss_norm = q_preds_lss[col_name].values / test_df['monitoredcapacity'].values
            pred_lgbm_norm = q_preds_lgbm[col_name].values / test_df['monitoredcapacity'].values
            y_test_norm = y_test.values  # Already normalized
            
            # CORRECT parameter order: quantile_loss(q, y_true, y_pred)
            ql_lss[q] = np.mean(quantile_loss(q, y_test_norm, pred_lss_norm))
            ql_lgbm[q] = np.mean(quantile_loss(q, y_test_norm, pred_lgbm_norm))
        else:
            # Scale down to [0,1] range for comparable losses
            y_max = max(y_test_actual.max(), q_preds_lss[col_name].max(), q_preds_lgbm[col_name].max())
            y_norm = y_test_actual.values / y_max
            pred_lss_norm = q_preds_lss[col_name].values / y_max
            pred_lgbm_norm = q_preds_lgbm[col_name].values / y_max
            
            # CORRECT parameter order: quantile_loss(q, y_true, y_pred)
            ql_lss[q] = np.mean(quantile_loss(q, y_norm, pred_lss_norm))
            ql_lgbm[q] = np.mean(quantile_loss(q, y_norm, pred_lgbm_norm))
    
    # Average pinball loss across all quantiles
    avg_ql_lss = np.mean(list(ql_lss.values()))
    avg_ql_lgbm = np.mean(list(ql_lgbm.values()))
    
    # Compute RMSE for median predictions (q=0.5)
    rmse_lss = np.sqrt(mean_squared_error(y_test_actual.values, q_preds_lss["quant_0.5"].values))
    rmse_lgbm = np.sqrt(mean_squared_error(y_test_actual.values, q_preds_lgbm["quant_0.5"].values))
    
    # Calculate CRPS
    crps_lss_result = crps(y_test_actual.values, q_preds_lss)
    crps_lgbm_result = crps(y_test_actual.values, q_preds_lgbm)
    
    # Extract scalar values if the function returns tuples
    crps_lss = crps_lss_result[0] if isinstance(crps_lss_result, tuple) else crps_lss_result
    crps_lgbm = crps_lgbm_result[0] if isinstance(crps_lgbm_result, tuple) else crps_lgbm_result
    
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
        "test_index": np.arange(len(y_test_actual)),
        "actual": y_test_actual.values,
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

# Change the file paths to be specific to the selected model
results_output_path = f"results/solar/comparison/{model_nwp}_comparison_results_{method_name}.csv"
detailed_output_path = f"logs/solar/comparison/{model_nwp}_detailed_comparison_{method_name}.csv"

# Save the results
results_df.to_csv(results_output_path, index=False)
print(f"✅ Summary results saved to {results_output_path}")
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
import os, sys, json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor
from ngboost.learners import default_linear_learner, default_tree_learner
from ngboost import NGBRegressor
from ngboost.distns import Normal, Poisson
from ngboost.scores import LogScore
import optuna
from optuna.samplers import TPESampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.safety import apply_safety_net

np.random.seed(123)

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
}

# Define training periods in days
training_periods = {
    "1_week": 7,
    "2_weeks": 14,
    "1_month": 30,
    "3_months": 90,
    "6_months": 180,
    "1_year": 365,
}

# Get cluster ID and other parameters from command line
if len(sys.argv) < 2:
    print("Usage: python m5_ngboost_HP_single_run.py <cluster_id> [mode] [natural_grad] [stabilization] [clip_value] [standardize]")
    sys.exit(1)

# Parse arguments
cluster_id = int(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "exp"
natural_grad = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
stabilization = sys.argv[4] if len(sys.argv) > 4 else 'None'
clip_value = None if sys.argv[5].lower() == "none" else float(sys.argv[5]) if len(sys.argv) > 5 else None
standardize = sys.argv[6].lower() == "true" if len(sys.argv) > 6 else False
dist = sys.argv[7] if len(sys.argv) > 7 else "NegativeBinomial"

method_name = f'{mode}_{natural_grad}_{stabilization}_{dist}'

# Log received parameters
print(f"Parameters: mode={mode}, natural_grad={natural_grad}, stabilization={stabilization}, clip_value={clip_value}, standardize={standardize}")

cluster_path = f"data/train_cluster_{cluster_id}.csv"

# Load data
df = pd.read_csv(cluster_path)

# Identify the last date
max_d = df["d"].max()

# Define quantiles for evaluation
quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

# Create directories for logs and results
os.makedirs("logs/m5/ngboost", exist_ok=True)
os.makedirs("results/m5/ngboost", exist_ok=True)

# Initialize results dataframes
results_df = pd.DataFrame()
detailed_scores = pd.DataFrame()

# Create test set (always the last day)
test_mask = df["d"] == max_d
test_df = df[test_mask].copy()
y_test = test_df["demand"]
X_test = test_df.drop(columns=["demand", "d"])

# Convert categorical columns in test set
for col in ["store_id", "item_id", "wday", "weekend_plus"]:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category').cat.codes

# Map distribution names to NGBoost distribution classes
dist_mapping = {
    "Gaussian": Normal,
    "Normal": Normal,
    "Poisson": Poisson
}

if dist not in dist_mapping:
    raise ValueError(f"Unknown distribution: {dist}")

selected_dist = dist_mapping[dist]

# Define base learners for hyperparameter tuning
b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
base_learner_choices = [b1, b2, b3]

# For each training period
for period_name, days in training_periods.items():
    print(f"\n{'='*50}")
    print(f"Running with training period: {period_name} ({days} days)")
    print(f"{'='*50}")
    
    # Calculate the minimum date to include in training
    min_train_d = max_d - days
    train_mask = (df["d"] < max_d) & (df["d"] >= min_train_d)
    train_df = df[train_mask].copy()
    
    # Skip this period if not enough data
    if len(train_df) < 100:
        print(f"Not enough data for period {period_name}, skipping...")
        continue
    
    # Separate features and target for training
    y_train = train_df["demand"]
    X_train_full = train_df.drop(columns=["demand", "d"])
    
    # Convert categorical columns
    for col in ["store_id", "item_id", "wday", "weekend_plus"]:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype('category').cat.codes
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train, test_size=0.25, random_state=42)
    
    # Start timing for this period
    start_time_total = time.time()
    
    # Define the Optuna objective function for hyperparameter tuning
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 500, 2000)
        minibatch_frac = trial.suggest_float("minibatch_frac", 0.5, 1.0)
        base_learner_idx = trial.suggest_categorical('base_learner', [0, 1, 2])
        
        base_learner = base_learner_choices[base_learner_idx]
        
        # Create and train model
        ngb = NGBRegressor(
            Base=base_learner,
            Dist=selected_dist,
            Score=LogScore,
            n_estimators=n_estimators,
            learning_rate=lr,
            natural_gradient=natural_grad,
            minibatch_frac=minibatch_frac,
            verbose=False
        )
        
        ngb.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=20)
        
        # Make predictions
        forecast = ngb.pred_dist(X_val)
        
        # Calculate validation metrics
        val_nll = -forecast.logpdf(y_val).mean()
        
        return val_nll
    
    # Run hyperparameter optimization with fewer trials for comparison
    print(f"Starting hyperparameter optimization for {period_name}...")
    hp_start_time = time.time()
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=1))
    study.optimize(objective, n_trials=10, timeout=600*60)  # Reduced to 10 trials, 10 hours timeout
    
    opt_params = study.best_params
    print(f"Best hyperparameters for {period_name}:", opt_params)
    
    # Store HP optimization time
    hp_opt_time = time.time() - hp_start_time
    
    # Train final model with best hyperparameters
    train_start_time = time.time()
    
    best_lr = opt_params["lr"]
    best_n_estimators = opt_params["n_estimators"]
    best_minibatch_frac = opt_params["minibatch_frac"]
    best_base_idx = opt_params["base_learner"]
    best_base = base_learner_choices[best_base_idx]
    
    final_ngb = NGBRegressor(
        Base=best_base,
        Dist=selected_dist,
        Score=LogScore,
        n_estimators=best_n_estimators,
        learning_rate=best_lr,
        natural_gradient=natural_grad,
        minibatch_frac=best_minibatch_frac,
        verbose=False
    )
    
    print(f"Training final model for {period_name}...")
    final_ngb.fit(X_train_full, y_train)
    n_rounds = final_ngb.best_val_loss_itr if hasattr(final_ngb, 'best_val_loss_itr') else best_n_estimators
    
    train_time = time.time() - train_start_time
    total_time = time.time() - start_time_total
    
    # Predict and evaluate
    print(f"Making predictions for {period_name}...")
    forecast_dist = final_ngb.pred_dist(X_test)
    
    # Get quantile predictions
    q_preds = {}
    for q in quantiles:
        q_pred = forecast_dist.ppf(q)
        # Round predictions to nearest integer since demand must be discrete
        q_preds[f"quant_{q}"] = np.round(q_pred).astype(int)
    
    q_preds_df = pd.DataFrame(q_preds)
    
    # Compute quantile losses
    ql = {}
    for q in quantiles:
        col_name = f"quant_{q}"
        ql[q] = np.mean(quantile_loss(y_test.values, q_preds_df[col_name].values, q))
    
    # Average pinball loss across all quantiles
    avg_ql = np.mean(list(ql.values()))
    
    # Compute RMSE for median predictions (q=0.5)
    rmse = np.sqrt(mean_squared_error(y_test.values, q_preds_df["quant_0.5"].values))
    
    # Create detailed per-sample metrics for this period
    detailed_period = pd.DataFrame({
        "test_index": np.arange(len(y_test)),
        "actual": y_test.values,
        "training_period": period_name,
        "training_days": days,
        "cluster_id": cluster_id,
    })
    
    # Add predictions for each quantile
    for col in q_preds_df.columns:
        detailed_period[f"ngboost_{col}"] = q_preds_df[col].values
    
    # Add method parameters
    detailed_period["mode"] = mode
    detailed_period["natural_grad"] = natural_grad
    detailed_period["stabilization"] = stabilization
    detailed_period["dist"] = dist
    
    # Append to detailed scores
    detailed_scores = pd.concat([detailed_scores, detailed_period], ignore_index=True)
    
    # Store results for this training period
    period_results = {
        "cluster_id": cluster_id,
        "training_period": period_name,
        "training_days": days,
        "training_samples": len(X_train_full),
        "mode": mode,
        "natural_grad": natural_grad,
        "stabilization": stabilization,
        "dist": dist,
        "train_time": train_time,
        "hp_opt_time": hp_opt_time,
        "total_time": total_time,
        "avg_pinball_loss": avg_ql,
        "rmse": rmse,
        "n_rounds": n_rounds,
    }
    
    # Add individual quantile losses
    for q in quantiles:
        period_results[f"ql_{q}"] = ql[q]
    
    # Append to results dataframe
    results_df = pd.concat([results_df, pd.DataFrame([period_results])], ignore_index=True)
    
    print(f"Results for {period_name}:")
    print(f"- NGBoost Avg Pinball Loss: {avg_ql:.4f}, RMSE: {rmse:.4f}, Train Time: {train_time:.1f}s")

# Change the file paths to be shared across clusters
results_output_path = f"results/m5/ngboost/all_clusters_results_{method_name}.csv"
detailed_output_path = f"logs/m5/ngboost/all_clusters_detailed_{method_name}.csv"

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
print(f"AVERAGE PINBALL LOSS: {results_df['avg_pinball_loss'].mean():.4f}")
print(f"AVERAGE RMSE: {results_df['rmse'].mean():.4f}")
print(f"AVERAGE TRAIN TIME: {results_df['train_time'].mean():.1f}s")
print(f"AVERAGE TOTAL TIME: {results_df['total_time'].mean():.1f}s")
print("="*80)
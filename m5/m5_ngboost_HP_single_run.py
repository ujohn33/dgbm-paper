import os, sys, json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor
from ngboost.learners import default_linear_learner, default_tree_learner
from ngboost import NGBRegressor
from ngboost.distns import Normal, Poisson
from ngboost.scores import LogScore
import optuna

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.safety import apply_safety_net

np.random.seed(123)

base_name_to_learner = {
    "tree": default_tree_learner,
    "linear": default_linear_learner,
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

# Log received parameters
print(f"Parameters: mode={mode}, natural_grad={natural_grad}, stabilization={stabilization}, clip_value={clip_value}, standardize={standardize}")

cluster_path = f"data/train_cluster_{cluster_id}.csv"

# Load data
df = pd.read_csv(cluster_path)

# Identify the last date
max_d = df["d"].max()

# Create train/test split like in R
test_mask = df["d"] == max_d
train_df = df[~test_mask].copy()
test_df = df[test_mask].copy()

# Separate features and target
y_trainval = train_df["demand"]
X_trainval = train_df.drop(columns=["demand", "d"])

y_test = test_df["demand"]
X_test = test_df.drop(columns=["demand", "d"])

# Convert categorical columns
for col in ["store_id", "item_id", "wday", "weekend_plus"]:
    if col in X_trainval.columns:
        # Convert to categorical then to integer codes
        X_trainval[col] = X_trainval[col].astype('category').cat.codes
        X_test[col] = X_test[col].astype('category').cat.codes

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

# Define base learners for hyperparameter tuning
b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
base_learner_choices = [b1, b2, b3]

# Map distribution names to NGBoost distribution classes
dist_mapping = {
    "Gaussian": Normal,
    "Normal": Normal,
    "Poisson": Poisson
}

if dist not in dist_mapping:
    raise ValueError(f"Unknown distribution: {dist}")

selected_dist = dist_mapping[dist]

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

# Run hyperparameter optimization
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=80, timeout=86400)  # 24 hours timeout

print("Best hyperparameters:", study.best_params)
opt_params = study.best_params

# Train final model with best hyperparameters
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

print("Training final model...")
final_ngb.fit(X_trainval, y_trainval)
n_rounds = final_ngb.best_val_loss_itr if hasattr(final_ngb, 'best_val_loss_itr') else best_n_estimators

# Predict and evaluate
print("Making predictions...")
forecast_dist = final_ngb.pred_dist(X_test)

# Evaluate at quantiles
quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
q_preds = pd.DataFrame({
    f"q{q}": forecast_dist.ppf(q) for q in quantiles
})

# Create directory for detailed logs
os.makedirs("logs/m5", exist_ok=True)

# Print detailed per-sample metrics
print("Detailed per-sample metrics:")
print(f"Actual values: {y_test.values}")
print(f"Quantile predictions: {q_preds}")

# Create a dataframe with per-sample metrics
per_sample_metrics = pd.DataFrame({
    "test_index": np.arange(len(y_test)),
    "actual": y_test.values,
})

per_sample_metrics = per_sample_metrics.join(q_preds)

print(per_sample_metrics)

# Add cluster info and parameters
per_sample_metrics["cluster_id"] = cluster_id
per_sample_metrics["mode"] = mode
per_sample_metrics["natural_grad"] = natural_grad
per_sample_metrics["stabilization"] = stabilization
per_sample_metrics["n_rounds"] = n_rounds
per_sample_metrics["dist"] = dist

# Save to CSV
log_file = f"logs/m5/ngboost_clusters_detailed_scores.csv"

# Check if the file exists
file_exists = os.path.exists(log_file)

# Write to file in append mode
with open(log_file, 'a+') as f:
    # If file doesn't exist, write header first
    if not file_exists:
        per_sample_metrics.to_csv(f, index=False)
        print(f"✅ Created new detailed scores file at {log_file}")
    else:
        # If file exists, append without header
        per_sample_metrics.to_csv(f, index=False, header=False)
        print(f"✅ Appended to existing scores in {log_file}")

print(f"✅ Detailed scores saved to {log_file}")

# Save results
os.makedirs("results/clusters/local", exist_ok=True)

results = {
    "cluster_id": cluster_id,
    "mode": mode,
    "natural_grad": natural_grad,
    "stabilization": stabilization,
    "clip_value": clip_value,
    "standardize": standardize,
    "n_rounds": n_rounds,
    "best_params": {k: (int(v) if k == "base_learner" else v) for k, v in opt_params.items()}
}

with open(f"results/clusters/local/ngboost_cluster_{cluster_id}_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Results saved to results/clusters/local/ngboost_cluster_{cluster_id}_results.json")
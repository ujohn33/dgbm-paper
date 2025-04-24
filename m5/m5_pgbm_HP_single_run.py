import os, sys, json, time
import pandas as pd
import numpy as np
import torch
import optuna
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from pgbm.torch import PGBM, PGBMRegressor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.safety import apply_safety_net
from utils.logging import log_predictions
from sklearn.preprocessing import LabelEncoder

np.random.seed(123)

# Get cluster ID and other parameters from command line
if len(sys.argv) < 2:
    print("Usage: python m5_pgbm_HP_single_run.py <cluster_id>")
    sys.exit(1)

# Parse arguments
cluster_id = int(sys.argv[1])
method_name = "pgbm"

# Log received parameters
print(f"Running PGBM for cluster {cluster_id}")

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
y_trainval = train_df["demand"].values
X_trainval = train_df.drop(columns=["demand", "d"]).values

cat_features = ["store_id", "item_id", "wday", "weekend_plus"]
for col in cat_features:
    X_trainval[col] = LabelEncoder().fit_transform(X_trainval[col]).fillna('Null')
    X_trainval[col] = X_trainval[col].astype('int16')


y_test = test_df["demand"].values
X_test = test_df.drop(columns=["demand", "d"]).values

for col in cat_features:
    X_test[col] = LabelEncoder().fit_transform(X_test[col]).fillna('Null')
    X_test[col] = X_test[col].astype('int16')

# Define the objective and metric for PGBM
def objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()
    return loss

# Hyperparameter tuning with Optuna
print('Starting hyperparameter optimization...')
start_time_hp = time.time()

class Objective(object):
    def __init__(self, X_train, y_train, dataset_name):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_name = dataset_name
        
    def __call__(self, trial):
        try:
            params = {
                'n_estimators': 2000,
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
                'max_leaves': trial.suggest_int('max_leaves', 8, 32),
                'max_bin': trial.suggest_int('max_bin', 32, 128),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),  # Constant for this example
                'early_stopping_rounds': 20,
                'device': 'gpu' if torch.cuda.is_available() else 'cpu',
                'distribution': 'normal',
                'verbose': 0
            }
            model = PGBMRegressor()
            model.set_params(**params)
            score = np.mean(cross_val_score(model, self.X_train, self.y_train, cv=3, n_jobs=3, scoring='neg_root_mean_squared_error', error_score="raise"))
            return score
    
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")

study = optuna.create_study(direction='maximize')
objective_tuning = Objective(X_trainval, y_trainval)

time_limit = 86400
study.optimize(objective_tuning, n_trials=20, timeout=time_limit)  # 1 day timeout


end_time_hp = time.time()
elapsed_time_hp = end_time_hp - start_time_hp
print(f"Hyperparameter optimization completed in {elapsed_time_hp:.2f} seconds")

# Get the best parameters
best_params = study.best_params
print(f"Best hyperparameters: {best_params}")

# Add fixed parameters to the best params
best_params.update({
    'n_estimators': 2000,
    'early_stopping_rounds': 20,
    'device': 'gpu' if torch.cuda.is_available() else 'cpu',
    'distribution': 'normal',
    'verbose': 1
})

# Train the final model
print("Training final model...")
start_time_train = time.time()
model = PGBM()
model.train((X_trainval, y_trainval), objective=objective, metric=rmseloss_metric, params=best_params)
end_time_train = time.time()
training_time = end_time_train - start_time_train
print(f"Model training completed in {training_time:.2f} seconds")

# Generate predictions
print("Generating predictions...")
n_forecasts = 200

# Define quantiles to evaluate
quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

# Compute quantile predictions
quantile_preds = {}
quantile_losses = []
for q in quantiles:
    q_key = f"q{q:.3f}"
    quantile_preds[q_key] = norm.ppf(q, loc=mu.cpu().numpy(), scale=std)
    q_loss = quantile_loss(q, y_test, quantile_preds[q_key]).mean()
    quantile_losses.append(q_loss)

# Average quantile loss
wql_avg = np.mean(quantile_losses)

# Create a dataframe for detailed per-sample metrics
per_sample_metrics = pd.DataFrame({
    "test_index": np.arange(len(y_test)),
    "actual": y_test,
    "predicted_mean": mu.cpu().numpy(),
    "predicted_std": std
})

# Add quantile predictions to the dataframe
for q in quantiles:
    q_key = f"q{q:.3f}"
    per_sample_metrics[q_key] = quantile_preds[q_key]

# Add cluster info and parameters
per_sample_metrics["cluster_id"] = cluster_id
per_sample_metrics["model"] = method_name
per_sample_metrics["rmse"] = rmse
per_sample_metrics["nll"] = nll
per_sample_metrics["crps"] = crps_score
per_sample_metrics["wql_avg"] = wql_avg
per_sample_metrics["training_time"] = training_time
per_sample_metrics["hp_time"] = elapsed_time_hp
per_sample_metrics["n_estimators"] = model.best_iteration if hasattr(model, 'best_iteration') else best_params['n_estimators']

# Create directory for detailed logs
os.makedirs("logs/m5", exist_ok=True)

# Save to CSV
log_file = f"logs/m5/pgbm_clusters_detailed_scores.csv"

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

# Save summary results
os.makedirs("results/clusters/local", exist_ok=True)

results = {
    "cluster_id": cluster_id,
    "model": method_name,
    "rmse": float(rmse),
    "nll": float(nll),
    "crps": float(crps_score),
    "crps_cal": float(crps_cal),
    "crps_sha": float(crps_sha),
    "wql_avg": float(wql_avg),
    "n_estimators": int(model.best_iteration if hasattr(model, 'best_iteration') else best_params['n_estimators']),
    "training_time": float(training_time),
    "hp_time": float(elapsed_time_hp),
    "best_params": {k: float(v) if isinstance(v, np.float64) else v for k, v in best_params.items() if k != 'device' and k != 'distribution' and k != 'verbose' and k != 'early_stopping_rounds' and k != 'n_estimators'}
}

with open(f"results/clusters/local/pgbm_cluster_{cluster_id}_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Results saved to results/clusters/local/pgbm_cluster_{cluster_id}_results.json")
import os, sys, json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from xgboostlss.distributions.NegativeBinomial import *
from xgboostlss.distributions.ZINB import *
from xgboostlss.distributions.Poisson import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.safety import apply_safety_net

np.random.seed(123)

# Get cluster ID and other parameters from command line
if len(sys.argv) < 2:
    print("Usage: python m5_xgblssboost_HP_single_run.py <cluster_id> [mode] [natural_grad] [stabilization] [clip_value] [standardize]")
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
X_trainval = train_df.drop(columns=["demand", "d", "item_id", "store_id"])

y_test = test_df["demand"]
X_test = test_df.drop(columns=["demand", "d", "item_id", "store_id"])

# X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)  # final split: 60/20/20

# Define model with parameters received from command line
# xgblss = XGBoostLSS(
#     Gaussian(
#         stabilization=stabilization, 
#         response_fn=mode, 
#         loss_fn="nll"
#         )
# )

if dist == "Gaussian":
    xgblss = XGBoostLSS(
        Gaussian(
            stabilization=stabilization,
            response_fn=mode,
            loss_fn="nll"
        )
    )
elif dist == "NegativeBinomial":
    # Define model with parameters received from command line
    xgblss = XGBoostLSS(
        NegativeBinomial(
            stabilization=stabilization,
            response_fn_total_count=mode,
        )
    )
else:
    raise ValueError(f"Unknown distribution: {dist}")

# If standardize is True, standardize the target
if standardize:
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    xgblss.start_values = np.array([0, 1])  # standard normal
else:
    xgblss.start_values = np.array([np.array(0.5) for _ in range(xgblss.dist.n_dist_param)])

param_dict = {
    "eta": ["float", {"low": 1e-5, "high": 0.4, "log": True}],
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "min_child_weight": ["int", {"low": 1, "high": 100, "log": True}],
    "subsample": ["float", {"low": 0.5, "high": 1.0, "log": False}],
    #'device':  ["categorical", ['cuda']],
}

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)

# HP tuning
opt_params = xgblss.hyper_opt(
    param_dict, dtrain,
    num_boost_round=2000,
    nfold=5,
    early_stopping_rounds=20,
    max_minutes=1440,
    n_trials=80,
    silence=True,
    seed=1,
    hp_seed=1,
)

n_rounds = opt_params.pop("opt_rounds")

# Train final model
dtrain_final = xgb.DMatrix(X_trainval, label=y_trainval)
model = xgblss.train(opt_params, dtrain_final, num_boost_round=n_rounds)

# Predict and evaluate
dtest = xgb.DMatrix(X_test)
forecast = xgblss.predict(dtest)
#forecast = apply_safety_net(forecast, y_trainval.values)

# Evaluate at quantiles
quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
q_preds = xgblss.predict(dtest, pred_type="quantiles",
                               n_samples=200,
                               quantiles=quantiles)

# Create directory for detailed logs
os.makedirs("logs/m5", exist_ok=True)

# print detailed per-sample metrics
print("Detailed per-sample metrics:")
print(f"Actual values: {y_test.values}")
print(f"Quantile predictions: {q_preds}")

# Create a dataframe with per-sample metrics
per_sample_metrics = pd.DataFrame({
    "test_index": np.arange(len(y_test)),
    "actual": y_test.values,
})

per_sample_metrics = per_sample_metrics.merge(q_preds, left_index=True, right_index=True)

print(per_sample_metrics)

# Add cluster info and parameters
per_sample_metrics["cluster_id"] = cluster_id
per_sample_metrics["mode"] = mode
per_sample_metrics["natural_grad"] = natural_grad
per_sample_metrics["stabilization"] = stabilization
per_sample_metrics["n_rounds"] = n_rounds
per_sample_metrics["dist"] = dist

# Save to CSV
log_file = f"logs/m5/xgblss_clusters_detailed_scores_.csv"

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
    "n_rounds": n_rounds
}
with open(f"results/clusters/local/cluster_{cluster_id}_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Results saved to results/clusters/local/cluster_{cluster_id}_results.json")
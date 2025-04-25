import os, sys, json, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import torch
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from lightgbmlss.distributions.NegativeBinomial import *
from lightgbmlss.distributions.Poisson import *
from lightgbmlss.distributions.ZINB import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import crps, quantile_loss
from utils.safety import apply_safety_net

np.random.seed(123)

# Get cluster ID and other parameters from command line
if len(sys.argv) < 2:
    print("Usage: python m5_lssboost_HP_single_run.py <cluster_id> [mode] [natural_grad] [stabilization] [clip_value] [standardize]")
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

cat_features = ["store_id", "item_id", "wday", "weekend_plus"]

# Convert categorical columns to consecutive integers starting from 0
for col in cat_features:
    # Create a mapping of original values to consecutive integers
    unique_values = df[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    # Apply the mapping to all datasets
    df[col] = df[col].map(mapping)

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

# X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)  # final split: 60/20/20

if dist == "Gaussian":
    lgblss = LightGBMLSS(
        Gaussian(
            stabilization=stabilization,
            response_fn=mode,
            loss_fn="nll"
        )
    )
elif dist == "NegativeBinomial":
    # Define model with parameters received from command line
    lgblss = LightGBMLSS(
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
    lgblss.start_values = np.array([0, 1])  # standard normal
else:
    #lgblss.start_values = np.array([np.mean(y_train), np.std(y_train)])
    lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])

param_dict = {
    "eta": ["float", {"low": 1e-5, "high": 1e-1, "log": True}],
    "max_depth": ["int", {"low": 2, "high": 10, "log": False}],
    "num_leaves": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "min_data_in_leaf": ["int", {"low": 20, "high": 100, "log": False}],  # Constant for this example
    "lambda_l1": ["float", {"low": 1e-8, "high": 10, "log": True}],
    "histogram_pool_size": ["int", {"low": 1e3, "high": 5e3, "log": True}],
    "feature_pre_filter": ["categorical", [False]],
    #'device': ["categorical", ['cuda']],
    #'categorical_feature': ["categorical", [cat_features]],
}

# if cuda is available, use it
param_dict["device"] = ["categorical", ['cuda']] if torch.cuda.is_available() else ["categorical", ['cpu']]

dtrain = lgb.Dataset(X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
dtrain_final = lgb.Dataset(X_trainval, y_trainval, categorical_feature=cat_features, free_raw_data=False)
dtest = lgb.Dataset(X_test, y_test, categorical_feature=cat_features, free_raw_data=False)


# HP tuning
opt_params = lgblss.hyper_opt(
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

print(f"Best number of boosters: {n_rounds}")

# Train final model
model = lgblss.train(opt_params, dtrain_final, num_boost_round=n_rounds, categorical_feature=cat_features)

# Predict and evaluate
#forecast = lgblss.predict(dtest)
#forecast = apply_safety_net(forecast, y_trainval.values)

# rmse = np.sqrt(mean_squared_error(forecast["loc"], y_test))
# nll = -norm(forecast["loc"], forecast["scale"]).logpdf(y_test).mean()
# crps_total, crps_cal, crps_sha = crps(y_test, samples)

# Evaluate at quantiles
quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
q_preds = lgblss.predict(X_test, pred_type="quantiles",
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
log_file = f"logs/m5/clusters_detailed_scores_cat_cpu.csv"

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


#print(f"✅ Detailed scores saved to {log_file}")

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
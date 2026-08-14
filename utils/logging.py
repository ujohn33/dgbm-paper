import os
import csv

import numpy as np

def log_predictions(fold, dataset_name, y_test, mu, std, quantile_preds, file_path="logs/openml/predictions_PBGM.csv"):
    """Logs the predictions to a CSV file."""
    header = ["fold", "dataset", "true_value", "mu", "std", "quantile_0.1", "quantile_0.5", "quantile_0.9"]

    # Create the parent directory if it does not exist yet: the log directories
    # are gitignored, so they are absent from a fresh clone.
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Coerce everything to flat arrays. Callers pass pandas Series whose index is
    # a permutation of the original rows, so positional indexing below would
    # otherwise do label lookup and raise KeyError.
    y_test = np.asarray(y_test).ravel()
    mu = np.asarray(mu).ravel()
    std = np.asarray(std).ravel()
    quantile_preds = {k: np.asarray(v).ravel() for k, v in quantile_preds.items()}

    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    # Open the file in append mode ('a+')
    with open(file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file does not exist or is empty
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(header)

        # Write each prediction for the current fold
        for i in range(len(y_test)):
            row_to_write = [
                fold,
                dataset_name,
                float(y_test[i].item()) if hasattr(y_test[i], 'item') else y_test[i],  # Convert tensor to float
                float(mu[i]) if hasattr(mu[i], 'item') else mu[i],  # Ensure mu is converted
                float(std[i]) if hasattr(std[i], 'item') else std[i],  # Ensure std is converted
                float(quantile_preds['0.1'][i]) if hasattr(quantile_preds['0.1'][i], 'item') else quantile_preds['0.1'][i],
                float(quantile_preds['0.5'][i]) if hasattr(quantile_preds['0.5'][i], 'item') else quantile_preds['0.5'][i],
                float(quantile_preds['0.9'][i]) if hasattr(quantile_preds['0.9'][i], 'item') else quantile_preds['0.9'][i]
            ]
            writer.writerow(row_to_write)

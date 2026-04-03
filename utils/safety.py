import numpy as np
import pandas as pd

def apply_safety_net(forecast: pd.DataFrame, train_labels: np.ndarray) -> pd.DataFrame:
    """
    Apply a safety net to prevent implosion of predictions.
    If forecast['loc'] is 100x higher than the mean of the train labels, replace it with the mean.

    Parameters:
    - forecast: pd.DataFrame with predictions, including a 'loc' column.
    - train_labels: np.ndarray with training labels to compute the mean.

    Returns:
    - Adjusted forecast DataFrame.
    """
    mean_label = np.mean(train_labels)
    safety_threshold = 100 * mean_label
    forecast['loc'] = forecast['loc'].apply(
        lambda x: mean_label if abs(x) > abs(safety_threshold) else x
    )
    std_label = np.std(train_labels)
    safety_threshold = 100 * std_label
    forecast['scale'] = forecast['scale'].apply(
        lambda x: std_label if abs(x) > abs(safety_threshold) else x
    )
    return forecast
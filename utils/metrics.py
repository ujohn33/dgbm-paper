import numpy as np

def crps(y_true, y_pred):
    """
    Computes the Continuous Ranked Probability Score (CRPS).

    Args:
        y_true (array-like): True values of the target variable, of shape (n_samples,)
        y_pred (array-like): Predicted values of the target variable, of shape (n_samples, n_forecasts)

    Returns:
        float: CRPS score.
    """
    # Ensure y_pred is a numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute absolute differences between forecasts and true values
    abs_diff_obs = np.abs(y_pred - y_true[:, None])
    crps_term1 = np.mean(abs_diff_obs)

    # Compute absolute differences among all pairs of forecasts
    abs_diff_members = np.abs(y_pred[:, :, None] - y_pred[:, None, :])
    crps_term2 = np.mean(abs_diff_members) / 2

    # Compute the CRPS score
    crps_score = crps_term1 - crps_term2

    return crps_score, crps_term1, crps_term2
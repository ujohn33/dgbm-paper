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
    n_samples = y_true.shape[0]
    n_forecasts = y_pred.shape[1]

    # Compute the first term of the CRPS formula
    crps_term1 = 0
    for t in range(n_samples):
        for i in range(n_forecasts):
            crps_term1 += abs(y_pred[t, i] - y_true[t])

    crps_term1 /= n_samples * n_forecasts

    # Compute the second term of the CRPS formula
    crps_term2 = 0
    for t in range(n_samples):
        for i in range(n_forecasts):
            for j in range(n_forecasts):
                crps_term2 += abs(y_pred[t, i] - y_pred[t, j])

    crps_term2 /= 2 * n_samples * n_forecasts ** 2

    # Compute the CRPS score
    crps_score = crps_term1 - crps_term2

    return crps_score, crps_term1, crps_term2
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def adjusted_r2(r2, n, p):
    """Retorna R² ajustado."""
    if n - p - 1 <= 0:
        return r2
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def calculate_metrics(y_true, y_pred, n_params):
    r2 = r2_score(y_true, y_pred)
    r2_adj = adjusted_r2(r2, len(y_true), n_params)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mean_y = y_true.mean()
    std_y = y_true.std()
    amplitude = y_true.max() - y_true.min()

    nrmse_range = rmse / amplitude if amplitude > 0 else np.nan
    cv_rmse = rmse / mean_y if mean_y != 0 else np.nan
    mae_pct = mae / mean_y if mean_y != 0 else np.nan

    return {
        "r2": r2,
        "r2_adj": r2_adj,
        "rmse": rmse,
        "mae": mae,
        "mean_y": mean_y,
        "std_y": std_y,
        "amplitude": amplitude,
        "max_y": y_true.max(),
        "min_y": y_true.min(),
        "nrmse_range": nrmse_range,
        "cv_rmse": cv_rmse,
        "mae_pct": mae_pct
    }

def get_quality_label(val, thresholds, labels):
    for t, lab in zip(thresholds, labels):
        if val <= t:
            return lab
    return labels[-1]

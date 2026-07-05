import numpy as np
from scipy.optimize import curve_fit


def count_model_parameters(model):
    """Número de parâmetros ajustáveis de um modelo baseado em _model_func."""
    return model._model_func.__code__.co_argcount - 2


def fit_logarithmic_model(model, X, y):
    """Ajusta o modelo em escala logarítmica (minimiza erro relativo)."""
    if np.any(y <= 0):
        raise ValueError("o ajuste em escala logarítmica exige valores de MR positivos.")

    n_params = count_model_parameters(model)
    p0 = [max(float(np.median(y)), 1e-9)] + [1.0] * (n_params - 1)
    lower_bounds = [1e-12] + [-np.inf] * (n_params - 1)
    upper_bounds = [np.inf] * n_params

    def log_model_func(X_flat, *params):
        predicted = model._model_func(X_flat, *params)
        if np.any(predicted <= 0):
            return np.full_like(predicted, np.inf, dtype=float)
        return np.log(predicted)

    model._params, _ = curve_fit(
        log_model_func,
        X,
        np.log(y),
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=200000,
    )

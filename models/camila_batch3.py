import numpy as np
from scipy.optimize import curve_fit
from .base_model import BaseModel

class Witczak1981Model(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        return k1 * (theta**k2 / self.Pa) * (sd**k3 / self.Pa)

    def fit(self, X, y):
        p0 = [y.mean() * (self.Pa**2), 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Witczak (1981)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} (θ^{{{k2:.4f}}}/{self.Pa:.6f}) (σ_d^{{{k3:.4f}}}/{self.Pa:.6f})$$"

class Pezo1993Model(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        return k1 * self.Pa * (s3 / self.Pa)**k2 * (sd / self.Pa)**k3

    def fit(self, X, y):
        p0 = [y.mean() / self.Pa, 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Pezo (1993)"

    def get_equation(self):
        k1, k2, k3 = self._params
        const = k1 * self.Pa
        return f"$$MR = {const:.4f} (σ_3/P_a)^{{{k2:.4f}}} (σ_d/P_a)^{{{k3:.4f}}}$$"

    def get_coefficients(self):
        k1, k2, k3 = self._params
        return [
            ("k1", float(k1)),
            ("k2", float(k2)),
            ("k3", float(k3)),
            ("k1·Pa (constante efetiva da equação)", float(k1 * self.Pa)),
        ]

class Pezo1993NonNormalizedModel(BaseModel):
    def __init__(self):
        self._params = None

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        return k1 * s3**k2 * sd**k3

    def fit(self, X, y):
        mean_s3 = X[:, 0].mean()
        mean_sd = X[:, 1].mean()
        k1_0 = y.mean() / (mean_s3 * mean_sd)
        self._params, _ = curve_fit(self._model_func, X, y, p0=[k1_0, 1.0, 1.0], maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Pezo (1993) - Não normalizada"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} σ_3^{{{k2:.4f}}} σ_d^{{{k3:.4f}}}$$"

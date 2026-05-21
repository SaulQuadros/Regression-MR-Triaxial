import numpy as np
from scipy.optimize import curve_fit
from .base_model import BaseModel

class DunlapModel(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2):
        s3 = X_flat[:, 0]
        return k1 * (s3 / self.Pa)**k2

    def fit(self, X, y):
        p0 = [y.mean(), 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Dunlap (1963)"

    def get_equation(self):
        k1, k2 = self._params
        return f"$$MR = {k1:.4f} (σ₃/{self.Pa:.6f})^{{{k2:.4f}}}$$"

class HicksModel(BaseModel):
    def __init__(self):
        self._params = None

    def _model_func(self, X_flat, k1, k2):
        sd = X_flat[:, 1]
        return k1 * sd**k2

    def fit(self, X, y):
        p0 = [y.mean(), 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Hicks (1970)"

    def get_equation(self):
        k1, k2 = self._params
        return f"$$MR = {k1:.4f} σ_d^{{{k2:.4f}}}$$"

class UzanModel(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        return k1 * (theta / self.Pa)**k2 * (sd / self.Pa)**k3

    def fit(self, X, y):
        p0 = [y.mean(), 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Uzan (1985)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} (θ/{self.Pa:.6f})^{{{k2:.4f}}} (σ_d/{self.Pa:.6f})^{{{k3:.4f}}}$$"

class JohnsonModel(BaseModel):
    def __init__(self):
        self._params = None

    def _model_func(self, X_flat, k1, k2):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        return k1 * theta**k2

    def fit(self, X, y):
        p0 = [y.mean(), 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Johnson et al. (1986)"

    def get_equation(self):
        k1, k2 = self._params
        return f"$$MR = {k1:.4f} θ^{{{k2:.4f}}}$$"

class WitczakUzan1988Model(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        tau_oct = 0.471 * sd
        return k1 * self.Pa * (theta / self.Pa)**k2 * (tau_oct / self.Pa)**k3

    def fit(self, X, y):
        p0 = [y.mean() / self.Pa, 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Witczak e Uzan (1988)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} P_a (θ/P_a)^{{{k2:.4f}}} (τ_{{oct}}/P_a)^{{{k3:.4f}}}$$"

class TamBrownModel(BaseModel):
    def __init__(self):
        self._params = None

    def _model_func(self, X_flat, k1, k2):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        sigma_oct = (sd + 3 * s3) / 3
        return k1 * (sigma_oct / sd)**k2

    def fit(self, X, y):
        p0 = [y.mean(), 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Tam e Brown (1988)"

    def get_equation(self):
        k1, k2 = self._params
        return f"$$MR = {k1:.4f} (σ_{{oct}}/σ_d)^{{{k2:.4f}}}$$"

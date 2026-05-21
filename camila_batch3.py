import numpy as np
from scipy.optimize import curve_fit
from .base_model import BaseModel

class HopkinsModel(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        tau_oct = 0.471 * sd
        return k1 * (s3 / self.Pa + 1)**k2 * (tau_oct / self.Pa + 1)**k3

    def fit(self, X, y):
        p0 = [y.mean(), 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Hopkins et al. (2001)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} (σ_3/P_a + 1)^{{{k2:.4f}}} (τ_{{oct}}/P_a + 1)^{{{k3:.4f}}}$$"

class NiModel(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        return k1 * self.Pa * (s3 / self.Pa + 1)**k2 * (sd / self.Pa + 1)**k3

    def fit(self, X, y):
        p0 = [y.mean() / self.Pa, 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Ni et al. (2002)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} P_a (σ_3/P_a + 1)^{{{k2:.4f}}} (σ_d/P_a + 1)^{{{k3:.4f}}}$$"

class NCHRP1_28AModel(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        return k1 * self.Pa * (theta / self.Pa)**k2 * (sd / self.Pa + 1)**k3

    def fit(self, X, y):
        p0 = [y.mean() / self.Pa, 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "NCHRP 1-28A (2004)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} P_a (θ/P_a)^{{{k2:.4f}}} (σ_d/P_a + 1)^{{{k3:.4f}}}$$"

class NCHRP1_37AModel(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        tau_oct = 0.471 * sd
        return k1 * self.Pa * (theta / self.Pa)**k2 * (tau_oct / self.Pa + 1)**k3

    def fit(self, X, y):
        p0 = [y.mean() / self.Pa, 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "NCHRP 1-37A (2004)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} P_a (θ/P_a)^{{{k2:.4f}}} (τ_{{oct}}/P_a + 1)^{{{k3:.4f}}}$$"

class Ooi1Model(BaseModel):
    def __init__(self):
        self._params = None
        self.Pa = 0.101325

    def _model_func(self, X_flat, k1, k2, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        return k1 * self.Pa * (theta / self.Pa + 1)**k2 * (sd / self.Pa + 1)**k3

    def fit(self, X, y):
        p0 = [y.mean() / self.Pa, 1.0, 1.0]
        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Ooi et al. (1) (2004)"

    def get_equation(self):
        k1, k2, k3 = self._params
        return f"$$MR = {k1:.4f} P_a (θ/P_a + 1)^{{{k2:.4f}}} (σ_d/P_a + 1)^{{{k3:.4f}}}$$"

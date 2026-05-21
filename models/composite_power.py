import numpy as np
from scipy.optimize import curve_fit
from .base_model import BaseModel

class CompositePowerModel(BaseModel):
    def __init__(self):
        self._params = None

    @staticmethod
    def _model_func(X_flat, a1, k1, a2, k2, a3, k3):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        return a1 * s3**k1 + a2 * (s3 * sd)**k2 + a3 * sd**k3

    def fit(self, X, y):
        mean_y = y.mean()
        mean_s3 = X[:, 0].mean()
        mean_sd = X[:, 1].mean()
        mean_s3sd = (X[:, 0] * X[:, 1]).mean()

        p0 = [mean_y/mean_s3, 1.0,
              mean_y/mean_s3sd, 1.0,
              mean_y/mean_sd, 1.0]

        self._params, _ = curve_fit(self._model_func, X, y, p0=p0, maxfev=200000)

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "Potência Composta (Genérico)"

    def get_equation(self):
        a1, k1, a2, k2, a3, k3 = self._params
        terms = [
            (a1, f"σ₃^{{{k1:.4f}}}"),
            (a2, f"(σ₃σ_d)^{{{k2:.4f}}}"),
            (a3, f"σ_d^{{{k3:.4f}}}"),
        ]
        eq = "$$MR = "
        eq += f"{terms[0][0]:.4f}{terms[0][1]}"
        for coef, term in terms[1:]:
            sign = " + " if coef >= 0 else " - "
            eq += f"{sign}{abs(coef):.4f}{term}"
        eq += "$$"
        return eq

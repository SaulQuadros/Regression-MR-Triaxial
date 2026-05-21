import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class PolynomialModel(BaseModel):
    def __init__(self, degree=2, include_intercept=True):
        self.degree = degree
        self.include_intercept = include_intercept
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.reg = LinearRegression(fit_intercept=include_intercept)
        self._coefs = None
        self._intercept = 0.0
        self._feature_names = None

    def fit(self, X, y):
        Xp = self.poly.fit_transform(X)
        self.reg.fit(Xp, y)
        self._coefs = self.reg.coef_
        self._intercept = self.reg.intercept_ if self.include_intercept else 0.0
        self._feature_names = self.poly.get_feature_names_out(["σ₃", "σ_d"])

    def predict(self, X):
        Xp = self.poly.transform(X)
        return self.reg.predict(Xp)

    @property
    def name(self):
        suffix = "com Intercepto" if self.include_intercept else "sem Intercepto"
        return f"Polinomial {suffix} (Grau {self.degree})"

    @property
    def intercept(self):
        return self._intercept

    def get_equation(self):
        terms_per_line = 4
        parts = []
        for coef, term in zip(self._coefs, self._feature_names):
            sign = " + " if coef >= 0 else " - "
            parts.append(f"{sign}{abs(coef):.4f}{term.replace(' ', '')}")
        
        lines = []
        if self.include_intercept:
            curr = f"MR = {self._intercept:.4f}"
        else:
            curr = f"MR = {self._coefs[0]:.4f}{self._feature_names[0].replace(' ', '')}"
            parts = parts[1:]

        for i, part in enumerate(parts):
            curr += part
            if (i + 1) % terms_per_line == 0:
                lines.append(curr)
                curr = ""
        if curr.strip():
            lines.append(curr)
        
        return "$$" + " \\\\ \n".join(lines) + "$$"

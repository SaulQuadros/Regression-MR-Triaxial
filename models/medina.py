import numpy as np
from scipy.optimize import curve_fit
from .base_model import BaseModel

class MeDiNaModel(BaseModel):
    """Expressão geral do MeDiNa (Método de Dimensionamento Nacional).

    MR = k1 · σ3^k2 · σd^k3 · θ^k4,  com θ = σd + 3·σ3

    O Manual do MeDiNa reúne todos os modelos constitutivos em uma única
    expressão matemática; cada comportamento é obtido zerando parâmetros
    (ex.: "modelo composto" => k4=0; "dependente do invariante" => k2=k3=0).
    Esta classe mantém os quatro parâmetros livres, sendo a forma mais geral.
    """

    def __init__(self):
        self._params = None

    def _model_func(self, X_flat, k1, k2, k3, k4):
        s3, sd = X_flat[:, 0], X_flat[:, 1]
        theta = sd + 3 * s3
        return k1 * s3**k2 * sd**k3 * theta**k4

    def fit(self, X, y):
        mean_s3 = X[:, 0].mean()
        mean_sd = X[:, 1].mean()
        # chute inicial reduzindo ao composto (k4 = 0), com expoentes unitários
        k1_0 = y.mean() / (mean_s3 * mean_sd)
        self._params, _ = curve_fit(
            self._model_func, X, y, p0=[k1_0, 1.0, 1.0, 0.0], maxfev=200000
        )

    def predict(self, X):
        return self._model_func(X, *self._params)

    @property
    def name(self):
        return "MeDiNa (expressão geral)"

    def get_equation(self):
        k1, k2, k3, k4 = self._params
        return (
            f"$$MR = {k1:.4f} σ_3^{{{k2:.4f}}} σ_d^{{{k3:.4f}}} θ^{{{k4:.4f}}}"
            f"\\quad (θ = σ_d + 3σ_3)$$"
        )

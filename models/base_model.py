from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Ajusta o modelo aos dados X (σ3, σd) e y (MR)."""
        pass

    @abstractmethod
    def predict(self, X):
        """Prediz os valores de MR para as tensões X."""
        pass

    @abstractmethod
    def get_equation(self):
        """Retorna a representação em LaTeX da equação ajustada."""
        pass

    @property
    @abstractmethod
    def name(self):
        """Retorna o nome do modelo."""
        pass

    @property
    def intercept(self):
        """Retorna o intercepto do modelo, se aplicável."""
        return 0.0

    def get_equation_note(self):
        """Nota complementar exibida em linha própria abaixo da equação.

        Retorna ``None`` quando não há nota (padrão). Modelos que precisam
        explicitar definições auxiliares (ex.: o MeDiNa, com θ = σd + 3σ3)
        sobrescrevem este método.
        """
        return None

    def get_coefficients(self):
        """Coeficientes calibrados como lista ordenada de (rótulo, valor).

        Inclui apenas os coeficientes efetivamente presentes na equação do
        modelo. Os rótulos (k1, k2, k3, k4, ...) são lidos da própria
        assinatura de ``_model_func``, de modo que, por exemplo, k4 só aparece
        em modelos que o utilizam (como o MeDiNa).
        """
        params = getattr(self, "_params", None)
        model_func = getattr(self, "_model_func", None)
        if params is None or model_func is None:
            return []
        code = model_func.__code__
        arg_names = list(code.co_varnames[:code.co_argcount])
        labels = arg_names[-len(params):]
        return [(label, float(value)) for label, value in zip(labels, params)]

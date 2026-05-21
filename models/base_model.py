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

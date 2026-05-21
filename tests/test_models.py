import numpy as np
import pytest
from models import MODELS_MAP

@pytest.fixture
def sample_data():
    # Dados fictícios coerentes com ensaios triaxiais
    # σ3 (MPa), σd (MPa)
    X = np.array([
        [0.02, 0.02], [0.02, 0.04], [0.02, 0.06],
        [0.05, 0.05], [0.05, 0.10], [0.05, 0.15],
        [0.10, 0.10], [0.10, 0.20], [0.10, 0.30]
    ])
    # MR (MPa) - simulando algum comportamento crescente com as tensões
    y = 500 * X[:, 0] + 200 * X[:, 1] + 100
    return X, y

@pytest.mark.parametrize("model_name", list(MODELS_MAP.keys()))
def test_all_models_fit_predict(model_name, sample_data):
    X, y = sample_data
    model_class = MODELS_MAP[model_name]

    # Alguns modelos (polinomiais) são lambdas no MODELS_MAP
    model = model_class()

    # Testar fit
    model.fit(X, y)

    # Testar predict
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert not np.any(np.isnan(y_pred))

    # Testar get_equation
    eq = model.get_equation()
    assert isinstance(eq, str)
    assert eq.startswith("$$") and eq.endswith("$$")

    # Testar name
    assert isinstance(model.name, str)
    assert len(model.name) > 0

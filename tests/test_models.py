import numpy as np
import pytest
from models import MODELS_MAP, MeDiNaModel

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


def test_medina_registrado_com_nome_correto():
    assert "MeDiNa (expressão geral)" in MODELS_MAP
    assert MODELS_MAP["MeDiNa (expressão geral)"] is MeDiNaModel
    assert MeDiNaModel().name == "MeDiNa (expressão geral)"


def test_medina_tem_quatro_coeficientes_incluindo_k4(sample_data):
    X, y = sample_data
    model = MeDiNaModel()
    model.fit(X, y)

    # A expressão geral do MeDiNa possui k1, k2, k3 e k4
    assert len(model._params) == 4

    labels = [label for label, _ in model.get_coefficients()]
    assert labels == ["k1", "k2", "k3", "k4"]

    # A equação deve conter o invariante de tensões θ
    assert "θ" in model.get_equation()


def test_medina_recupera_parametros_conhecidos():
    # Dados gerados pela própria equação MR = k1·σ3^k2·σd^k3·θ^k4 (θ = σd + 3σ3)
    X = np.array([
        [0.02, 0.02], [0.02, 0.04], [0.02, 0.06],
        [0.05, 0.05], [0.05, 0.10], [0.05, 0.15],
        [0.10, 0.10], [0.10, 0.20], [0.10, 0.30],
    ])
    k1, k2, k3, k4 = 12.0, 0.45, -0.20, 0.30
    s3, sd = X[:, 0], X[:, 1]
    theta = sd + 3 * s3
    y = k1 * s3**k2 * sd**k3 * theta**k4

    model = MeDiNaModel()
    model.fit(X, y)

    np.testing.assert_allclose(model._params, [k1, k2, k3, k4], rtol=1e-4)
    np.testing.assert_allclose(model.predict(X), y, rtol=1e-6)

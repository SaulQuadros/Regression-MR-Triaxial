import numpy as np
import pytest
from models import MODELS_MAP, MeDiNaModel, Pezo1993Model

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


@pytest.mark.parametrize("pa", [1.0, 0.101325])
def test_pezo_normalizado_expoe_constante_efetiva(sample_data, pa):
    X, y = sample_data
    model = Pezo1993Model()
    model.Pa = pa
    model.fit(X, y)

    k1, k2, k3 = model._params
    coefficients = dict(model.get_coefficients())

    # Mantém k1, k2, k3 e acrescenta a constante efetiva k1·Pa
    assert coefficients["k1"] == pytest.approx(k1)
    assert coefficients["k2"] == pytest.approx(k2)
    assert coefficients["k3"] == pytest.approx(k3)

    effective_label = "k1·Pa (constante efetiva da equação)"
    assert effective_label in coefficients
    assert coefficients[effective_label] == pytest.approx(k1 * pa)

    # A constante efetiva deve coincidir com o coeficiente inicial da equação
    equation_leading = float(model.get_equation().split("MR = ")[1].split(" ")[0])
    assert equation_leading == pytest.approx(k1 * pa, rel=1e-3)


def test_geracao_relatorio_pdf(sample_data):
    pytest.importorskip("reportlab")
    import pandas as pd
    from utils.metrics import calculate_metrics
    from utils.plotting import plot_3d_surface
    from utils.reports import generate_pdf_doc

    X, y = sample_data
    df = pd.DataFrame({"σ3": X[:, 0], "σd": X[:, 1], "MR": y})

    model = MeDiNaModel()
    model.fit(X, y)
    metrics = calculate_metrics(y, model.predict(X), len(model._params))
    fig = plot_3d_surface(df, model)

    pdf_buf = generate_pdf_doc(
        model, metrics, df, fig, "Normal",
        {"Analista": "Teste"}, {"Modelo ajustado": model.name},
    )
    data = pdf_buf.getvalue()

    # Deve ser um PDF válido e não trivial
    assert data[:4] == b"%PDF"
    assert len(data) > 1000

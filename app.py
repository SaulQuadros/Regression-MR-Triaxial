import sys
import os
from pathlib import Path

# Adiciona o diretório atual ao sys.path para garantir que os módulos models e utils sejam encontrados no deploy
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Tenta importar os módulos customizados com tratamento de erro amigável
try:
    from models import MODELS_MAP
    from utils.metrics import calculate_metrics, get_quality_label
    from utils.plotting import plot_3d_surface
except ImportError as e:
    st.error(f"❌ Erro crítico de importação: {e}. Verifique se as pastas 'models' e 'utils' (e seus arquivos __init__.py) foram enviados corretamente para o repositório.")
    st.stop()


def build_template_workbook() -> io.BytesIO:
    template_df = pd.DataFrame(
        {
            "σ3": [0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.10, 0.10, 0.10],
            "σd": [0.02, 0.04, 0.06, 0.05, 0.10, 0.15, 0.10, 0.20, 0.30],
            "MR": [114.0, 118.0, 122.0, 135.0, 145.0, 155.0, 170.0, 190.0, 210.0],
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="dados")
    buffer.seek(0)
    return buffer


def count_model_parameters(model) -> int:
    return model._model_func.__code__.co_argcount - 2


def fit_logarithmic_model(model, X, y):
    from scipy.optimize import curve_fit

    if np.any(y <= 0):
        raise ValueError("o ajuste em escala logarítmica exige valores de MR positivos.")

    n_params = count_model_parameters(model)
    p0 = [max(float(np.median(y)), 1e-9)] + [1.0] * (n_params - 1)
    lower_bounds = [1e-12] + [-np.inf] * (n_params - 1)
    upper_bounds = [np.inf] * n_params

    def log_model_func(X_flat, *params):
        predicted = model._model_func(X_flat, *params)
        if np.any(predicted <= 0):
            return np.full_like(predicted, np.inf, dtype=float)
        return np.log(predicted)

    model._params, _ = curve_fit(
        log_model_func,
        X,
        np.log(y),
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=200000,
    )

st.set_page_config(page_title="Modelos de MR - Camila Carvalho (2023)", layout="wide")
st.title("Modelos de Regressão para MR")
st.markdown("""
Esta ferramenta permite o ajuste de modelos matemáticos para o Módulo de Resiliência (MR), 
incluindo os 14 modelos selecionados na dissertação de **Camila Luiza Mello Carvalho (UFJF, 2023)**.
""")

# --- Sidebar ---
modelagem_tab, rastreabilidade_tab = st.sidebar.tabs(["Modelagem", "Rastreabilidade"])

with modelagem_tab:
    # Modelo de exemplo
    template_path = current_dir / "00_Resilience_Module.xlsx"
    try:
        if template_path.exists():
            template_data = template_path.read_bytes()
        else:
            template_data = build_template_workbook()
        st.download_button(
            "📥 Modelo planilha",
            template_data,
            "00_Resilience_Module.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"Não foi possível preparar a planilha modelo: {e}")

    st.divider()

    general_models = [
        "Polinomial c/ Intercepto",
        "Polinomial s/Intercepto",
        "Potência Composta (Genérico)",
    ]
    tested_models = [name for name in MODELS_MAP.keys() if name not in general_models]

    st.markdown("**Escolha o modelo de regressão**")
    model_group = st.selectbox(
        "Grupo de modelos",
        ["Modelos testados", "Modelos gerais"],
    )
    available_models = tested_models if model_group == "Modelos testados" else general_models
    model_name = st.selectbox(
        model_group,
        available_models,
    )

    pezo_variant = None
    if model_name == "Pezo (1993)":
        pezo_variant = st.selectbox(
            "Pezo – Tipo",
            ["Normalizada", "Não normalizada"],
            index=0,
        )

    degree = 2
    if "Polinomial" in model_name:
        degree = st.selectbox("Grau (polinomial)", [2, 3, 4, 5, 6])

    energy = st.selectbox("Energia", ["Normal", "Intermediária", "Modificada"])

    if model_group == "Modelos testados":
        fit_method = st.selectbox(
            "Método de ajuste",
            [
                "Escala natural (minimiza erro em MR)",
                "Escala logarítmica (minimiza erro relativo)",
            ],
        )
        with st.expander("Sobre o método de ajuste", expanded=False):
            st.markdown(
                """
                - **Escala natural** ajusta os parâmetros minimizando diferenças absolutas em MR.
                - **Escala logarítmica** ajusta os parâmetros minimizando diferenças proporcionais; no Anexo 1 de Carvalho (2023), o modelo Hopkins é calibrado dessa forma.
                - A escala logarítmica exige valores positivos de MR e termos previstos positivos.
                - As métricas exibidas continuam sendo calculadas em MR natural, para manter a interpretação em MPa.
                """
            )
    else:
        fit_method = "Escala natural (minimiza erro em MR)"
        st.caption("Método de ajuste: escala natural para os modelos gerais.")

    pa_option = st.selectbox(
        "Pressão atmosférica de referência (Pa)",
        [
            "Pa = 1,0 (normalizado, Carvalho 2023)",
            "Pa = 0,101325 MPa (valor físico)",
        ],
    )
    pa_value = 1.0 if pa_option.startswith("Pa = 1,0") else 0.101325

    with st.expander("Sobre a escolha de Pa", expanded=False):
        st.markdown(
            """
            - **Pa = 1,0** segue o critério adotado por Carvalho (2023) para conversão/normalização das unidades.
            - **Pa = 0,101325 MPa** usa o valor físico aproximado da pressão atmosférica no mesmo sistema de unidades das tensões.
            - Em modelos com termos do tipo `+1` após a normalização por `Pa`, essa escolha pode alterar a geometria do ajuste.
            - Em modelos normalizados sem `+1`, como Pezo normalizado, a mudança tende a ser absorvida principalmente pelo coeficiente `k1`, mas os coeficientes ajustados não ficam diretamente comparáveis.
            """
        )

    st.divider()

    with st.expander("⚙️ Orientação do gráfico 3D (exportação Word)", expanded=False):
        st.session_state["azim_offset"] = st.slider("Ajuste horizontal (azim, °)", -180, 180, 0, 1)
        st.session_state["elev_offset"] = st.slider("Ajuste vertical (elev, °)", -90, 90, 0, 1)

with rastreabilidade_tab:
    if "trace_data_hora_calibracao" not in st.session_state:
        st.session_state["trace_data_hora_calibracao"] = datetime.now().isoformat(timespec="seconds")

    trace_analista = st.text_input("Analista", value="", key="trace_analista")
    trace_funcao_analista = st.text_input("Função do analista", value="", key="trace_funcao_analista")
    trace_identificacao_analista = st.text_input("Identificação do analista", value="", key="trace_identificacao_analista")
    trace_projeto = st.text_input("Projeto", value="", key="trace_projeto")
    trace_instituicao = st.text_input("Instituição", value="", key="trace_instituicao")
    trace_data_hora_calibracao = st.text_input("Data e hora de calibração", key="trace_data_hora_calibracao")

uploaded = st.file_uploader("Arquivo de entrada (CSV ou XLSX)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload de um arquivo com colunas σ3, σd e MR para continuar.")
    st.stop()

try:
    df = pd.read_csv(uploaded, decimal=",") if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    st.write("### Dados Carregados")
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ Erro ao ler o arquivo: {e}")
    st.stop()

def compute_calc_signature():
    """Assinatura das entradas que exigem novo cálculo. Energia e os sliders de
    orientação do gráfico ficam de fora de propósito (não afetam o ajuste)."""
    try:
        data_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
    except Exception:
        data_hash = hash(df.to_csv(index=False))
    return (
        model_name,
        pezo_variant,
        degree if "Polinomial" in model_name else None,
        fit_method,
        pa_value,
        uploaded.name,
        data_hash,
    )

calc_signature = compute_calc_signature()

if st.button("Calcular Ajuste"):
    try:
        X = df[["σ3", "σd"]].values
        y = df["MR"].values
    except KeyError:
        st.error("❌ O arquivo deve conter as colunas exatas: σ3, σd e MR.")
        st.stop()

    # Instanciar modelo
    model_class = MODELS_MAP[model_name]
    if model_name == "Pezo (1993)" and pezo_variant == "Não normalizada":
        from models import Pezo1993NonNormalizedModel

        model = Pezo1993NonNormalizedModel()
    elif "Polinomial" in model_name:
        model = model_class()
        model.degree = degree
        model.poly.degree = degree
    else:
        model = model_class()

    if hasattr(model, "Pa"):
        model.Pa = pa_value

    try:
        if model_group == "Modelos testados" and fit_method.startswith("Escala logarítmica"):
            fit_logarithmic_model(model, X, y)
        else:
            model.fit(X, y)
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"❌ Erro ao ajustar o modelo {model_name}: {e}")
        st.stop()

    # Métricas
    n_params = 0
    if hasattr(model, "_params") and model._params is not None:
        n_params = len(model._params)
    elif hasattr(model, "_coefs") and model._coefs is not None:
        n_params = len(model._coefs)

    metrics = calculate_metrics(y, y_pred, n_params)

    try:
        fig = plot_3d_surface(df, model)
    except Exception as e:
        fig = None
        st.warning(f"Não foi possível gerar o gráfico 3D: {e}")

    # Persiste os resultados para sobreviver aos reruns (ex.: baixar relatório)
    st.session_state["results"] = {
        "signature": calc_signature,
        "model": model,
        "metrics": metrics,
        "model_name": model_name,
        "df": df,
        "fig": fig,
    }
    st.session_state.pop("report_cache", None)

# --- Exibição a partir do estado persistido ---
stored = st.session_state.get("results")
if stored is None:
    st.info("Configure as opções no menu lateral e clique em **Calcular Ajuste**.")
    st.stop()
if stored["signature"] != calc_signature:
    st.warning(
        "As opções de cálculo mudaram desde o último ajuste. "
        "Clique em **Calcular Ajuste** para atualizar os resultados."
    )
    st.stop()

model = stored["model"]
metrics = stored["metrics"]
model_name = stored["model_name"]
df = stored["df"]
fig = stored["fig"]

if np.isnan(metrics["r2"]) or metrics["r2"] < 0:
    st.warning(f"⚠️ O ajuste resultou em um R² inválido ou negativo ({metrics['r2']:.4f}). Verifique seus dados.")

# Resultados
st.write(f"### Resultado: {model.name}")
st.latex(model.get_equation().strip("$$"))
equation_note = model.get_equation_note()
if equation_note:
    st.latex(equation_note)

st.write("### Indicadores Estatísticos")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        - **R²:** {metrics['r2']:.6f} → aproximadamente {metrics['r2'] * 100:.2f}% da variabilidade de MR é explicada pelo modelo.
        - **R² ajustado:** {metrics['r2_adj']:.6f} → considera a quantidade de parâmetros usados no ajuste.
        - **RMSE:** {metrics['rmse']:.4f} MPa → erro quadrático médio na unidade original do MR.
        - **MAE:** {metrics['mae']:.4f} MPa → erro absoluto médio, menos sensível a erros extremos.
        """
    )
with col2:
    st.markdown(
        f"""
        - **Média MR:** {metrics['mean_y']:.4f} MPa → valor médio observado no conjunto calibrado.
        - **Desvio padrão MR:** {metrics['std_y']:.4f} MPa → dispersão dos valores de MR em torno da média.
        - **Amplitude:** {metrics['amplitude']:.4f} MPa → diferença entre o maior e o menor MR observado.
        - **MR máximo / mínimo:** {metrics['max_y']:.4f} / {metrics['min_y']:.4f} MPa → limites observados nos dados.
        """
    )

st.write("### Qualidade do Ajuste")
labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
labels_cv = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]

c1, c2, c3 = st.columns(3)
with c1:
    st.write(f"**NRMSE:** {metrics['nrmse_range']:.2%} → {get_quality_label(metrics['nrmse_range'], [0.05, 0.10], labels_nrmse)}")
with c2:
    st.write(f"**CV(RMSE):** {metrics['cv_rmse']:.2%} → {get_quality_label(metrics['cv_rmse'], [0.10, 0.20], labels_cv)}")
with c3:
    st.write(f"**MAE %:** {metrics['mae_pct']:.2%} → {get_quality_label(metrics['mae_pct'], [0.10, 0.20], labels_cv)}")

st.write("### Gráfico 3D da Superfície")
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Gráfico 3D indisponível para este ajuste.")

# Downloads (gerados sob demanda e cacheados por assinatura + energia/rastreabilidade/orientação)
st.write("### Exportar Resultados")
try:
    from utils.reports import generate_word_doc, generate_latex_zip, generate_pdf_doc

    traceability = {
        "Analista": trace_analista,
        "Função do analista": trace_funcao_analista,
        "Identificação do analista": trace_identificacao_analista,
        "Projeto": trace_projeto,
        "Instituição": trace_instituicao,
        "Data e hora de calibração": trace_data_hora_calibracao,
    }
    modeling_metadata = {
        "Arquivo de entrada": uploaded.name,
        "Grupo de modelos": model_group,
        "Modelo selecionado": model_name,
        "Modelo ajustado": model.name,
        "Energia": energy,
        "Método de ajuste": fit_method,
        "Pressão atmosférica de referência (Pa)": pa_option,
        "Número de registros": len(df),
    }
    if pezo_variant is not None:
        modeling_metadata["Pezo – Tipo"] = pezo_variant
    if "Polinomial" in model_name:
        modeling_metadata["Grau polinomial"] = degree

    azim_offset = st.session_state.get("azim_offset", 0)
    elev_offset = st.session_state.get("elev_offset", 0)
    report_key = (calc_signature, energy, tuple(sorted(traceability.items())), azim_offset, elev_offset)

    cache = st.session_state.get("report_cache")
    if not cache or cache.get("key") != report_key:
        zip_buf, _ = generate_latex_zip(model, metrics, df, fig, energy, traceability, modeling_metadata)
        cache = {
            "key": report_key,
            "word": generate_word_doc(model, metrics, df, fig, energy, traceability, modeling_metadata).getvalue(),
            "zip": zip_buf.getvalue(),
            "pdf": generate_pdf_doc(model, metrics, df, fig, energy, traceability, modeling_metadata).getvalue(),
        }
        st.session_state["report_cache"] = cache

    fname = model_name.replace(" ", "_")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.download_button("📄 Baixar Relatório Word", cache["word"], f"Relatorio_{fname}.docx")
    with dcol2:
        st.download_button("📦 Baixar LaTeX (ZIP)", cache["zip"], f"Relatorio_{fname}.zip")
    with dcol3:
        st.download_button("🧾 Baixar Relatório PDF", cache["pdf"], f"Relatorio_{fname}.pdf", mime="application/pdf")
except Exception as e:
    st.error(f"Erro ao gerar arquivos para download: {e}")

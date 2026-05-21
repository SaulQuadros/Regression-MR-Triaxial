import streamlit as st
import pandas as pd
import numpy as np
import io

from models import MODELS_MAP
from utils.metrics import calculate_metrics, get_quality_label
from utils.plotting import plot_3d_surface
from utils.reports import generate_word_doc, generate_latex_zip

st.set_page_config(page_title="Modelos de MR - Camila Carvalho (2023)", layout="wide")
st.title("Modelos de Regressão para MR")
st.markdown("""
Esta ferramenta permite o ajuste de modelos matemáticos para o Módulo de Resiliência (MR),
incluindo os 14 modelos selecionados na dissertação de **Camila Luiza Mello Carvalho (UFJF, 2023)**.
""")

# --- Sidebar ---
st.sidebar.header("Configurações")

# Modelo de exemplo
try:
    with open("00_Resilience_Module.xlsx", "rb") as f:
        st.sidebar.download_button("📥 Modelo planilha", f, "00_Resilience_Module.xlsx")
except FileNotFoundError:
    st.sidebar.warning("Modelo não encontrado.")

uploaded = st.file_uploader("Arquivo de entrada (CSV ou XLSX)", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload de um arquivo com colunas σ3, σd e MR para continuar.")
    st.stop()

df = pd.read_csv(uploaded, decimal=",") if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write("### Dados Carregados")
st.dataframe(df)

model_name = st.sidebar.selectbox(
    "Escolha o modelo de regressão",
    list(MODELS_MAP.keys())
)

degree = 2
if "Polinomial" in model_name:
    degree = st.sidebar.selectbox("Grau (polinomial)", [2, 3, 4, 5, 6])

energy = st.sidebar.selectbox("Energia", ["Normal", "Intermediária", "Modificada"])

with st.sidebar.expander("⚙️ Orientação do gráfico 3D (exportação Word)", expanded=False):
    st.session_state["azim_offset"] = st.slider("Ajuste horizontal (azim, °)", -180, 180, 0, 1)
    st.session_state["elev_offset"] = st.slider("Ajuste vertical (elev, °)", -90, 90, 0, 1)

if st.button("Calcular Ajuste"):
    X = df[["σ3", "σd"]].values
    y = df["MR"].values

    # Instanciar modelo
    model_class = MODELS_MAP[model_name]
    if "Polinomial" in model_name:
        model = model_class()
        model.degree = degree
        model.poly.degree = degree
    else:
        model = model_class()

    try:
        model.fit(X, y)
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"❌ Erro ao ajustar o modelo {model_name}: {e}")
        st.stop()

    # Métricas
    # Tenta obter o número de parâmetros dos atributos conhecidos
    n_params = 0
    if hasattr(model, "_params") and model._params is not None:
        n_params = len(model._params)
    elif hasattr(model, "_coefs") and model._coefs is not None:
        n_params = len(model._coefs)

    metrics = calculate_metrics(y, y_pred, n_params)

    if np.isnan(metrics["r2"]) or metrics["r2"] < 0:
        st.warning(f"⚠️ O ajuste resultou em um R² inválido ou negativo ({metrics['r2']:.4f}). Verifique seus dados.")

    # Resultados
    st.write(f"### Resultado: {model.name}")
    st.latex(model.get_equation().strip("$$"))

    st.write("### Indicadores Estatísticos")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{metrics['r2']:.6f}")
        st.metric("R² Ajustado", f"{metrics['r2_adj']:.6f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f} MPa")
        st.metric("MAE", f"{metrics['mae']:.4f} MPa")
    with col3:
        st.metric("Média MR", f"{metrics['mean_y']:.4f} MPa")
        st.metric("Amplitude", f"{metrics['amplitude']:.4f} MPa")

    st.write("### Qualidade do Ajuste")
    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**NRMSE:** {metrics['nrmse_range']:.2%} → {get_quality_label(metrics['nrmse_range'], [0.05, 0.10], labels_nrmse)}")
    with c2:
        st.write(f"**CV(RMSE):** {metrics['cv_rmse']:.2%} → {get_quality_label(metrics['cv_rmse'], [0.10, 0.20], labels_cv)}")

    st.write("### Gráfico 3D da Superfície")
    fig = plot_3d_surface(df, model)
    st.plotly_chart(fig, use_container_width=True)

    # Downloads
    st.write("### Exportar Resultados")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        doc_buf = generate_word_doc(model, metrics, df, fig, energy)
        st.download_button("📄 Baixar Relatório Word", doc_buf, f"Relatorio_{model_name.replace(' ', '_')}.docx")
    with dcol2:
        zip_buf, tex_content = generate_latex_zip(model, metrics, df, fig, energy)
        st.download_button("📦 Baixar LaTeX (ZIP)", zip_buf, f"Relatorio_{model_name.replace(' ', '_')}.zip")

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

    st.divider()

    with st.expander("⚙️ Orientação do gráfico 3D (exportação Word)", expanded=False):
        st.session_state["azim_offset"] = st.slider("Ajuste horizontal (azim, °)", -180, 180, 0, 1)
        st.session_state["elev_offset"] = st.slider("Ajuste vertical (elev, °)", -90, 90, 0, 1)

with rastreabilidade_tab:
    trace_analista = st.text_input("Analista", value="Saul Germano Rabello Quadros", key="trace_analista")
    trace_funcao_analista = st.text_input("Função do analista", value="Pesquisador", key="trace_funcao_analista")
    trace_identificacao_analista = st.text_input("Identificação do analista", value="Discente", key="trace_identificacao_analista")
    trace_projeto = st.text_input("Projeto", value="Tese", key="trace_projeto")
    trace_instituicao = st.text_input("Instituição", value="PEC | UFJF", key="trace_instituicao")
    trace_data_hora_calibracao = st.text_input("Data e hora de calibração", value="2026-05-13T08:49:51", key="trace_data_hora_calibracao")

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

    try:
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
    
    if np.isnan(metrics["r2"]) or metrics["r2"] < 0:
        st.warning(f"⚠️ O ajuste resultou em um R² inválido ou negativo ({metrics['r2']:.4f}). Verifique seus dados.")

    # Resultados
    st.write(f"### Resultado: {model.name}")
    st.latex(model.get_equation().strip("$$"))

    st.write("### Indicadores Estatísticos")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{metrics['r2']:.6f}")
        st.metric("R² Ajustado", f"{metrics['r2_adj']:.6f}")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f} MPa")
        st.metric("MAE", f"{metrics['mae']:.4f} MPa")
    with col3:
        st.metric("Média MR", f"{metrics['mean_y']:.4f} MPa")
        st.metric("Desvio Padrão MR", f"{metrics['std_y']:.4f} MPa")
    with col4:
        st.metric("Amplitude", f"{metrics['amplitude']:.4f} MPa")
        st.metric("MR Máx / Mín", f"{metrics['max_y']:.4f} / {metrics['min_y']:.4f} MPa")

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
    try:
        fig = plot_3d_surface(df, model)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Não foi possível gerar o gráfico 3D: {e}")

    # Downloads
    st.write("### Exportar Resultados")
    dcol1, dcol2 = st.columns(2)
    try:
        from utils.reports import generate_word_doc, generate_latex_zip

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
            "Número de registros": len(df),
        }
        if pezo_variant is not None:
            modeling_metadata["Pezo – Tipo"] = pezo_variant
        if "Polinomial" in model_name:
            modeling_metadata["Grau polinomial"] = degree

        with dcol1:
            doc_buf = generate_word_doc(model, metrics, df, fig, energy, traceability, modeling_metadata)
            st.download_button("📄 Baixar Relatório Word", doc_buf, f"Relatorio_{model_name.replace(' ', '_')}.docx")
        with dcol2:
            zip_buf, tex_content = generate_latex_zip(model, metrics, df, fig, energy, traceability, modeling_metadata)
            st.download_button("📦 Baixar LaTeX (ZIP)", zip_buf, f"Relatorio_{model_name.replace(' ', '_')}.zip")
    except Exception as e:
        st.error(f"Erro ao gerar arquivos para download: {e}")

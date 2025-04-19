#!/usr/bin/env python
# coding: utf-8

import os
import sys

# 1) Determina o diretório do script e ajusta o sys.path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.chdir(app_dir)

import streamlit as st

# ———————————————— PRIMEIRA E ÚNICA CHAMADA ————————————————
st.set_page_config(page_title="Modelos de MR", layout="wide")

import pandas as pd
import io
import zipfile

from app_calc import calcular_modelo, interpret_metrics, plot_3d_surface
from app_latex import generate_latex_doc, generate_word_doc

# Estado inicial
if "calculated" not in st.session_state:
    st.session_state.calculated = False

st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

# Upload de dados
uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

df = (
    pd.read_csv(uploaded, decimal=",")
    if uploaded.name.endswith(".csv")
    else pd.read_excel(uploaded)
)
st.write("### Dados Carregados")
st.dataframe(df)

# Configurações na sidebar
st.sidebar.header("Configurações")
model_type = st.sidebar.selectbox(
    "Escolha o modelo de regressão",
    [
        "Polinomial c/ Intercepto",
        "Polinomial s/Intercepto",
        "Potência Composta",
        "Pezo"
    ]
)

degree = None
if model_type.startswith("Polinomial"):
    degree = st.sidebar.selectbox("Grau (polinomial)", [2, 3, 4, 5, 6], index=0)

energy = st.sidebar.selectbox(
    "Energia",
    ["Normal", "Intermediária", "Modificada"],
    index=0
)

# Botão de cálculo
if st.button("Calcular"):
    # 1) Cálculo do modelo
    result = calcular_modelo(df, model_type, degree)

    # 2) Interpretação de métricas e gráfico
    eq_latex   = result["eq_latex"]
    metrics_txt = interpret_metrics(
        result["r2"], result["r2_adj"], result["rmse"], result["mae"], df["MR"].values
    )
    fig = plot_3d_surface(
        df,
        result["model_obj"],
        result["poly_obj"],
        "MR",
        is_power=result["is_power"],
        power_params=result["power_params"]
    )

    # 3) Gera ZIP LaTeX
    tex_content, img_data = generate_latex_doc(
        eq_latex,
        result["r2"],
        result["r2_adj"],
        result["rmse"],
        result["mae"],
        result["mean_MR"],
        result["std_MR"],
        energy,
        degree,
        result["intercept"],
        df,
        fig
    )
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        zf.writestr("main.tex", tex_content)
        zf.writestr("surface_plot.png", img_data)
    zip_buf.seek(0)

    # 4) Gera DOCX (OMML ou fallback)
    try:
        import pypandoc
        pypandoc.download_pandoc("latest")
        docx_bytes = pypandoc.convert_text(tex_content, "docx", format="latex")
    except Exception:
        buf = generate_word_doc(
            eq_latex,
            metrics_txt,
            fig,
            energy,
            degree,
            result["intercept"],
            df
        )
        buf.seek(0)
        docx_bytes = buf.read()

    # 5) Armazena em session_state
    st.session_state.calculated = True
    st.session_state.result       = result
    st.session_state.metrics_txt  = metrics_txt
    st.session_state.fig          = fig
    st.session_state.zip_buf      = zip_buf
    st.session_state.docx_bytes   = docx_bytes

# Exibição de resultados e downloads após cálculo
if st.session_state.calculated:
    res = st.session_state.result
    st.write("### Equação Ajustada")
    st.latex(res["eq_latex"].strip("$$"))

    st.write("### Indicadores Estatísticos")
    indicators = [
        ("R²",           f"{res['r2']:.6f}",      f"Explica {res['r2']*100:.2f}% dos dados."),
        ("R² Ajustado",  f"{res['r2_adj']:.6f}",  "Penaliza uso excessivo de termos."),
        ("RMSE",         f"{res['rmse']:.4f} MPa", "Erro quadrático médio"),
        ("MAE",          f"{res['mae']:.4f} MPa",  "Erro absoluto médio"),
        ("Média MR",     f"{res['mean_MR']:.4f} MPa", "Média observada"),
        ("Desvio Padrão MR", f"{res['std_MR']:.4f} MPa", "Dispersão dos dados")
    ]
    for name, val, tip in indicators:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    st.write(f"**Intercepto:** {res['intercept']:.4f}")
    st.markdown(
        "Função válida para 0,020≤σ₃≤0,14 e 0,02≤σ_d≤0,42 (DNIT 134/2018‑ME).",
        unsafe_allow_html=True
    )

    # Qualidade do ajuste
    amp       = df["MR"].max() - df["MR"].min()
    nrmse     = res["rmse"] / amp if amp > 0 else float("nan")
    cv_rmse   = res["rmse"] / res["mean_MR"] if res["mean_MR"] else float("nan")
    mae_pct   = res["mae"]  / res["mean_MR"] if res["mean_MR"] else float("nan")

    def quality_label(val, th, labels):
        for t, lab in zip(th, labels):
            if val <= t:
                return lab
        return labels[-1]

    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]
    labels_mae   = labels_cv

    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    st.markdown(
        f"- **NRMSE_range:** {nrmse:.2%} → {quality_label(nrmse, [0.05,0.10], labels_nrmse)}",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **CV(RMSE):** {cv_rmse:.2%} → {quality_label(cv_rmse, [0.10,0.20], labels_cv)}",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **MAE %:** {mae_pct:.2%} → {quality_label(mae_pct, [0.10,0.20], labels_mae)}",
        unsafe_allow_html=True
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(st.session_state.fig, use_container_width=True)

    # Botões de download
    st.download_button(
        "Salvar LaTeX",
        data=st.session_state.zip_buf,
        file_name="Relatorio_Regressao.zip",
        mime="application/zip"
    )
    st.download_button(
        "Converter para Word",
        data=st.session_state.docx_bytes,
        file_name="Relatorio_Regressao.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import streamlit as st
import pandas as pd
import io, zipfile

# Ajusta o path para importar módulos locais
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.chdir(app_dir)

# Chamada única de configuração de página
st.set_page_config(page_title="Modelos de MR", layout="wide")

from app_calc import calcular_modelo, interpret_metrics, plot_3d_surface
from app_latex import generate_latex_doc, generate_word_doc

# --- Sidebar: botão de download do template Excel ---
template_file = "00_Resilience_Module.xlsx"
if os.path.exists(template_file):
    with open(template_file, "rb") as f:
        template_data = f.read()
    st.sidebar.download_button(
        label="📥 Modelo planilha",
        data=template_data,
        file_name=template_file,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Estado inicial para controlar limpeza de resultados
if "calculated" not in st.session_state:
    st.session_state.calculated = False

def reset_results():
    st.session_state.calculated = False

# --- Cabeçalho ---
st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

# --- Upload de dados ---
uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

df = (pd.read_csv(uploaded, decimal=",")
      if uploaded.name.lower().endswith(".csv")
      else pd.read_excel(uploaded))
st.write("### Dados Carregados")
st.dataframe(df)

# --- Configurações na sidebar ---
st.sidebar.header("Configurações")

model_type = st.sidebar.selectbox(
    "Escolha o modelo de regressão",
    [
        "Polinomial c/ Intercepto",
        "Polinomial s/Intercepto",
        "Potência Composta",
        "Pezo"
    ],
    key="model_type",
    on_change=reset_results
)

degree = None
if model_type.startswith("Polinomial"):
    degree = st.sidebar.selectbox(
        "Grau (polinomial)",
        [2, 3, 4, 5, 6],
        index=0,
        key="degree",
        on_change=reset_results
    )

energy = st.sidebar.selectbox(
    "Energia",
    ["Normal", "Intermediária", "Modificada"],
    index=0,
    key="energy",
    on_change=reset_results
)

# --- Botão de cálculo ---
if st.button("Calcular"):
    # 1) Ajusta o modelo
    res = calcular_modelo(df, model_type, degree)

    # 2) Métricas e gráfico
    eq_latex   = res["eq_latex"]
    metrics_md = interpret_metrics(
        res["r2"], res["r2_adj"], res["rmse"], res["mae"], df["MR"].values
    )
    fig = plot_3d_surface(
        df, res["model_obj"], res["poly_obj"],
        "MR", is_power=res["is_power"], power_params=res["power_params"]
    )

    # 3) Gera ZIP com LaTeX
    tex, img = generate_latex_doc(
        eq_latex, res["r2"], res["r2_adj"], res["rmse"],
        res["mae"], res["mean_MR"], res["std_MR"],
        energy, degree, res["intercept"], df, fig
    )
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("main.tex", tex)
        zf.writestr("surface_plot.png", img)
    zip_buf.seek(0)

    # 4) Converte para DOCX (OMML ou fallback)
    try:
        import pypandoc
        pypandoc.download_pandoc("latest")
        docx_bytes = pypandoc.convert_text(tex, "docx", format="latex")
    except Exception:
        buf = generate_word_doc(eq_latex, metrics_md, fig,
                                energy, degree, res["intercept"], df)
        buf.seek(0)
        docx_bytes = buf.read()

    # 5) Salva no session_state
    st.session_state.update({
        "calculated":  True,
        "result":      res,
        "metrics_md":  metrics_md,
        "fig":         fig,
        "zip_buf":     zip_buf,
        "docx_bytes":  docx_bytes
    })

# --- Exibição de resultados ---
if st.session_state.calculated:
    r = st.session_state.result

    st.write("### Equação Ajustada")
    st.latex(r["eq_latex"].strip("$$"))

    st.write("### Indicadores Estatísticos")
    stats = [
        ("R²",               f"{r['r2']:.6f}",
         f"Este valor indica que aproximadamente {r['r2']*100:.2f}% da variabilidade dos dados de MR é explicada pelo modelo."),
        ("R² Ajustado",      f"{r['r2_adj']:.6f}",
         "Essa métrica penaliza o uso excessivo de termos."),
        ("RMSE",             f"{r['rmse']:.4f} MPa",
         f"Erro quadrático médio: {r['rmse']:.4f} MPa."),
        ("MAE",              f"{r['mae']:.4f} MPa",
         f"Erro absoluto médio: {r['mae']:.4f} MPa."),
        ("Média MR",         f"{r['mean_MR']:.4f} MPa",
         "Média dos valores observados."),
        ("Desvio Padrão MR", f"{r['std_MR']:.4f} MPa",
         "Dispersão dos dados em torno da média.")
    ]
    for name, val, tip in stats:
        st.markdown(f"**{name}:** {val} <span title='{tip}'>ℹ️</span>",
                    unsafe_allow_html=True)

    st.write(f"**Intercepto:** {r['intercept']:.4f}")
    st.markdown(
        "Função válida para 0,020≤σ₃≤0,14 e 0,02≤σ_d≤0,42 (DNIT 134/2018‑ME).",
        unsafe_allow_html=True
    )

    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    nrmse, qual_nrmse, tip_n = r["quality"]["NRMSE"]
    cv_, qual_cv, tip_cv      = r["quality"]["CV(RMSE)"]
    mae_, qual_mae, tip_mae   = r["quality"]["MAE %"]

    st.markdown(
        f"- **NRMSE:** {nrmse:.2%} → {qual_nrmse} <span title='{tip_n}'>ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **CV(RMSE):** {cv_:.2%} → {qual_cv} <span title='{tip_cv}'>ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **MAE %:** {mae_:.2%} → {qual_mae} <span title='{tip_mae}'>ℹ️</span>",
        unsafe_allow_html=True
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(st.session_state["fig"], use_container_width=True)

    st.download_button(
        "Salvar LaTeX",
        data=st.session_state["zip_buf"],
        file_name="Relatorio_Regressao.zip",
        mime="application/zip"
    )
    st.download_button(
        "Converter para Word",
        data=st.session_state["docx_bytes"],
        file_name="Relatorio_Regressao.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

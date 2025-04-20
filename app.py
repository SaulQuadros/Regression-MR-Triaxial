#!/usr/bin/env python
# coding: utf-8

import os
import sys
import base64

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.chdir(app_dir)

import streamlit as st
import pandas as pd
import io
import zipfile

from app_calc import calcular_modelo, interpret_metrics, plot_3d_surface
from app_latex import generate_latex_doc, generate_word_doc

st.set_page_config(page_title="Modelos de MR", layout="wide")

template_path = os.path.join(app_dir, "00_Resilience_Module.xlsx")
if os.path.exists(template_path):
    data = open(template_path, "rb").read()
    b64  = base64.b64encode(data).decode()
    href = (
        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
        f'download="00_Resilience_Module.xlsx" '
        f'style="display:flex; align-items:center; gap:0.5rem; '
        f'background-color:#007BFF; color:#FFFFFF; padding:0.5rem 1rem; '
        f'border-radius:5px; font-weight:bold; text-decoration:none;">'
        f'📥 Modelo planilha</a>'
    )
    st.sidebar.markdown(href, unsafe_allow_html=True)

if "calculated" not in st.session_state:
    st.session_state.calculated = False

def reset_results():
    st.session_state.calculated = False

st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

df = pd.read_csv(uploaded, decimal=",") if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write("### Dados Carregados")
st.dataframe(df)

st.sidebar.header("Configurações")
model_type = st.sidebar.selectbox("Modelo de regressão",
    ["Polinomial c/ Intercepto","Polinomial s/Intercepto","Potência Composta","Pezo"],
    key="model_type", on_change=reset_results)
degree = st.sidebar.selectbox("Grau (polinomial)",[2,3,4,5,6],
    index=0, key="degree", on_change=reset_results) if model_type.startswith("Polinomial") else None
energy = st.sidebar.selectbox("Energia",["Normal","Intermediária","Modificada"],
    index=0, key="energy", on_change=reset_results)

if st.button("Calcular"):
    result = calcular_modelo(df, model_type, degree)
    eq_latex = result["eq_latex"]
    fig      = plot_3d_surface(
        df, result["model_obj"], result["poly_obj"],
        "MR", is_power=result["is_power"], power_params=result["power_params"]
    )

    tex_content, img_data = generate_latex_doc(
        eq_latex, result["r2"], result["r2_adj"],
        result["rmse"], result["mae"],
        result["mean_MR"], result["std_MR"],
        energy, degree, result["intercept"], df, fig
    )
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("main.tex", tex_content)
        zf.writestr("surface_plot.png", img_data)
    zip_buf.seek(0)

    try:
        import pypandoc
        pypandoc.download_pandoc("latest")
        docx_bytes = pypandoc.convert_text(tex_content, "docx", format="latex")
    except:
        buf = generate_word_doc(eq_latex, "", fig, energy, degree, result["intercept"], df)
        buf.seek(0)
        docx_bytes = buf.read()

    st.session_state.calculated   = True
    st.session_state.result       = result
    st.session_state.fig          = fig
    st.session_state.zip_buf      = zip_buf
    st.session_state.docx_bytes   = docx_bytes

if st.session_state.calculated:
    res = st.session_state.result
    st.write("### Equação Ajustada")
    st.latex(res["eq_latex"].strip("$$"))
    st.write("### Indicadores Estatísticos")
    for name,val,tip in [
        ("R²", f"{res['r2']:.6f}", f"Este valor indica que aproximadamente {res['r2']*100:.2f}% da variabilidade dos dados de MR é explicada pelo modelo."),
        ("R² Ajustado", f"{res['r2_adj']:.6f}", "Essa métrica penaliza o uso excessivo de termos."),
        ("RMSE", f"{res['rmse']:.4f} MPa", f"Erro quadrático médio: {res['rmse']:.4f} MPa."),
        ("MAE", f"{res['mae']:.4f} MPa", f"Erro absoluto médio: {res['mae']:.4f} MPa."),
        ("Média MR", f"{res['mean_MR']:.4f} MPa","Média dos valores observados."),
        ("Desvio Padrão MR", f"{res['std_MR']:.4f} MPa","Dispersão dos dados em torno da média.")
    ]:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    st.write(f"**Intercepto:** {res['intercept']:.4f}")
    st.markdown("Função válida para 0,020≤σ₃≤0,14 e 0,02≤σ_d≤0,42 (DNIT 134/2018‑ME).", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    for key in ["NRMSE","CV(RMSE)","MAE %"]:
        val, qual, tip = res["quality"][key]
        st.markdown(f"- **{key}:** {val:.2%} → {qual} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(st.session_state.fig, use_container_width=True)

    st.download_button("Salvar LaTeX", data=st.session_state.zip_buf, file_name="Relatorio_Regressao.zip", mime="application/zip")
    st.download_button("Converter para Word", data=st.session_state.docx_bytes, file_name="Relatorio_Regressao.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import base64
import streamlit as st
import pandas as pd
import io
import zipfile

# Ajusta caminho para import local
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
os.chdir(app_dir)

from app_calc import calcular_modelo, interpret_metrics, plot_3d_surface, CLASSICOS
from app_latex import generate_latex_doc, generate_word_doc

st.set_page_config(page_title="Modelos de MR", layout="wide")

# Download do template
template_path = os.path.join(app_dir, "00_Resilience_Module.xlsx")
if os.path.exists(template_path):
    with open(template_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = (
        '<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + b64 + '" '
        'download="00_Resilience_Module.xlsx" '
        'style="display:inline-flex;align-items:center;gap:0.5rem;padding:0.5rem 1rem;'
        'background-color:#f0f0f0;border:1px solid #ccc;border-radius:0.375rem;'
        'text-decoration:none;color:#333;font-weight:600;">'
        '📥 Modelo planilha'
        '</a>'
    )
    st.sidebar.markdown(href, unsafe_allow_html=True)

# Estado inicial
if "calculated" not in st.session_state:
    st.session_state.calculated = False
if "var_pair" not in st.session_state:
    st.session_state.var_pair = ("σ3", "σd")
if "model_category" not in st.session_state:
    st.session_state.model_category = "Genéricos"

def reset_results():
    st.session_state.calculated = False

def reset_all():
    reset_results()

# Sidebar: seleção de variáveis independentes
st.sidebar.header("Seleção de Variáveis")

label_map = {
    "σ3": "σ₃",
    "σd": "σ_d",
    "θ": "θ",
    "τ_oct": "τ_oct"
}
var_pairs = [
    ("σ3","σd"),
    ("θ","σd"),
    ("θ","τ_oct"),
    ("σ3","τ_oct"),
    ("σd","τ_oct")
]
pairs_str = [f"{label_map[a]} & {label_map[b]}" for a,b in var_pairs]

sel = st.sidebar.radio(
    "Escolha o par de variáveis independentes",
    pairs_str,
    index=0,
    on_change=reset_all
)
st.session_state.var_pair = var_pairs[pairs_str.index(sel)]

# Sidebar: categoria de modelo
st.sidebar.header("Tipo de Modelo")
st.session_state.model_category = st.sidebar.radio(
    "Categoria", ["Genéricos","Clássicos"],
    index=0, on_change=reset_all
)

# Sidebar: escolha de modelo
st.sidebar.header("Modelos Disponíveis")
if st.session_state.model_category == "Genéricos":
    model_options = [
        "Polinomial c/ Intercepto",
        "Polinomial s/Intercepto",
        "Potência Composta",
        "Pezo"
    ]
else:
    model_options = list(CLASSICOS.keys())

model_type = st.sidebar.selectbox(
    "Escolha o modelo de regressão",
    model_options,
    on_change=reset_results
)

# Configurações adicionais
degree = None
if st.session_state.model_category == "Genéricos" and model_type.startswith("Polinomial"):
    degree = st.sidebar.selectbox(
        "Grau (polinomial)", [2,3,4,5,6],
        index=0, on_change=reset_results
    )
energy = st.sidebar.selectbox(
    "Energia", ["Normal","Intermediária","Modificada"],
    index=0, on_change=reset_results
)

# Título e instruções dinâmicas
st.title("Modelos de Regressão para MR")
var1, var2 = st.session_state.var_pair
st.markdown(
    f"Envie um CSV ou XLSX com colunas **{label_map[var1]}**, **{label_map[var2]}** e **MR**."
)

# Upload de dados
uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

# Carrega dados
df = pd.read_csv(uploaded, decimal=",") if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write("### Dados Carregados")
st.dataframe(df)

# Cálculo
if st.button("Calcular"):
    result = calcular_modelo(df, model_type, degree)
    eq_latex = result["eq_latex"]
    metrics_txt = interpret_metrics(
        result["r2"], result["r2_adj"], result["rmse"], result["mae"], df["MR"].values
    )
    fig = plot_3d_surface(
        df, result["model_obj"], result["poly_obj"],
        "MR", is_power=result["is_power"], power_params=result["power_params"]
    )
    tex_content, img_data = generate_latex_doc(
        eq_latex, result["r2"], result["r2_adj"], result["rmse"], result["mae"],
        result["mean_MR"], result["std_MR"], energy, degree,
        result["intercept"], df, fig
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
    except Exception:
        buf = generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, result["intercept"], df)
        buf.seek(0)
        docx_bytes = buf.read()
    st.session_state.calculated = True
    st.session_state.result = result
    st.session_state.fig = fig
    st.session_state.zip_buf = zip_buf
    st.session_state.docx_bytes = docx_bytes

# Exibe resultados
if st.session_state.calculated:
    res = st.session_state.result
    st.write("### Equação Ajustada")
    st.latex(res["eq_latex"].strip("$$"))
    st.write("### Indicadores Estatísticos")
    indicators = [
        ("R²", f"{res['r2']:.6f}", f"{res['r2']*100:.2f}% explicado"),
        ("R² Ajustado", f"{res['r2_adj']:.6f}", "Penaliza termos excessivos"),
        ("RMSE", f"{res['rmse']:.4f} MPa", f"{res['rmse']:.4f}"),
        ("MAE", f"{res['mae']:.4f} MPa", f"{res['mae']:.4f}"),
        ("Média MR", f"{res['mean_MR']:.4f} MPa", "Média observada"),
        ("Desvio Padrão MR", f"{res['std_MR']:.4f} MPa", "Dispersão dos dados")
    ]
    for name, val, tip in indicators:
        st.markdown(f'**{name}:** {val} <span title="{tip}">ℹ️</span>', unsafe_allow_html=True)
    st.write(f"**Intercepto:** {res['intercept']:.4f}")
    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    for key, (val, lab, tip) in res["quality"].items():
        st.markdown(f'- **{key}:** {val:.2%} → {lab} <span title="{tip}">ℹ️</span>', unsafe_allow_html=True)
    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(st.session_state.fig, use_container_width=True)
    st.download_button("Salvar LaTeX", data=st.session_state.zip_buf, file_name="Relatorio_Regressao.zip", mime="application/zip")
    st.download_button("Converter para Word", data=st.session_state.docx_bytes, file_name="Relatorio_Regressao.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

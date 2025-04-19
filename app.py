#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# --- app.py ---
import os
import sys

# 1) Caminho absoluto até este script e diretório
app_dir = os.path.dirname(os.path.abspath(__file__))

# 2) Adiciona ao path para importar módulos locais antes de tudo
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# 3) Garante que o cwd seja o diretório do script
os.chdir(app_dir)

import streamlit as st
import pandas as pd
import io
import zipfile

# Debug (remova depois)
st.write("Pasta atual:", os.getcwd())
st.write("Arquivos nesta pasta:", os.listdir(os.getcwd()))

# Importa somente uma vez cada módulo
from app_calc import calcular_modelo, interpret_metrics, plot_3d_surface
from app_latex import generate_latex_doc, generate_word_doc

# --- Script app.py  ---
st.set_page_config(page_title="Modelos de MR", layout="wide")
st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

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

# --- Configurações ---
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

if st.button("Calcular"):
    # Chama o módulo de cálculo
    result = calcular_modelo(df, model_type, degree)
    eq_latex   = result["eq_latex"]
    intercept  = result["intercept"]
    r2         = result["r2"]
    r2_adj     = result["r2_adj"]
    rmse       = result["rmse"]
    mae        = result["mae"]
    mean_MR    = result["mean_MR"]
    std_MR     = result["std_MR"]
    model_obj  = result["model_obj"]
    poly_obj   = result["poly_obj"]
    is_power   = result["is_power"]
    power_params = result["power_params"]

    # Métricas textuais
    metrics_txt = interpret_metrics(r2, r2_adj, rmse, mae, df["MR"].values)

    # Gráfico 3D
    fig = plot_3d_surface(df, model_obj, poly_obj, "MR", is_power=is_power, power_params=power_params)

    # Exibição dos resultados
    st.write("### Equação Ajustada")
    st.latex(eq_latex.strip("$$"))

    st.write("### Indicadores Estatísticos")
    indicators = [
        ("R²", f"{r2:.6f}", f"Explica {r2*100:.2f}% da variabilidade dos dados."),
        ("R² Ajustado", f"{r2_adj:.6f}", "Penaliza uso excessivo de termos."),
        ("RMSE", f"{rmse:.4f} MPa", "Erro quadrático médio"),
        ("MAE", f"{mae:.4f} MPa", "Erro absoluto médio"),
        ("Média MR", f"{mean_MR:.4f} MPa", "Média dos valores observados"),
        ("Desvio Padrão MR", f"{std_MR:.4f} MPa", "Dispersão dos dados")
    ]
    for name, val, tip in indicators:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    st.write(f"**Intercepto:** {intercept:.4f}")
    st.markdown(
        "A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e 0,02≤σ_d≤0,42 observada a norma DNIT 134/2018‑ME.",
        unsafe_allow_html=True
    )

    # Avaliação da Qualidade do Ajuste
    amp = df["MR"].max() - df["MR"].min()
    nrmse_range = rmse / amp if amp > 0 else float("nan")
    cv_rmse     = rmse / mean_MR if mean_MR != 0 else float("nan")
    mae_pct     = mae  / mean_MR if mean_MR  != 0 else float("nan")

    def quality_label(val, thresholds, labels):
        for t, lab in zip(thresholds, labels):
            if val <= t:
                return lab
        return labels[-1]

    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]
    labels_mae   = labels_cv

    qual_nrmse = quality_label(nrmse_range, [0.05, 0.10], labels_nrmse)
    qual_cv     = quality_label(cv_rmse,     [0.10, 0.20], labels_cv)
    qual_mae    = quality_label(mae_pct,     [0.10, 0.20], labels_mae)

    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    st.markdown(
        f"- **NRMSE_range:** {nrmse_range:.2%} → {qual_nrmse} <span title=\"NRMSE_range\">ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **CV(RMSE):** {cv_rmse:.2%} → {qual_cv} <span title=\"CV(RMSE)\">ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **MAE %:** {mae_pct:.2%} → {qual_mae} <span title=\"MAE %\">ℹ️</span>",
        unsafe_allow_html=True
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(fig, use_container_width=True)

    # Download LaTeX (.zip)
    tex_content, img_data = generate_latex_doc(
        eq_latex, r2, r2_adj, rmse, mae,
        mean_MR, std_MR, energy, degree,
        intercept, df, fig
    )
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        zf.writestr("main.tex", tex_content)
        zf.writestr("surface_plot.png", img_data)
    zip_buf.seek(0)
    st.download_button(
        "Salvar LaTeX",
        data=zip_buf,
        file_name="Relatorio_Regressao.zip",
        mime="application/zip"
    )

    # Download Word (OMML ou fallback)
    try:
        import pypandoc
        pypandoc.download_pandoc('latest')
        docx_bytes = pypandoc.convert_text(tex_content, 'docx', format='latex')
        st.download_button(
            "Converter para Word (OMML)",
            data=docx_bytes,
            file_name="Relatorio_Regressao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception:
        buf = generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df)
        buf.seek(0)
        st.download_button(
            "Converter para Word (Texto enriquecido)",
            data=buf,
            file_name="Relatorio_Regressao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


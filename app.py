#!/usr/bin/env python
# coding: utf-8

import os
import sys
import base64

# 1) Determina o diretório do script e ajusta o sys.path
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

# Estado inicial
if "calculated" not in st.session_state:
    st.session_state.calculated = False

def reset_results():
    """Limpa resultados quando parâmetros mudam."""
    st.session_state.calculated = False

st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

# Upload
uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

df = pd.read_csv(uploaded, decimal=",") if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write("### Dados Carregados")
st.dataframe(df)

# Configurações
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
degree = st.sidebar.selectbox(
    "Grau (polinomial)",
    [2, 3, 4, 5, 6],
    index=0,
    key="degree",
    on_change=reset_results
) if model_type.startswith("Polinomial") else None
energy = st.sidebar.selectbox(
    "Energia",
    ["Normal", "Intermediária", "Modificada"],
    index=0,
    key="energy",
    on_change=reset_results
)

# Botão Modelo planilha no rodapé da barra lateral
template_path = os.path.join(app_dir, "00_Resilience_Module.xlsx")
try:
    with open(template_path, "rb") as f:
        template_bytes = f.read()
    b64 = base64.b64encode(template_bytes).decode()
    download_link = f'<a download="00_Resilience_Module.xlsx" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"><button style="background-color:#007bff;color:white;width:100%;border:none;padding:0.5rem 0;border-radius:0.25rem;">Modelo planilha</button></a>'
    st.sidebar.markdown(f'<div style="position: sticky; bottom: 0; padding:10px 0;">{download_link}</div>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error("Erro ao carregar modelo de planilha.")

# Cálculo
if st.button("Calcular"):
    # cria barra de progresso
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    def progress_callback(pct, msg):
        progress_bar.progress(pct)
        status_text.text(msg)

    # chama cálculo com callback
    result = calcular_modelo(df, model_type, degree, progress_callback)

    # passo final antes dos relatórios
    status_text.text("Gerando relatórios")
    progress_bar.progress(90)

    # geração de LaTeX/Word
    eq_latex = result["eq_latex"]
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

    try:
        import pypandoc
        pypandoc.download_pandoc("latest")
        docx_bytes = pypandoc.convert_text(tex_content, "docx", format="latex")
    except Exception:
        buf = generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, result["intercept"], df)
        buf.seek(0)
        docx_bytes = buf.read()

    # finaliza progresso
    progress_bar.progress(100)
    st.sidebar.success("Processo concluído")

    # armazena resultados
    st.session_state.calculated = True
    st.session_state.result = result
    st.session_state.metrics_txt = metrics_txt
    st.session_state.fig = fig
    st.session_state.zip_buf = zip_buf
    st.session_state.docx_bytes = docx_bytes

# Exibição de resultados
if st.session_state.calculated:
    res = st.session_state.result

    st.write("### Equação Ajustada")
    st.latex(res["eq_latex"].strip("$$"))

    st.write("### Indicadores Estatísticos")
    indicators = [
        ("R²", f"{res['r2']:.6f}", f"Este valor indica que aproximadamente {res['r2']*100:.2f}% da variabilidade dos dados de MR é explicada pelo modelo."),
        ("R² Ajustado", f"{res['r2_adj']:.6f}", "Essa métrica penaliza o uso excessivo de termos."),
        ("RMSE", f"{res['rmse']:.4f} MPa", f"Erro quadrático médio: {res['rmse']:.4f} MPa."),
        ("MAE", f"{res['mae']:.4f} MPa", f"Erro absoluto médio: {res['mae']:.4f} MPa."),
        ("Média MR", f"{res['mean_MR']:.4f} MPa", "Média dos valores observados."),
        ("Desvio Padrão MR", f"{res['std_MR']:.4f} MPa", "Dispersão dos dados em torno da média.")
    ]
    for name, val, tip in indicators:
        st.markdown(f"**{name}:** {val} <span title='{tip}'>ℹ️</span>", unsafe_allow_html=True)

    st.write(f"**Intercepto:** {res['intercept']:.4f}")
    st.markdown(
        "Função válida para 0,020≤σ₃≤0,14 e 0,02≤σ_d≤0,42 observada a norma DNIT 134/2018‑ME.",
        unsafe_allow_html=True
    )

    # Avaliação da Qualidade do Ajuste
    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    nrmse_range, qual_nrmse, _ = res["quality"]["NRMSE_range"]
    cv_rmse, qual_cv, _ = res["quality"]["CV(RMSE)"]
    mae_pct, qual_mae, _ = res["quality"]["MAE %"]

    st.markdown(
        f"- **NRMSE_range:** {nrmse_range:.2%} → {qual_nrmse} <span title='NRMSE_range: RMSE normalizado pela amplitude dos valores de MR; indicador associado ao RMSE.'>ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **CV(RMSE):** {cv_rmse:.2%} → {qual_cv} <span title='CV(RMSE): coeficiente de variação do RMSE (RMSE/média MR); indicador associado ao RMSE.'>ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **MAE %:** {mae_pct:.2%} → {qual_mae} <span title='MAE %: MAE dividido pela média de MR; indicador associado ao MAE.'>ℹ️</span>",
        unsafe_allow_html=True
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(st.session_state.fig, use_container_width=True)

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

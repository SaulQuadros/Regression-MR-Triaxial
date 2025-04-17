#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st 
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
from docx import Document
from docx.shared import Inches
import base64
import re

# Função para calcular R² ajustado
def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

# Função para construir a equação em LaTeX COM intercepto, com quebras de linha
def build_latex_equation(coefs, intercept, feature_names):
    terms_per_line = 4
    eq_parts = [f"{intercept:.4f}"]
    for coef, term in zip(coefs[1:], feature_names[1:]):
        sign = " + " if coef >= 0 else " - "
        eq_parts.append(sign + f"{abs(coef):.4f}" + term.replace(" ", ""))
    lines, current, count = [], "MR = " + eq_parts[0], 0
    for part in eq_parts[1:]:
        current += part
        count += 1
        if count % terms_per_line == 0:
            lines.append(current)
            current = ""
    if current.strip():
        lines.append(current)
    return "$$" + " \\\\ \n".join(lines) + "$$"

# Função para construir a equação em LaTeX SEM intercepto
def build_latex_equation_no_intercept(coefs, feature_names):
    terms_per_line = 4
    # começa pelo primeiro coeficiente (coef de σ₃)
    eq_parts = [f"{coefs[0]:.4f}" + feature_names[0].replace(" ", "")]
    for coef, term in zip(coefs[1:], feature_names[1:]):
        sign = " + " if coef >= 0 else " - "
        eq_parts.append(sign + f"{abs(coef):.4f}" + term.replace(" ", ""))
    lines, current, count = [], "MR = " + eq_parts[0], 0
    for part in eq_parts[1:]:
        current += part
        count += 1
        if count % terms_per_line == 0:
            lines.append(current)
            current = ""
    if current.strip():
        lines.append(current)
    return "$$" + " \\\\ \n".join(lines) + "$$"

# Converte string de equação para parágrafo Word, formatando expoentes e subscritos
def add_formatted_equation(document, equation_text):
    eq = equation_text.strip().strip("$").strip()
    p = document.add_paragraph(); i = 0
    while i < len(eq):
        if eq[i] == '^':
            i += 1; exp = ""
            while i < len(eq) and (eq[i].isdigit() or eq[i] in ['.', '-']):
                exp += eq[i]; i += 1
            r = p.add_run(exp); r.font.superscript = True
        elif eq[i] == '~':
            i += 1
            if i < len(eq):
                r = p.add_run(eq[i]); r.font.subscript = True; i += 1
        else:
            r = p.add_run(eq[i]); i += 1
    return p

# Insere tabela de dados no Word
def add_data_table(document, df):
    document.add_heading("Dados do Ensaio Triaxial", level=2)
    rows, cols = df.shape[0] + 1, df.shape[1]
    table = document.add_table(rows=rows, cols=cols)
    table.style = 'Light List Accent 1'
    for j, col in enumerate(df.columns):
        table.rows[0].cells[j].text = str(col)
    for i in range(df.shape[0]):
        for j, col in enumerate(df.columns):
            table.rows[i+1].cells[j].text = str(df.iloc[i, j])
    return document

# Gera gráfico 3D
def plot_3d_surface(df, model, poly, energy_col):
    s3 = np.linspace(df["σ3"].min(), df["σ3"].max(), 30)
    sd = np.linspace(df["σd"].min(), df["σd"].max(), 30)
    s3g, sdg = np.meshgrid(s3, sd)
    Xg = np.c_[s3g.ravel(), sdg.ravel()]
    MRg = model.predict(poly.transform(Xg)).reshape(s3g.shape)
    fig = go.Figure(data=[go.Surface(x=s3g, y=sdg, z=MRg, colorscale='Viridis')])
    fig.add_trace(go.Scatter3d(
        x=df["σ3"], y=df["σd"], z=df[energy_col],
        mode='markers', marker=dict(size=5, color='red'), name="Dados"
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='σ₃ (MPa)',
            yaxis_title='σ<sub>d</sub> (MPa)',
            zaxis_title='MR (MPa)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

# Interpretação das métricas
def interpret_metrics(r2, r2_adj, rmse, mae, y):
    text = (
        f"**R²:** {r2:.6f}. Aproximadamente {r2*100:.2f}% da variabilidade de MR explicada.\n\n"
        f"**R² Ajustado:** {r2_adj:.6f}. Penaliza termos extras, indicando bom ajuste.\n\n"
        f"**RMSE:** {rmse:.4f} MPa.\n\n"
        f"**MAE:** {mae:.4f} MPa.\n\n"
        f"**Média de MR:** {y.mean():.4f} MPa.\n\n"
        f"**Desvio Padrão de MR:** {y.std():.4f} MPa.\n\n"
    )
    return text

# Gera documento Word completo
def generate_word_doc(equation_latex, metrics_text, fig, energy_type, degree, intercept, df):
    doc = Document()
    doc.add_heading("Relatório de Regressão Polinomial", level=1)
    doc.add_heading("Configurações do Modelo", level=2)
    doc.add_paragraph(f"Tipo de energia: {energy_type}")
    doc.add_paragraph(f"Grau do polinomial: {degree}")
    doc.add_heading("Equação de Regressão", level=2)
    doc.add_paragraph("Equação ajustada:")
    eqw = equation_latex.replace("\\\\", " ").replace("σ_d","σ~d")
    add_formatted_equation(doc, eqw)
    doc.add_heading("Indicadores Estatísticos", level=2)
    doc.add_paragraph(metrics_text)
    doc.add_paragraph(f"**Intercepto:** {intercept:.4f}")
    doc.add_paragraph(
        "A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e "
        "0,02≤$\\sigma_{d}$≤0,42 observada a norma: DNIT 134/2018-ME "
        "(versão corrigida em 20/04/2023)."
    )
    doc.add_page_break()
    add_data_table(doc, df)
    doc.add_heading("Gráfico 3D da Superfície", level=2)
    img = fig.to_image(format="png")
    doc.add_picture(BytesIO(img), width=Inches(6))
    buf = BytesIO(); doc.save(buf)
    return buf

# --- Streamlit App ---
st.set_page_config(page_title="Reg. Polinomial MR", layout="wide")
st.title("Regressão Polinomial para MR")
st.markdown(
    "Upload de tabela com *σ₃*, *σ_d* e *MR* e ajuste de polinomial (grau 2–6)."
)

uploaded_file = st.file_uploader("Upload CSV ou Excel", type=["csv","xlsx"])
if uploaded_file:
    df = (pd.read_csv(uploaded_file, decimal=",")
          if uploaded_file.name.endswith(".csv")
          else pd.read_excel(uploaded_file))
    st.write("### Dados Carregados"); st.dataframe(df)

    st.sidebar.header("Configurações do Modelo")
    model_type = st.sidebar.selectbox(
        "Escolha o modelo de regressão",
        ["Polinomial c/ Intercepto", "Polinomial s/Intercepto"],
        index=0
    )
    degree = st.sidebar.selectbox("Grau do polinomial", [2,3,4,5,6], index=0)
    energy_type = st.sidebar.selectbox("Tipo de energia", ["Normal","Intermediária","Modificada"], index=0)

    if st.button("Calcular"):
        X = df[["σ3","σd"]].values; y = df["MR"].values
        poly = PolynomialFeatures(degree=degree)
        Xp = poly.fit_transform(X)

        # Seleciona modelo com ou sem intercepto
        if model_type == "Polinomial s/Intercepto":
            reg = LinearRegression(fit_intercept=False)
        else:
            reg = LinearRegression()
        reg.fit(Xp, y)
        y_pred = reg.predict(Xp)

        r2 = r2_score(y, y_pred)
        n, p = len(y), Xp.shape[1] - (0 if model_type.startswith("Polinomial s/") else 1)
        r2_adj = adjusted_r2(r2, n, p)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        feature_names = poly.get_feature_names_out(["σ₃","σ_d"])
        # Gera equação apropriada
        if model_type == "Polinomial s/Intercepto":
            eq_latex = build_latex_equation_no_intercept(reg.coef_, feature_names)
        else:
            eq_latex = build_latex_equation(reg.coef_, reg.intercept_, feature_names)

        metrics_txt = interpret_metrics(r2, r2_adj, rmse, mae, y)

        st.write("### Equação de Regressão")
        st.latex(eq_latex.strip("$$"))

        st.write("### Indicadores Estatísticos")
        st.markdown(metrics_txt)
        st.write(f"**Intercepto:** {getattr(reg,'intercept_',0):.4f}")

        st.markdown(
            "A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e "
            "0,02≤$\\sigma_{d}$≤0,42 observada a norma: DNIT 134/2018-ME "
            "(versão corrigida em 20/04/2023)."
        )

        fig = plot_3d_surface(df, reg, poly, "MR")
        st.write("### Gráfico 3D da Superfície")
        st.plotly_chart(fig, use_container_width=True)

        buf = generate_word_doc(eq_latex, metrics_txt, fig,
                                energy_type, degree,
                                getattr(reg,'intercept_',0), df)
        buf.seek(0)
        st.download_button(
            "Salvar Word",
            data=buf,
            file_name="Relatorio_Regressao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


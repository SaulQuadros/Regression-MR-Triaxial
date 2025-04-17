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
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from docx import Document
from docx.shared import Inches

# --- Funções Auxiliares ---

def adjusted_r2(r2, n, p):
    """Retorna R² ajustado."""
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def build_latex_equation(coefs, intercept, feature_names):
    terms_per_line = 4
    parts = []
    for coef, term in zip(coefs, feature_names):
        sign = " + " if coef >= 0 else " - "
        parts.append(f"{sign}{abs(coef):.4f}{term.replace(' ', '')}")
    lines = []
    curr = f"MR = {intercept:.4f}"
    for i, part in enumerate(parts):
        curr += part
        if (i + 1) % terms_per_line == 0:
            lines.append(curr)
            curr = ""
    if curr.strip():
        lines.append(curr)
    return "$$" + " \\\\ \n".join(lines) + "$$"

def build_latex_equation_no_intercept(coefs, feature_names):
    terms_per_line = 4
    parts = []
    for coef, term in zip(coefs, feature_names):
        sign = " + " if coef >= 0 else " - "
        parts.append(f"{sign}{abs(coef):.4f}{term.replace(' ', '')}")
    lines = []
    curr = f"MR = {coefs[0]:.4f}{feature_names[0].replace(' ', '')}"
    for i, part in enumerate(parts[1:]):
        curr += part
        if (i + 1) % terms_per_line == 0:
            lines.append(curr)
            curr = ""
    if curr.strip():
        lines.append(curr)
    return "$$" + " \\\\ \n".join(lines) + "$$"

def add_formatted_equation(doc, eq_text):
    eq = eq_text.strip().strip("$")
    p = doc.add_paragraph()
    i = 0
    while i < len(eq):
        if eq[i] == '^':
            i += 1
            exp = ""
            while i < len(eq) and (eq[i].isdigit() or eq[i] in ['.', '-']):
                exp += eq[i]
                i += 1
            r = p.add_run(exp); r.font.superscript = True
        elif eq[i] == '~':
            i += 1
            if i < len(eq):
                r = p.add_run(eq[i]); r.font.subscript = True
                i += 1
        else:
            r = p.add_run(eq[i]); i += 1
    return p

def add_data_table(doc, df):
    doc.add_heading("Dados do Ensaio Triaxial", level=2)
    table = doc.add_table(rows=df.shape[0]+1, cols=df.shape[1])
    table.style = 'Light List Accent 1'
    for j, col in enumerate(df.columns):
        table.rows[0].cells[j].text = str(col)
    for i in range(df.shape[0]):
        for j, col in enumerate(df.columns):
            table.rows[i+1].cells[j].text = str(df.iloc[i, j])
    return doc

def plot_3d_surface(df, model, poly, energy_col, is_power=False, power_params=None):
    s3 = np.linspace(df["σ3"].min(), df["σ3"].max(), 30)
    sd = np.linspace(df["σd"].min(), df["σd"].max(), 30)
    s3g, sdg = np.meshgrid(s3, sd)
    Xg = np.c_[s3g.ravel(), sdg.ravel()]
    MRg = (model(Xg, *power_params) if is_power 
           else model.predict(poly.transform(Xg)))
    MRg = MRg.reshape(s3g.shape)
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

def interpret_metrics(r2, r2_adj, rmse, mae, y):
    txt = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
    txt += f"**R² Ajustado:** {r2_adj:.6f}\n\n"
    txt += (
        f"**RMSE:** {rmse:.4f} MPa\n\n"
        f"**MAE:** {mae:.4f} MPa\n\n"
        f"**Média MR:** {y.mean():.4f} MPa\n\n"
        f"**Desvio Padrão MR:** {y.std():.4f} MPa\n\n"
    )
    return txt

def generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df):
    doc = Document()
    doc.add_heading("Relatório de Regressão", level=1)
    doc.add_heading("Configurações", level=2)
    doc.add_paragraph(f"Tipo de energia: {energy}")
    if degree is not None:
        doc.add_paragraph(f"Grau polinomial: {degree}")
    doc.add_heading("Equação Ajustada", level=2)
    add_formatted_equation(doc, eq_latex)
    doc.add_heading("Indicadores Estatísticos", level=2)
    doc.add_paragraph(metrics_txt)
    doc.add_paragraph(f"**Intercepto:** {intercept:.4f}")
    doc.add_paragraph(
        "A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e "
        "0,02≤$\\sigma_{d}$≤0,42 observada a norma DNIT 134/2018‑ME."
    )
    doc.add_page_break()
    add_data_table(doc, df)
    doc.add_heading("Gráfico 3D da Superfície", level=2)
    img = fig.to_image(format="png")
    doc.add_picture(BytesIO(img), width=Inches(6))
    buf = BytesIO(); doc.save(buf)
    return buf

# --- Streamlit App ---
st.set_page_config(page_title="Modelos de MR", layout="wide")
st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

df = (pd.read_csv(uploaded, decimal=",") 
      if uploaded.name.endswith(".csv") 
      else pd.read_excel(uploaded))
st.write("### Dados Carregados")
st.dataframe(df)

# --- Configurações na barra lateral ---
st.sidebar.header("Configurações")
model_type = st.sidebar.selectbox(
    "Modelo de regressão",
    [
        "Polinomial c/ Intercepto",
        "Polinomial s/Intercepto",
        "Potência Composta c/ Intercepto",
        "Potência Composta s/Intercepto"
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
    X = df[["σ3", "σd"]].values
    y = df["MR"].values

    if model_type in ("Polinomial c/ Intercepto", "Polinomial s/Intercepto"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = poly.fit_transform(X)
        fit_int = (model_type == "Polinomial c/ Intercepto")
        reg = LinearRegression(fit_intercept=fit_int)
        reg.fit(Xp, y)
        y_pred = reg.predict(Xp)

        r2 = r2_score(y, y_pred)
        p_feat = Xp.shape[1]
        if len(y) > p_feat + 1:
            raw = adjusted_r2(r2, len(y), p_feat)
            r2_adj = min(raw, r2, 1.0)
        else:
            r2_adj = r2

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        fnames = poly.get_feature_names_out(["σ₃", "σ_d"])
        if fit_int:
            coefs = np.concatenate(([reg.intercept_], reg.coef_))
            feature_names = [""] + fnames.tolist()
            eq_latex = build_latex_equation(coefs, reg.intercept_, feature_names)
            intercept = reg.intercept_
        else:
            eq_latex = build_latex_equation_no_intercept(reg.coef_, fnames)
            intercept = 0.0

        is_power = False
        power_params = None
        model_obj = reg

    else:
        def pot_model(X_flat, a0, a1, k1, a2, k2, a3, k3):
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return a0 + a1 * s3**k1 + a2 * (s3 * sd)**k2 + a3 * sd**k3

        # --- palpite inteligente para os parâmetros iniciais ---
        mean_y = y.mean()
        mean_s3 = X[:,0].mean()
        mean_sd = X[:,1].mean()
        mean_s3sd = (X[:,0]*X[:,1]).mean()
        p0 = [
            mean_y,
            mean_y/mean_s3, 1,
            mean_y/mean_s3sd, 1,
            mean_y/mean_sd, 1
        ]

        try:
            popt, _ = curve_fit(pot_model, X, y, p0=p0, maxfev=200000)
        except RuntimeError:
            st.error(
                "❌ Não foi possível ajustar o modelo de Potência Composta. "
                "Verifique seus dados ou tente outro modelo."
            )
            st.stop()

        y_pred = pot_model(X, *popt)
        r2 = r2_score(y, y_pred)
        if len(y) > len(popt) + 1:
            raw = adjusted_r2(r2, len(y), len(popt))
            r2_adj = min(raw, r2, 1.0)
        else:
            r2_adj = r2

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        a0, a1, k1, a2, k2, a3, k3 = popt
        has_int = model_type.endswith("c/ Intercepto")
        if not has_int:
            eq_body = (
                f"{a1:.4f}\\sigma_3^{{{k1:.4f}}}"
                f" + {a2:.4f}(\\sigma_3\\sigma_d)^{{{k2:.4f}}}"
                f" + {a3:.4f}\\sigma_d^{{{k3:.4f}}}"
            )
            eq_latex = "$$MR = " + eq_body + "$$"
            intercept = 0.0
        else:
            eq_latex = (
                f"$$MR = {a0:.4f} + {a1:.4f}\\sigma_3^{{{k1:.4f}}}"
                f" + {a2:.4f}(\\sigma_3\\sigma_d)^{{{k2:.4f}}}"
                f" + {a3:.4f}\\sigma_d^{{{k3:.4f}}}$$"
            )
            intercept = a0

        is_power = True
        power_params = popt
        model_obj = pot_model
        poly = None

    metrics_txt = interpret_metrics(r2, r2_adj, rmse, mae, y)
    fig = plot_3d_surface(df, model_obj, poly, "MR",
                          is_power=is_power, power_params=power_params)

    st.write("### Equação Ajustada")
    st.latex(eq_latex.strip("$$"))

    st.write("### Indicadores Estatísticos")
    st.markdown(metrics_txt)

    st.write(f"**Intercepto:** {intercept:.4f}")
    st.markdown(
        "A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e "
        "0,02≤$\\sigma_{d}$≤0,42 observada a norma DNIT 134/2018‑ME."
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(fig, use_container_width=True)

    buf = generate_word_doc(eq_latex, metrics_txt, fig, energy,
                            degree, intercept, df)
    buf.seek(0)
    st.download_button(
        "Salvar Word",
        data=buf,
        file_name="Relatorio_Regressao.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )



#!/usr/bin/env python
# coding: utf-8

import streamlit as st  
import pandas as pd
import numpy as np
from io import BytesIO
import zipfile
import io
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from docx import Document
from docx.shared import Inches

# --- Funções Auxiliares ---

def adjusted_r2(r2, n, p):
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
    eq = eq_text.strip().strip("$$")
    p = doc.add_paragraph()
    i = 0
    while i < len(eq):
        ch = eq[i]
        if ch == '^':
            i += 1
            exp = ""
            while i < len(eq) and (eq[i].isdigit() or eq[i] in ['.', '-']):
                exp += eq[i]; i += 1
            run = p.add_run(exp); run.font.superscript = True
        elif ch in ['_', '~']:
            i += 1
            if i < len(eq):
                run = p.add_run(eq[i]); run.font.subscript = True; i += 1
        elif ch == 'σ':
            run_sigma = p.add_run('σ'); i += 1
            if i < len(eq) and (eq[i].isdigit() or eq[i].isalpha()):
                run_sub = p.add_run(eq[i]); run_sub.font.subscript = True; i += 1
        else:
            p.add_run(ch); i += 1
    return p

def add_data_table(doc, df):
    doc.add_heading("Dados do Ensaio Triaxial", level=2)
    table = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])
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
    MRg = (model(Xg, *power_params) if is_power else model.predict(poly.transform(Xg)))
    MRg = MRg.reshape(s3g.shape)
    fig = go.Figure(data=[go.Surface(x=s3g, y=sdg, z=MRg, colorscale='Viridis')])
    fig.add_trace(go.Scatter3d(x=df["σ3"], y=df["σd"], z=df[energy_col],
                               mode='markers', marker=dict(size=5, color='red'), name="Dados"))
    fig.update_layout(scene=dict(xaxis_title='σ₃ (MPa)', yaxis_title='σ_d (MPa)', zaxis_title='MR (MPa)'),
                      margin=dict(l=0, r=0, b=0, t=30))
    return fig

def interpret_metrics(r2, r2_adj, rmse, mae, y):
    amp = y.max() - y.min()
    nrmse_range = rmse / amp if amp > 0 else float("nan")
    cv_rmse = rmse / y.mean() if y.mean() else float("nan")
    mae_pct = mae / y.mean() if y.mean() else float("nan")
    return r2, r2_adj, rmse, mae, amp, nrmse_range, cv_rmse, mae_pct

def generate_latex_doc(eq_latex, r2, r2_adj, rmse, mae, mean_MR, std_MR,
                       energy, degree, intercept, df, fig):
    lines = [r"\documentclass{article}", r"\usepackage[utf8]{inputenc}",
             r"\usepackage{booktabs,graphicx}", r"\begin{document}",
             r"\section*{Relatório de Regressão}",
             r"\subsection*{Configurações}", f"Tipo de energia: {energy}\\\\",
             *( [f"Grau polinomial: {degree}\\\\" ] if degree else []),
             r"\subsection*{Equação Ajustada}", eq_latex,
             r"\subsection*{Indicadores Estatísticos}", r"\begin{itemize}",
             f"  \\item \\textbf{{R$^2$}}: {r2:.6f} (~{r2*100:.2f}\\%)",
             f"  \\item \\textbf{{R$^2$ Ajustado}}: {r2_adj:.6f}",
             f"  \\item \\textbf{{RMSE}}: {rmse:.4f} MPa",
             f"  \\item \\textbf{{MAE}}: {mae:.4f} MPa",
             f"  \\item \\textbf{{Média MR}}: {mean_MR:.4f} MPa",
             f"  \\item \\textbf{{Desvio Padrão MR}}: {std_MR:.4f} MPa", r"\end{itemize}",
             r"\subsection*{Avaliação da Qualidade do Ajuste}", r"\begin{itemize}",
             f"  \\item \\textbf{{NRMSE_range}}: {rmse/amp:.2%}",
             f"  \\item \\textbf{{CV(RMSE)}}: {cv_rmse:.2%}",
             f"  \\item \\textbf{{MAE \\%}}: {mae_pct:.2%}", r"\end{itemize}",
             f"Intercepto: {intercept:.4f}\\\\", r"\newpage",
             r"\section*{Dados do Ensaio Triaxial}",
             r"\begin{tabular}{" + "l"*len(df.columns) + r"}",
             " & ".join(df.columns) + r" \\ \midrule",
             *[ " & ".join(str(v) for v in row.values) + r" \\" for _, row in df.iterrows() ],
             r"\end{tabular}", r"\section*{Gráfico 3D da Superfície}",
             r"\includegraphics[width=\linewidth]{surface_plot.png}", r"\end{document}"]
    tex_content = "\n".join(lines)
    img_data = fig.to_image(format="png")
    return tex_content, img_data

# Streamlit App continues unchanged...
# ... (rest of the code unchanged, generating Word doc and zip download with main.tex)

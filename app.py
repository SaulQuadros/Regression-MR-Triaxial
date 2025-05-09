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
    return "$$" + " \\ 
".join(lines) + "$$"


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
    return "$$" + " \\ 
".join(lines) + "$$"


def add_formatted_equation(doc, eq_text):
    """
    Adiciona a equação ao Word, formatando:
    - σ seguido de subscrito
    - ^ para sobrescrito
    - _ ou ~ para subscrito
    """
    eq = eq_text.strip().strip("$$")
    p = doc.add_paragraph()
    i = 0
    while i < len(eq):
        ch = eq[i]
        if ch == '^':
            # superscrito
            i += 1
            exp = ""
            while i < len(eq) and (eq[i].isdigit() or eq[i] in ['.', '-']):
                exp += eq[i]
                i += 1
            run = p.add_run(exp)
            run.font.superscript = True
        elif ch in ['_', '~']:
            # subescrito
            i += 1
            if i < len(eq):
                run = p.add_run(eq[i])
                run.font.subscript = True
                i += 1
        elif ch == 'σ':
            # sigma + possível subscrito
            run_sigma = p.add_run('σ')
            i += 1
            if i < len(eq) and (eq[i].isdigit() or eq[i].isalpha()):
                run_sub = p.add_run(eq[i])
                run_sub.font.subscript = True
                i += 1
        else:
            p.add_run(ch)
            i += 1
    return p


def add_data_table(doc, df):
    doc.add_heading("Dados do Ensaio Triaxial", level=2)
    table = doc.add_table(rows=df.shape[0] + 1, cols=df.shape[1])
    table.style = 'Light List Accent 1'
    # cabeçalho
    for j, col in enumerate(df.columns):
        table.rows[0].cells[j].text = str(col)
    # dados
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
    fig = go.Figure(data=[go.Surface(x=s3g, y=sdg, z=MRg)])
    fig.add_trace(go.Scatter3d(
        x=df["σ3"], y=df["σd"], z=df[energy_col],
        mode='markers', marker=dict(size=5, color='red'), name="Dados"
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='σ₃ (MPa)',
            yaxis_title='σd (MPa)',
            zaxis_title='MR (MPa)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig


def interpret_metrics(r2, r2_adj, rmse, mae, y):
    """Gera texto para relatório Word."""
    txt = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)

"
    txt += f"**R² Ajustado:** {r2_adj:.6f}

"
    txt += f"**RMSE:** {rmse:.4f} MPa

"
    txt += f"**MAE:** {mae:.4f} MPa

"
    txt += f"**Média MR:** {y.mean():.4f} MPa

"
    txt += f"**Desvio Padrão MR:** {y.std():.4f} MPa

"
    return txt




def generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df):
    from io import BytesIO
    from docx.shared import Inches
    import re

    doc = Document()
    doc.add_heading("Relatório de Regressão", level=1)
    doc.add_heading("Configurações", level:2)
    doc.add_paragraph(f"Tipo de energia: {energy}")
    if degree is not None:
        doc.add_paragraph(f"Grau polinomial: {degree}")

    doc.add_heading("Equação Ajustada", level=2)
    add_formatted_equation(doc, eq_latex.strip("$$"))

    doc.add_heading("Indicadores Estatísticos", level=2)
    doc.add_paragraph(metrics_txt)

    amplitude = float(df["MR"].max() - df["MR"].min())
    max_mr = float(df["MR"].max())
    min_mr = float(df["MR"].min())

    rmse_match = re.search(r"RMSE:\s*([0-9\.]+)", metrics_txt)
    mae_match = re.search(r"MAE:\s*([0-9\.]+)", metrics_txt)
    rmse_val = float(rmse_match.group(1)) if rmse_match else float("nan")
    mae_val = float(mae_match.group(1)) if mae_match else float("nan")

    params = [
        ("Amplitude", f"{amplitude:.4f} MPa"),
        ("MR Máximo", f"{max_mr:.4f} MPa"),
        ("MR Mínimo", f"{min_mr:.4f} MPa"),
        ("Intercepto", f"{intercept:.4f} MPa")
    ]
    for name, val in params:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"**{name}:** {val}")

    quality = [
        ("NRMSE_range", f"{rmse_val/amplitude:.2%}" if amplitude>0 else "nan"),
        ("CV(RMSE)", f"{rmse_val/df["MR"].mean():.2%}" if df["MR"].mean()!=0 else "nan"),
        ("MAE %", f"{mae_val/df["MR"].mean():.2%}" if df["MR"].mean()!=0 else "nan")
    ]
    doc.add_heading("Avaliação da Qualidade do Ajuste", level=2)
    for name, val in quality:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"**{name}:** {val}")

    doc.add_page_break()
    add_data_table(doc, df)
    doc.add_heading("Gráfico 3D da Superfície", level=2)
    img = fig.to_image(format="png")
    doc.add_picture(BytesIO(img), width=Inches(6))
    buf = BytesIO()
    doc.save(buf)
    return buf

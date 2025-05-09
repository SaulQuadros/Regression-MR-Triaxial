#!/usr/bin/env python
# coding: utf-8

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
    """
    Adiciona a equação ao Word, formatando:
    - σ seguido de subscrito
    - ^{...} ou _{...} para sobrescrito/subscrito
    """
    eq = eq_text.strip().strip("$$")
    p = doc.add_paragraph()
    i = 0
    while i < len(eq):
        ch = eq[i]
        if ch in ('^', '_'):
            is_sup = (ch == '^')
            i += 1
            # Handle brace-enclosed or single-character
            if i < len(eq) and eq[i] == '{':
                i += 1
                content = ''
                # collect until closing brace
                while i < len(eq) and eq[i] != '}':
                    content += eq[i]
                    i += 1
                i += 1  # skip '}'
            else:
                content = eq[i]
                i += 1
            run = p.add_run(content)
            if is_sup:
                run.font.superscript = True
            else:
                run.font.subscript = True
        elif ch == 'σ':
            # sigma plus optional subscript in LaTeX (_{n}) or direct
            run_sigma = p.add_run('σ')
            i += 1
            if i < len(eq) and eq[i] == '_':
                i += 1
                if i < len(eq) and eq[i] == '{':
                    i += 1
                    sub = ''
                    while i < len(eq) and eq[i] != '}':
                        sub += eq[i]
                        i += 1
                    i += 1
                else:
                    sub = eq[i]
                    i += 1
                run_sub = p.add_run(sub)
                run_sub.font.subscript = True
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
    fig = go.Figure(data=[go.Surface(x=s3g, y=sdg, z=MRg, colorscale='Viridis')])
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
    txt = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
    txt += f"**R² Ajustado:** {r2_adj:.6f}\n\n"
    txt += f"**RMSE:** {rmse:.4f} MPa\n\n"
    txt += f"**MAE:** {mae:.4f} MPa\n\n"
    txt += f"**Média MR:** {y.mean():.4f} MPa\n\n"
    txt += f"**Desvio Padrão MR:** {y.std():.4f} MPa\n\n"
    return txt




def generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df):
    from io import BytesIO
    from docx.shared import Inches
    import re

    doc = Document()
    doc.add_heading("Relatório de Regressão", level=1)
    doc.add_heading("Configurações", level=2)
    doc.add_paragraph(f"Tipo de energia: {energy}")
    if degree is not None:
        doc.add_paragraph(f"Grau polinomial: {degree}")

    doc.add_heading("Equação Ajustada", level=2)
    # Exibe a equação completa sem quebras forçadas
    add_formatted_equation(doc, eq_latex.strip("$$"))

    # Indicadores Estatísticos
    doc.add_heading("Indicadores Estatísticos", level=2)
    doc.add_paragraph(metrics_txt)

    # Cálculo de amplitude e extremos
    amplitude = float(df["MR"].max() - df["MR"].min())
    max_mr = float(df["MR"].max())
    min_mr = float(df["MR"].min())

    # Parse RMSE e MAE do metrics_txt
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

    # Avaliação da Qualidade do Ajuste
    nrmse_range = rmse_val / amplitude if amplitude > 0 else float("nan")
    cv_rmse = rmse_val / df["MR"].mean() if df["MR"].mean() != 0 else float("nan")
    mae_pct = mae_val / df["MR"].mean() if df["MR"].mean() != 0 else float("nan")
    quality = [
        ("NRMSE_range", f"{nrmse_range:.2%}"),
        ("CV(RMSE)", f"{cv_rmse:.2%}"),
        ("MAE %", f"{mae_pct:.2%}")
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




def generate_latex_doc(eq_latex, r2, r2_adj, rmse, mae,
                       mean_MR, std_MR, energy, degree,
                       intercept, df, fig):
    # Monta o documento LaTeX e retorna o conteúdo e imagem
    lines = []
    lines.append(r"\documentclass{article}")
    lines.append(r"\usepackage[utf8]{inputenc}")
    lines.append(r"\usepackage{booktabs,graphicx}")
    lines.append(r"\begin{document}")
    lines.append(r"\section*{Relatório de Regressão}")
    lines.append(r"\subsection*{Configurações}")
    lines.append(f"Tipo de energia: {energy}\\\\")
    if degree is not None:
        lines.append(f"Grau polinomial: {degree}\\\\")
    lines.append(r"\subsection*{Equação Ajustada}")
    lines.append(eq_latex)

    # Indicadores Estatísticos
    lines.append(r"\subsection*{Indicadores Estatísticos}")
    lines.append(r"\begin{itemize}")
    lines.append(f"  \\item \\textbf{{R$^2$}}: {r2:.6f} (aprox. {r2*100:.2f}\\% explicado)")
    lines.append(f"  \\item \\textbf{{R$^2$ Ajustado}}: {r2_adj:.6f}")
    lines.append(f"  \\item \\textbf{{RMSE}}: {rmse:.4f} MPa")
    lines.append(f"  \\item \\textbf{{MAE}}: {mae:.4f} MPa")
    lines.append(f"  \\item \\textbf{{Média MR}}: {mean_MR:.4f} MPa")
    lines.append(f"  \\item \\textbf{{Desvio Padrão MR}}: {std_MR:.4f} MPa")
    lines.append(r"\end{itemize}")

    # Avaliação da Qualidade do Ajuste
    amp = df["MR"].max() - df["MR"].min()
    nrmse_range = rmse / amp if amp > 0 else float("nan")
    cv_rmse     = rmse / mean_MR if mean_MR != 0 else float("nan")
    mae_pct     = mae  / mean_MR if mean_MR  != 0 else float("nan")

    lines.append(r"\subsection*{Avaliação da Qualidade do Ajuste}")
    lines.append(r"\begin{itemize}")
    lines.append(f"  \\item \\textbf{{NRMSE\_range}}: {nrmse_range:.2%}")
    lines.append(f"  \\item \\textbf{{CV(RMSE)}}: {cv_rmse:.2%}")
    lines.append(f"  \\item \\textbf{{MAE \\%}}: {mae_pct:.2%}")
    lines.append(r"\end{itemize}")

    # Intercepto e demais seções
    lines.append(f"Intercepto: {intercept:.4f}\\\\")
    lines.append(r"\newpage")

    # Tabela de dados
    cols = len(df.columns)
    lines.append(r"\section*{Dados do Ensaio Triaxial}")
    lines.append(r"\begin{tabular}{" + "l" * cols + r"}")
    lines.append(" & ".join(df.columns) + r" \\ \midrule")
    for _, row in df.iterrows():
        vals = [str(v) for v in row.values]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\end{tabular}")

    # Gráfico 3D
    lines.append(r"\section*{Gráfico 3D da Superfície}")
    lines.append(r"\includegraphics[width=\linewidth]{surface_plot.png}")
    lines.append(r"\end{document}")

    # gera bytes da figura
    img_data = fig.to_image(format="png")
    tex_content = "\n".join(lines)
    return tex_content, img_data

# --- Streamlit App ---

st.set_page_config(page_title="Modelos de MR", layout="wide")
st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

# --- Botão Modelo planilha ---
try:
    with open("00_Resilience_Module.xlsx", "rb") as f:
        st.sidebar.download_button(
            label="📥 Modelo planilha",
            data=f,
            file_name="00_Resilience_Module.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_modelo_planilha"
        )
except FileNotFoundError:
    st.sidebar.warning("Arquivo de modelo não encontrado.")

uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

df = (pd.read_csv(uploaded, decimal=",") 
      if uploaded.name.endswith(".csv") 
      else pd.read_excel(uploaded))
st.write("### Dados Carregados")
st.dataframe(df)

# --- Persistência de resultados ---
if "calculated" not in st.session_state:
    st.session_state.calculated = False

# --- Configurações na barra lateral ---
st.sidebar.header("Configurações")
model_type = st.sidebar.selectbox(
    "Escolha o modelo de regressão",
    [
        "Polinomial c/ Intercepto",
        "Polinomial s/Intercepto",
        "Potência Composta",
        "Witczak",
        "Pezo"
    ]
)

# Novo: opção de normalização somente para Pezo
if model_type == "Pezo":
    pezo_option = st.sidebar.selectbox(
        "Pezo – Tipo",
        ["Normalizada", "Não normalizada"],
        index=0
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
    intercept = 0.0  # inicializa intercept
    st.session_state.calculated = True
    X = df[["σ3", "σd"]].values
    y = df["MR"].values

    # — Modelo Polinomial —
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
        mae  = mean_absolute_error(y, y_pred)

        fnames = poly.get_feature_names_out(["σ₃", "σ_d"])
        if fit_int:
            coefs = reg.coef_
            feature_names = fnames.tolist()
            eq_latex = build_latex_equation(coefs, reg.intercept_, feature_names)
        else:
            eq_latex = build_latex_equation_no_intercept(reg.coef_, fnames)
            intercept = 0.0

        is_power = False
        power_params = None
        model_obj = reg
        poly_obj   = poly

    # — Modelo de Potência Composta sem intercepto —
    elif model_type == "Potência Composta":
        def pot_no_int(X_flat, a1, k1, a2, k2, a3, k3):
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return a1 * s3**k1 + a2 * (s3 * sd)**k2 + a3 * sd**k3

        mean_y     = y.mean()
        mean_s3    = X[:, 0].mean()
        mean_sd    = X[:, 1].mean()
        mean_s3sd  = (X[:, 0] * X[:, 1]).mean()

        p0_no   = [mean_y/mean_s3, 1,
                   mean_y/mean_s3sd, 1,
                   mean_y/mean_sd, 1]

        fit_func, p0 = pot_no_int, p0_no

        try:
            popt, _ = curve_fit(fit_func, X, y, p0=p0, maxfev=200000)
        except RuntimeError:
            st.error("❌ Não foi possível ajustar o modelo de Potência Composta.")
            st.stop()

        y_pred = fit_func(X, *popt)
        r2     = r2_score(y, y_pred)
        if len(y) > len(popt) + 1:
            raw = adjusted_r2(r2, len(y), len(popt))
            r2_adj = min(raw, r2, 1.0)
        else:
            r2_adj = r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae  = mean_absolute_error(y, y_pred)

        a1, k1, a2, k2, a3, k3 = popt
        # Construção da equação LaTeX com sinal dinâmico
        terms = [
            (a1, f"σ₃^{{{k1:.4f}}}"),
            (a2, f"(σ₃σ_d)^{{{k2:.4f}}}"),
            (a3, f"σ_d^{{{k3:.4f}}}"),
        ]
        eq = "$$MR = "
        eq += f"{terms[0][0]:.4f}{terms[0][1]}"
        for coef, term in terms[1:]:
            sign = " + " if coef >= 0 else " - "
            eq += f"{sign}{abs(coef):.4f}{term}"
        eq += "$$"
        eq_latex = eq
        intercept = 0.0

        is_power     = True
        power_params = popt
        model_obj    = fit_func
        poly_obj     = None

    
    # — Modelo Witczak —
    elif model_type == "Witczak":
        def witczak_model(X_flat, k1, k2, k3):
            Pa = 0.101325
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            θ = sd + 3 * s3
            # MR = k1 * (θ^k2)/Pa * (σ_d^k3)/Pa
            return k1 * (θ**k2 / Pa) * (sd**k3 / Pa)

        # estimativas iniciais
        mean_y      = y.mean()
        Pa_display  = 0.101325
        θ_arr       = X[:, 1] + 3 * X[:, 0]
        mean_θ      = θ_arr.mean()
        mean_sd     = X[:, 1].mean()
        # a partir de MR = k1 * (mean_θ * mean_sd)/(Pa*Pa) => k1 = MR*(Pa*Pa)/(mean_θ*mean_sd)
        k1_0        = mean_y * (Pa_display**2) / (mean_θ * mean_sd)

        try:
            popt, _ = curve_fit(
                witczak_model, X, y,
                p0=[k1_0, 1.0, 1.0],
                maxfev=200000
            )
        except RuntimeError:
            st.error("❌ Não foi possível ajustar o modelo Witczak.")
            st.stop()

        # predição e métricas
        y_pred      = witczak_model(X, *popt)
        r2          = r2_score(y, y_pred)
        if len(y) > len(popt) + 1:
            raw       = adjusted_r2(r2, len(y), len(popt))
            r2_adj    = min(raw, r2, 1.0)
        else:
            r2_adj    = r2
        rmse        = np.sqrt(mean_squared_error(y, y_pred))
        mae         = mean_absolute_error(y, y_pred)

        k1, k2, k3  = popt
        eq_latex    = (
            f"$$MR = {k1:.4f}" +
            f" \cdot (θ^{{k2:.4f}}/{Pa_display:.6f})" +
            f" \cdot ((σ_d^{{k3:.4f}})/{Pa_display:.6f})$$"
        )
        intercept   = 0.0

        is_power     = True
        power_params = popt
        model_obj    = witczak_model
        poly_obj     = None

# — Modelo Pezo (normalizado ou não normalizado) —
    else:
        # Pezo Normalizado (com Pa)
        if pezo_option == "Normalizada":
            def pezo_model(X_flat, k1, k2, k3):
                Pa = 0.101325
                s3, sd = X_flat[:, 0], X_flat[:, 1]
                return k1 * Pa * (s3/Pa)**k2 * (sd/Pa)**k3

            mean_y      = y.mean()
            Pa_display  = 0.101325
            k1_0        = mean_y / (Pa_display * (X[:,0].mean()/Pa_display) * (X[:,1].mean()/Pa_display))

            try:
                popt, _ = curve_fit(pezo_model, X, y, p0=[k1_0, 1.0, 1.0], maxfev=200000)
            except RuntimeError:
                st.error("❌ Não foi possível ajustar o modelo Pezo.")
                st.stop()

            y_pred = pezo_model(X, *popt)
            r2     = r2_score(y, y_pred)
            if len(y) > len(popt) + 1:
                raw = adjusted_r2(r2, len(y), len(popt))
                r2_adj = min(raw, r2, 1.0)
            else:
                r2_adj = r2
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae  = mean_absolute_error(y, y_pred)

            k1, k2, k3 = popt
            eq_latex    = (
                f"$$MR = {k1:.4f}" +
                f" \cdot (θ^{{k2:.4f}}/{Pa_display:.6f})" +
                f" \cdot ((σ_d^{{k3:.4f}})/{Pa_display:.6f})$$"
            )
            intercept = 0.0

            is_power     = True
            power_params = popt
            model_obj    = pezo_model
            poly_obj     = None

        # Pezo Não Normalizado (direto σ₃^k2 · σd^k3)
        else:
            def pezo_model_nonnorm(X_flat, k1, k2, k3):
                s3, sd = X_flat[:, 0], X_flat[:, 1]
                return k1 * s3**k2 * sd**k3

            mean_y   = y.mean()
            mean_s3  = X[:, 0].mean()
            mean_sd  = X[:, 1].mean()
            k1_0     = mean_y / (mean_s3 * mean_sd)

            try:
                popt, _ = curve_fit(pezo_model_nonnorm, X, y, p0=[k1_0, 1.0, 1.0], maxfev=200000)
            except RuntimeError:
                st.error("❌ Não foi possível ajustar o modelo Pezo não normalizado.")
                st.stop()

            y_pred = pezo_model_nonnorm(X, *popt)
            r2     = r2_score(y, y_pred)
            if len(y) > len(popt) + 1:
                raw = adjusted_r2(r2, len(y), len(popt))
                r2_adj = min(raw, r2, 1.0)
            else:
                r2_adj = r2
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae  = mean_absolute_error(y, y_pred)

            k1, k2, k3 = popt
            eq_latex    = (
                f"$$MR = {k1:.4f}" +
                f" \cdot (θ^{{k2:.4f}}/{Pa_display:.6f})" +
                f" \cdot ((σ_d^{{k3:.4f}})/{Pa_display:.6f})$$"
            )

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    eq = eq_text.strip().strip("$$")
    p = doc.add_paragraph()
    i = 0
    while i < len(eq):
        ch = eq[i]
        if ch == '^':
            i += 1
            exp = ""
            while i < len(eq) and (eq[i].isdigit() or eq[i] in ['.', '-']):
                exp += eq[i]
                i += 1
            run = p.add_run(exp)
            run.font.superscript = True
        elif ch in ['_', '~']:
            i += 1
            if i < len(eq):
                run = p.add_run(eq[i])
                run.font.subscript = True
                i += 1
        elif ch == 'σ':
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
    MRg = model(Xg, *power_params) if is_power else model.predict(poly.transform(Xg))
    MRg = MRg.reshape(s3g.shape)
    fig = go.Figure(data=[go.Surface(x=s3g, y=sdg, z=MRg, colorscale='Viridis')])
    fig.add_trace(go.Scatter3d(x=df["σ3"], y=df["σd"], z=df[energy_col], mode='markers', marker=dict(size=5, color='red'), name="Dados"))
    fig.update_layout(scene=dict(xaxis_title='σ₃ (MPa)', yaxis_title='σ_d (MPa)', zaxis_title='MR (MPa)'), margin=dict(l=0, r=0, b=0, t=30))
    return fig


def interpret_metrics(r2, r2_adj, rmse, mae, y):
    txt = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
    txt += f"**R² Ajustado:** {r2_adj:.6f}\n\n"
    txt += f"**RMSE:** {rmse:.4f} MPa\n\n"
    txt += f"**MAE:** {mae:.4f} MPa\n\n"
    txt += f"**Média MR:** {y.mean():.4f} MPa\n\n"
    txt += f"**Desvio Padrão MR:** {y.std():.4f} MPa\n\n"
    return txt


def generate_word_doc(eq_latex, metrics_txt, quality_txt, fig, energy, degree, intercept, df):
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
    p = doc.add_paragraph()
    p.add_run("A função de MR é válida apenas para valores de 0,020≤")
    r1 = p.add_run("σ"); r1.font.subscript = False
    r2 = p.add_run("3"); r2.font.subscript = True
    p.add_run("≤0,14 e 0,02≤")
    r3 = p.add_run("σ"); r3.font.subscript = False
    r4 = p.add_run("d"); r4.font.subscript = True
    p.add_run("≤0,42 observada a norma DNIT 134/2018‑ME.")
    doc.add_heading("Avaliação da Qualidade do Ajuste", level=2)
    doc.add_paragraph(quality_txt)
    doc.add_page_break()
    add_data_table(doc, df)
    doc.add_heading("Gráfico 3D da Superfície", level=2)
    img = fig.to_image(format="png")
    doc.add_picture(BytesIO(img), width=Inches(6))
    buf = BytesIO()
    doc.save(buf)
    return buf


def generate_latex_doc(eq_latex, r2, r2_adj, rmse, mae, mean_MR, std_MR,
                       energy, degree, intercept, df, fig,
                       nrmse_range, qual_nrmse, cv_rmse, qual_cv, mae_pct, qual_mae):
    lines = []
    lines.append(r"\documentclass{article}")
    lines.append(r"\usepackage[utf8]{inputenc}")
    lines.append(r"\usepackage{booktabs,graphicx}")
    lines.append(r"\begin{document}")
    lines.append(r"\section*{Relatório de Regressão}")
    lines.append(r"\subsection*{Configurações}")
    lines.append(f"Tipo de energia: {energy}\\")
    if degree is not None:
        lines.append(f"Grau polinomial: {degree}\\")
    lines.append(r"\subsection*{Equação Ajustada}")
    lines.append(eq_latex)
    lines.append(r"\subsection*{Indicadores Estatísticos}")
    lines.append(r"\begin{itemize}")
    lines.append(f"  \\item \\textbf{{R$^2$}}: {r2:.6f} (aprox. {r2*100:.2f}\\% explicado)")
    lines.append(f"  \\item \\textbf{{R$^2$ Ajustado}}: {r2_adj:.6f}")
    lines.append(f"  \\item \\textbf{{RMSE}}: {rmse:.4f} MPa")
    lines.append(f"  \\item \\textbf{{MAE}}: {mae:.4f} MPa")
    lines.append(f"  \\item \\textbf{{Média MR}}: {mean_MR:.4f} MPa")
    lines.append(f"  \\item \\textbf{{Desvio Padrão MR}}: {std_MR:.4f} MPa")
    lines.append(r"\end{itemize}")
    lines.append(f"Intercepto: {intercept:.4f}\\")
    lines.append(r"\section*{Avaliação da Qualidade do Ajuste}")
    lines.append(r"\begin{itemize}")
    lines.append(f"  \\item NRMSE_range: {nrmse_range:.2%} → {qual_nrmse}")
    lines.append(f"  \\item CV(RMSE): {cv_rmse:.2%} → {qual_cv}")
    lines.append(f"  \\item MAE \%: {mae_pct:.2%} → {qual_mae}")
    lines.append(r"\end{itemize}")
    lines.append(r"\newpage")
    cols = len(df.columns)
    lines.append(r"\section*{Dados do Ensaio Triaxial}")
    lines.append(r"\begin{tabular}{" + "l" * cols + r"}")
    lines.append(" & ".join(df.columns) + r" \\ \midrule")
    for _, row in df.iterrows():
        vals = [str(v) for v in row.values]
        lines.append(" & ".join(vals) + r" \")
    lines.append(r"\end{tabular}")
    lines.append(r"\section*{Gráfico 3D da Superfície}")
    lines.append(r"\includegraphics[width=\linewidth]{surface_plot.png}")
    lines.append(r"\end{document}")
    img_data = fig.to_image(format="png")
    tex_content = "\n".join(lines)
    return tex_content, img_data

# --- Streamlit App ---

st.set_page_config(page_title="Modelos de MR", layout="wide")
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
model_type = st.sidebar.selectbox("Escolha o modelo de regressão",["Polinomial c/ Intercepto","Polinomial s/Intercepto","Potência Composta","Pezo"])

degree = None
if model_type.startswith("Polinomial"):
    degree = st.sidebar.selectbox("Grau (polinomial)",[2,3,4,5,6],index=0)
energy = st.sidebar.selectbox("Energia",["Normal","Intermediária","Modificada"],index=0)

if st.button("Calcular"):
    X = df[["σ3", "σd"]].values
    y = df["MR"].values
    # --- ajuste dos modelos (inalterado) ---
    # ... código omitido para brevidade ...
    # cálculo de métricas
    r2 = r2_score(y, y_pred)
    # ... demais métricas calculadas ...
    metrics_txt = interpret_metrics(r2, r2_adj, rmse, mae, y)

    # qualidade do ajuste
    amp = y.max() - y.min()
    mr_mean = y.mean()
    nrmse_range = rmse / amp if amp > 0 else np.nan
    cv_rmse = rmse / mr_mean if mr_mean != 0 else np.nan
    mae_pct = mae / mr_mean if mr_mean != 0 else np.nan
    labels_nrmse = ["Excelente (≤5%)","Bom (≤10%)","Insuficiente (>10%)"]
    labels_cv = ["Excelente (≤10%)","Bom (≤20%)","Insuficiente (>20%)"]
    def quality_label(val, thresholds, labels):
        for t, lab in zip(thresholds, labels):
            if val <= t:
                return lab
        return labels[-1]
    qual_nrmse = quality_label(nrmse_range, [0.05,0.10], labels_nrmse)
    qual_cv = quality_label(cv_rmse, [0.10,0.20], labels_cv)
    qual_mae = quality_label(mae_pct, [0.10,0.20], labels_cv)
    quality_txt = (
        f"**NRMSE_range:** {nrmse_range:.2%} → {qual_nrmse}\n\n"
        f"**CV(RMSE):** {cv_rmse:.2%} → {qual_cv}\n\n"
        f"**MAE %:** {mae_pct:.2%} → {qual_mae}\n\n"
    )

    # exibição no app
    st.write("### Equação Ajustada")
    st.latex(eq_latex.strip("$$"))
    st.write("### Indicadores Estatísticos")
    for name, val, tip in [
        ("R²", f"{r2:.6f}", f"Explica ~{r2*100:.2f}%"),
        ("R² Ajustado", f"{r2_adj:.6f}", "Penaliza termos extras"),
        ("RMSE", f"{rmse:.4f} MPa", "Erro quadrático médio"),
        ("MAE", f"{mae:.4f} MPa", "Erro absoluto médio"),
        ("Média MR", f"{y.mean():.4f} MPa", "Média observada"),
        ("Desvio Padrão MR", f"{y.std():.4f} MPa", "Dispersion")
    ]:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)
    st.write(f"**Intercepto:** {intercept:.4f}")
    st.markdown("A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e 0,02≤σ_d≤0,42 (DNIT 134/2018‑ME)", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    st.markdown(f"- **NRMSE_range:** {nrmse_range:.2%} → {qual_nrmse}", unsafe_allow_html=True)
    st.markdown(f"- **CV(RMSE):** {cv_rmse:.2%} → {qual_cv}", unsafe_allow_html=True)
    st.markdown(f"- **MAE %:** {mae_pct:.2%} → {qual_mae}", unsafe_allow_html=True)

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(fig, use_container_width=True)

    # gera buffers para download
    tex_content, img_data = generate_latex_doc(
        eq_latex, r2, r2_adj, rmse, mae, y.mean(), y.std(),
        energy, degree, intercept, df, fig,
        nrmse_range, qual_nrmse, cv_rmse, qual_cv, mae_pct, qual_mae
    )
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        zf.writestr("Relatorio_Regressao.tex", tex_content)
        zf.writestr("surface_plot.png", img_data)
    zip_buf.seek(0)
    st.session_state["tex_buf"] = zip_buf
    try:
        import pypandoc; pypandoc.download_pandoc('latest')
        docx_bytes = pypandoc.convert_text(tex_content, 'docx', format='latex')
        st.session_state["docx_bytes"] = docx_bytes
    except Exception:
        buf = generate_word_doc(eq_latex, metrics_txt, quality_txt, fig, energy, degree, intercept, df)
        buf.seek(0)
        st.session_state["word_buf"] = buf

# botões de download persistentes
if "tex_buf" in st.session_state:
    st.download_button(
        "Salvar LaTex",
        data=st.session_state["tex_buf"],
        file_name="Relatorio_Regressao.zip",
        mime="application/zip"
    )
if "docx_bytes" in st.session_state:
    st.download_button(
        "Gerar Word",
        data=st.session_state["docx_bytes"],
        file_name="Relatorio_Regressao.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
elif "word_buf" in st.session_state:
    st.download_button(
        "Gerar Word",
        data=st.session_state["word_buf"],
        file_name="Relatorio_Regressao.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


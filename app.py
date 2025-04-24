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
    doc = Document()
    doc.add_heading("Relatório de Regressão", level=1)
    doc.add_heading("Configurações", level=2)
    doc.add_paragraph(f"Tipo de energia: {energy}")
    if degree is not None:
        doc.add_paragraph(f"Grau polinomial: {degree}")
        doc.add_heading("Equação Ajustada", level=2)
    raw_eq   = eq_latex.strip("$$")
    # Ajuste sintaxe de expoentes para formatação correta
    raw_eq = raw_eq.replace("^{", "^").replace("}", "")
    eq_lines = [ln.strip() for ln in raw_eq.split("\\\\")]
    for ln in eq_lines:
        # transforma σ₃ → σ_3 e σd → σ_d para garantir que add_formatted_equation
        # pegue o '_' e aplique subescrito tanto em '3' quanto em 'd'
        ln = ln.replace("σ₃", "σ_3").replace("σd", "σ_d")
        add_formatted_equation(doc, ln)

    #add_formatted_equation(doc, eq_latex)
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
    p.add_run("≤0,42 observada a norma DNIT 134/2018‑ME e a precisão do equipamento.")
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
        # Primeiro termo
        eq += f"{terms[0][0]:.4f}{terms[0][1]}"
        # Demais termos
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

    # — Modelo Pezo —
    else:
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
        const = k1 * Pa_display
        eq_latex = (
            f"$$MR = {const:.4f}(σ₃/{Pa_display:.6f})^{{{k2:.4f}}}(σ_d/{Pa_display:.6f})^{{{k3:.4f}}}$$"
        )
        intercept = 0.0

        is_power     = True
        power_params = popt
        model_obj    = pezo_model
        poly_obj     = None

    # --- Saída e Relatório ---
        # Validação do ajuste: impede R² negativo
        if np.isnan(r2) or r2 < 0:
            st.error(f"❌ O modelo não convergiu adequadamente (R² = {r2:.4f}).")
            st.stop()
    metrics_txt = interpret_metrics(r2, r2_adj, rmse, mae, y)
    fig = plot_3d_surface(df, model_obj, poly_obj, "MR", is_power=is_power, power_params=power_params)

    st.write("### Equação Ajustada")
    st.latex(eq_latex.strip("$$"))

    st.write("### Indicadores Estatísticos")
    mean_MR = y.mean()
    std_MR  = y.std()
    indicators = [
        ("R²", f"{r2:.6f}", f"Este valor indica que aproximadamente {r2*100:.2f}% da variabilidade dos dados de MR é explicada pelo modelo."),
        ("R² Ajustado", f"{r2_adj:.6f}", "Essa métrica penaliza o uso excessivo de termos."),
        ("RMSE", f"{rmse:.4f} MPa", f"Erro quadrático médio: {rmse:.4f} MPa."),
        ("MAE", f"{mae:.4f} MPa", f"Erro absoluto médio: {mae:.4f} MPa."),
        ("Média MR", f"{mean_MR:.4f} MPa", "Média dos valores observados."),
        ("Desvio Padrão MR", f"{std_MR:.4f} MPa", "Dispersão dos dados em torno da média."),
    ]
    for name, val, tip in indicators:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    # Informações de amplitude e valores extremos
    amplitude = np.max(y) - np.min(y)
    max_mr = np.max(y)
    min_mr = np.min(y)
    st.markdown(f"**Amplitude:** {amplitude:.4f} MPa <span title='Diferença entre valor máximo e mínimo observados de MR.'>ℹ️</span>", unsafe_allow_html=True)
    st.markdown(f"**MR Máximo:** {max_mr:.4f} MPa")
    st.markdown(f"**MR Mínimo:** {min_mr:.4f} MPa")

    st.write(f"**Intercepto:** {intercept:.4f}")
    st.markdown(
        "A função de MR é válida apenas para valores de 0,02≤σ₃≤0,14 e 0,02≤σ_d≤0,42 observada a norma DNIT 134/2018‑ME e a precisão do equipamento.",
        unsafe_allow_html=True
    )

    # Métricas normalizadas
    amp        = y.max() - y.min()
    mr_mean    = y.mean()
    nrmse_range = rmse / amp if amp > 0 else np.nan
    cv_rmse     = rmse / mr_mean if mr_mean != 0 else np.nan
    mae_pct     = mae  / mr_mean if mr_mean  != 0 else np.nan

    def quality_label(val, thresholds, labels):
        for t, lab in zip(thresholds, labels):
            if val <= t:
                return lab
        return labels[-1]

    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]

    qual_nrmse = quality_label(nrmse_range, [0.05, 0.10], labels_nrmse)
    qual_cv     = quality_label(cv_rmse,     [0.10, 0.20], labels_cv)
    qual_mae    = quality_label(mae_pct,     [0.10, 0.20], labels_cv)

    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    st.markdown(
        f"- **NRMSE:** {nrmse_range:.2%} → {qual_nrmse} <span title=\"NRMSE: RMSE normalizado pela amplitude dos valores de MR; indicador associado ao RMSE.\">ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **CV(RMSE):** {cv_rmse:.2%} → {qual_cv} <span title=\"CV(RMSE): coeficiente de variação do RMSE (RMSE/média MR); indicador associado ao RMSE.\">ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **MAE %:** {mae_pct:.2%} → {qual_mae} <span title=\"MAE %: MAE dividido pela média de MR; indicador associado ao MAE.\">ℹ️</span>",
        unsafe_allow_html=True
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(fig, use_container_width=True)

    # Downloads LaTeX com gráfico e Word
    tex_content, img_data = generate_latex_doc(
        eq_latex, r2, r2_adj, rmse, mae,
        mean_MR, std_MR, energy, degree,
        intercept, df, fig
    )
    # cria um ZIP com .tex e imagem
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w") as zf:
        # Overleaf abre automaticamente o main.tex
        zf.writestr("main.tex", tex_content)
        zf.writestr("surface_plot.png", img_data)
    zip_buf.seek(0)
    st.download_button(
        "Salvar LaTeX",
        data=zip_buf,
        file_name="Relatorio_Regressao.zip",
        mime="application/zip"
    )

    try:
        import pypandoc
        pypandoc.download_pandoc('latest')
        docx_bytes = pypandoc.convert_text(tex_content, 'docx', format='latex')
        st.download_button(
            "Converter: Word (OMML)",
            data=docx_bytes,
            file_name="Relatorio_Regressao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception:
        buf = generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df)
        buf.seek(0)
        st.download_button(
        "Converter: Word",
            data=buf,
            file_name="Relatorio_Regressao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


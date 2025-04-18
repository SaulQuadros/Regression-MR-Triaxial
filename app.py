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
    eq = eq_text.strip().strip("$$")
    p = doc.add_paragraph()
    i = 0
    while i < len(eq):
        # \sigma_{...}
        if eq.startswith('\\sigma', i):
            p.add_run('σ')
            i += len('\\sigma')
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
                r = p.add_run(sub)
                r.font.subscript = True
            continue
        # Superscript ^{...}
        if eq[i] == '^':
            i += 1
            exp = ''
            if i < len(eq) and eq[i] == '{':
                i += 1
                while i < len(eq) and eq[i] != '}':
                    exp += eq[i]
                    i += 1
                i += 1
            else:
                while i < len(eq) and (eq[i].isdigit() or eq[i] == '.'):
                    exp += eq[i]
                    i += 1
            r = p.add_run(exp)
            r.font.superscript = True
            continue
        # Legacy subscript ~
        if eq[i] == '~':
            i += 1
            if i < len(eq):
                r = p.add_run(eq[i])
                r.font.subscript = True
                i += 1
            continue
        # Default char
        p.add_run(eq[i])
        i += 1
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

# ... (restante do código permanece inalterado) ...

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
    # Parágrafo com sigmas corretamente subscritos
    p = doc.add_paragraph()
    p.add_run("A função de MR é válida apenas para valores de 0,020≤")
    r1 = p.add_run("σ")
    r2 = p.add_run("3")
    r2.font.subscript = True
    p.add_run("≤0,14 e 0,02≤")
    r3 = p.add_run("σ")
    r4 = p.add_run("d")
    r4.font.subscript = True
    p.add_run("≤0,42 observada a norma DNIT 134/2018‑ME.")

    doc.add_page_break()
    add_data_table(doc, df)
    doc.add_heading("Gráfico 3D da Superfície", level=2)
    img = fig.to_image(format="png")
    doc.add_picture(BytesIO(img), width=Inches(6))
    buf = BytesIO(); doc.save(buf)
    return buf

def generate_latex_doc(eq_latex, r2, r2_adj, rmse, mae, mean_MR, std_MR, energy, degree, intercept, df, fig):
    # Monta o documento LaTeX
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
    lines.append(r"\subsection*{Indicadores Estatísticos}")
    lines.append(r"\begin{itemize}")
    lines.append(f"  \\item \\textbf{{R$^2$}}: {r2:.6f} (aprox. {r2*100:.2f}\\% explicado)")
    lines.append(f"  \\item \\textbf{{R$^2$ Ajustado}}: {r2_adj:.6f}")
    lines.append(f"  \\item \\textbf{{RMSE}}: {rmse:.4f} MPa")
    lines.append(f"  \\item \\textbf{{MAE}}: {mae:.4f} MPa")
    lines.append(f"  \\item \\textbf{{Média MR}}: {mean_MR:.4f} MPa")
    lines.append(f"  \\item \\textbf{{Desvio Padrão MR}}: {std_MR:.4f} MPa")
    lines.append(r"\end{itemize}")
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
    img_data = fig.to_image(format="png")
    with open("surface_plot.png", "wb") as f:
        f.write(img_data)
    lines.append(r"\section*{Gráfico 3D da Superfície}")
    lines.append(r"\includegraphics[width=\linewidth]{surface_plot.png}")
    lines.append(r"\end{document}")
    return "\n".join(lines)

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
    "Escolha o modelo de regressão",
    [
        "Polinomial c/ Intercepto",
        "Polinomial s/Intercepto",
        "Potência Composta c/Intercepto",
        "Potência Composta s/Intercepto",
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
    X = df[["σ3", "σd"]].values
    y = df["MR"].values

    # — Polinomial —
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

    # — Potência Composta —
    elif model_type in ("Potência Composta c/Intercepto", "Potência Composta s/Intercepto"):
        def pot_with_int(X_flat, a0, a1, k1, a2, k2, a3, k3):
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return a0 + a1 * s3**k1 + a2 * (s3 * sd)**k2 + a3 * sd**k3

        def pot_no_int(X_flat, a1, k1, a2, k2, a3, k3):
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return a1 * s3**k1 + a2 * (s3 * sd)**k2 + a3 * sd**k3

        mean_y = y.mean()
        mean_s3 = X[:,0].mean()
        mean_sd = X[:,1].mean()
        mean_s3sd = (X[:,0]*X[:,1]).mean()
        p0_with = [
            mean_y,
            mean_y/mean_s3, 1,
            mean_y/mean_s3sd, 1,
            mean_y/mean_sd, 1
        ]
        p0_no = [
            mean_y/mean_s3, 1,
            mean_y/mean_s3sd, 1,
            mean_y/mean_sd, 1
        ]

        if model_type == "Potência Composta c/Intercepto":
            fit_func = pot_with_int
            p0 = p0_with
        else:
            fit_func = pot_no_int
            p0 = p0_no

        try:
            popt, _ = curve_fit(fit_func, X, y, p0=p0, maxfev=200000)
        except RuntimeError:
            st.error(
                "❌ Não foi possível ajustar o modelo de Potência Composta. "
                "Verifique seus dados ou tente outro modelo."
            )
            st.stop()

        y_pred = fit_func(X, *popt)
        r2 = r2_score(y, y_pred)
        if len(y) > len(popt) + 1:
            raw = adjusted_r2(r2, len(y), len(popt))
            r2_adj = min(raw, r2, 1.0)
        else:
            r2_adj = r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        if model_type == "Potência Composta s/Intercepto":
            a1, k1, a2, k2, a3, k3 = popt
            eq_latex = (
                f"$$MR = {a1:.4f}\\sigma_3^{{{k1:.4f}}} "
                f"+ {a2:.4f}(\\sigma_3\\sigma_d)^{{{k2:.4f}}} "
                f"+ {a3:.4f}\\sigma_d^{{{k3:.4f}}}$$"
            )
            intercept = 0.0
        else:
            a0, a1, k1, a2, k2, a3, k3 = popt
            eq_latex = (
                f"$$MR = {a0:.4f} + {a1:.4f}\\sigma_3^{{{k1:.4f}}} "
                f"+ {a2:.4f}(\\sigma_3\\sigma_d)^{{{k2:.4f}}} "
                f"+ {a3:.4f}\\sigma_d^{{{k3:.4f}}}$$"
            )
            intercept = a0

        is_power = True
        power_params = popt
        model_obj = fit_func
        poly = None

    # — Pezo —
    else:
        def pezo_model(X_flat, k1, k2, k3):
            Pa = 0.101325
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return k1 * Pa * (s3/Pa)**k2 * (sd/Pa)**k3

        mean_y = y.mean()
        mean_s3 = X[:,0].mean()
        mean_sd = X[:,1].mean()
        Pa_display = 0.101325
        k1_0 = mean_y / (Pa_display * (mean_s3/Pa_display)**1 * (mean_sd/Pa_display)**1)
        p0 = [k1_0, 1.0, 1.0]

        try:
            popt, _ = curve_fit(pezo_model, X, y, p0=p0, maxfev=200000)
        except RuntimeError:
            st.error(
                "❌ Não foi possível ajustar o modelo Pezo. "
                "Verifique seus dados ou tente outro modelo."
            )
            st.stop()

        y_pred = pezo_model(X, *popt)
        r2 = r2_score(y, y_pred)
        if len(y) > len(popt) + 1:
            raw = adjusted_r2(r2, len(y), len(popt))
            r2_adj = min(raw, r2, 1.0)
        else:
            r2_adj = r2

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        k1, k2, k3 = popt
        const = k1 * Pa_display
        eq_latex = (
            f"$$MR = {const:.4f}("
            f"\\sigma_3/{Pa_display:.6f})^{{{k2:.4f}}}"
            f"(\\sigma_d/{Pa_display:.6f})^{{{k3:.4f}}}$$"
        )
        intercept = 0.0

        is_power = True
        power_params = popt
        model_obj = pezo_model
        poly = None

    # — Saída dos Resultados —
    metrics_txt = interpret_metrics(r2, r2_adj, rmse, mae, y)
    fig = plot_3d_surface(df, model_obj, poly, "MR",
                          is_power=is_power, power_params=power_params)

    st.write("### Equação Ajustada")
    st.latex(eq_latex.strip("$$"))

    st.write("### Indicadores Estatísticos")
    mean_MR = y.mean()
    std_MR = y.std()
    indicators = [
        ("R²", f"{r2:.6f}", f"Este valor indica que aproximadamente {r2*100:.2f}% da variabilidade dos dados de MR é explicada pelo modelo."),
        ("R² Ajustado", f"{r2_adj:.6f}", "Essa métrica penaliza o uso excessivo de termos. A alta similaridade com o R² indica ausência de superajuste."),
        ("RMSE", f"{rmse:.4f} MPa", f"Em média, a previsão difere dos valores observados por {rmse:.4f} MPa (sensível a erros grandes)."),
        ("MAE", f"{mae:.4f} MPa", f"Em média, o erro absoluto entre o valor previsto e o real é de {mae:.4f} MPa."),
        ("Média MR", f"{mean_MR:.4f} MPa", "Essa é a média dos valores observados."),
        ("Desvio Padrão MR", f"{std_MR:.4f} MPa", "Esse valor representa a dispersão dos dados em torno da média."),
    ]
    for name, val, tip in indicators:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    st.write(f"**Intercepto:** {intercept:.4f}")
    st.markdown(
        "A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e "
        "0,02≤$\\sigma_{d}$≤0,42 observada a norma DNIT 134/2018‑ME.",
        unsafe_allow_html=True
    )

    # amplitude e média
    mr_min, mr_max = y.min(), y.max()
    amp = mr_max - mr_min
    mr_mean = y.mean()

    # métricas normalizadas
    nrmse_range = rmse / amp if amp > 0 else np.nan
    cv_rmse    = rmse / mr_mean if mr_mean != 0 else np.nan
    mae_pct    = mae  / mr_mean if mr_mean != 0 else np.nan

    # avaliação de qualidade
    def quality_label(val, thresholds, labels):
        for t, lab in zip(thresholds, labels):
            if val <= t:
                return lab
        return labels[-1]

    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]

    qual_nrmse = quality_label(nrmse_range, [0.05, 0.10], labels_nrmse)
    qual_cv     = quality_label(cv_rmse,    [0.10, 0.20], labels_cv)
    qual_mae    = quality_label(mae_pct,    [0.10, 0.20], labels_cv)

    st.write("---")
    st.subheader("Avaliação da Qualidade do Ajuste")
    st.markdown(
        f"- **NRMSE_range:** {nrmse_range:.2%} → {qual_nrmse} <span title=\"NRMSE_range: RMSE normalizado pela amplitude dos valores de MR; indicador associado ao RMSE.\">ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **CV(RMSE):** {cv_rmse:.2%} → {qual_cv} <span title=\"CV(RMSE): coeficiente de variação do RMSE (RMSE/média MR); indicador associado ao RMSE.\">ℹ️</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"- **MAE %:** {mae_pct:.2%} → {qual_mae} <span title=\"MAE %: MAE dividido pela média de MR, expressando o erro absoluto médio em porcentagem; indicador associado ao MAE.\">ℹ️</span>",
        unsafe_allow_html=True
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(fig, use_container_width=True)

    # --- Geração de LaTeX e download imediato em .tex e .docx ---
    tex_content = generate_latex_doc(
        eq_latex, r2, r2_adj, rmse, mae, mean_MR, std_MR,
        energy, degree, intercept, df, fig
    )

    st.download_button(
        "Salvar LaTeX",
        data=tex_content,
        file_name="Relatorio_Regressao.tex",
        mime="text/x-tex"
    )

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


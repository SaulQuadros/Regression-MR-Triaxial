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
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from docx import Document
from docx.shared import Inches, Pt

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
    p = doc.add_paragraph(style="List Paragraph")
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
    # malhas 1D para garantir semântica clara (Plotly: z.shape == (len(y), len(x)))
    s3 = np.linspace(df["σ3"].min(), df["σ3"].max(), 30)
    sd = np.linspace(df["σd"].min(), df["σd"].max(), 30)
    s3g, sdg = np.meshgrid(s3, sd, indexing="xy")  # shapes: (len(sd), len(s3))
    Xg = np.c_[s3g.ravel(order="C"), sdg.ravel(order="C")]
    MRg = (model(Xg, *power_params) if is_power else model.predict(poly.transform(Xg)))
    MRg = MRg.reshape(s3g.shape, order="C")  # => (len(sd), len(s3))

    # Use x=s3 (1D), y=sd (1D), z=MRg (2D) para evitar ambiguidades
    fig = go.Figure(data=[go.Surface(x=s3, y=sd, z=MRg, colorscale='Viridis')])

    # Pontos observados
    fig.add_trace(go.Scatter3d(
        x=df["σ3"], y=df["σd"], z=df[energy_col],
        mode='markers', marker=dict(size=5, color='red'), name="Dados"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='σ₃ (MPa)',
            yaxis_title='σ_d (MPa)',
            zaxis_title='MR (MPa)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig


def interpret_metrics(r2, r2_adj, rmse, mae, y):
    """Gera texto para relatório Word."""
    txt = f"R² = {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
    txt += f"R² Ajustado = {r2_adj:.6f}\n\n"
    txt += f"RMSE = {rmse:.4f} MPa\n\n"
    txt += f"MAE = {mae:.4f} MPa\n\n"
    txt += f"Média MR = {y.mean():.4f} MPa\n\n"
    txt += f"Desvio Padrão MR = {y.std():.4f} MPa\n\n"
    return txt






def _camera_to_view_init(camera_eye):
    """Map Plotly camera eye to Matplotlib elev/azim angles."""
    try:
        x = float(camera_eye.get("x", 1.5))
        y = float(camera_eye.get("y", 1.5))
        z = float(camera_eye.get("z", 1.0))
        azim = math.degrees(math.atan2(y, x))
        rxy = (x**2 + y**2) ** 0.5
        elev = math.degrees(math.atan2(z, rxy))
        return elev, azim
    except Exception:
        return 30, -60  # Matplotlib defaults

def export_plotly_figure_png(fig, azim_offset_deg=0.0, elev_offset_deg=0.0):
    """Export Plotly 3D surface to PNG using Matplotlib only.
    This version guarantees that the X axis in Matplotlib corresponds to scene.xaxis (Plotly)
    and Y to scene.yaxis, by checking Z shape and transposing when necessary.
    """
    import io, math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.gridspec import GridSpec

    # ----- surface trace -----
    surface_trace = None
    for tr in fig.data:
        if getattr(tr, "type", "") == "surface":
            surface_trace = tr
            break
    if surface_trace is None:
        raise RuntimeError("Figura não contém trace 'surface' para exportação.")

    # Raw arrays
    Z_raw = np.array(surface_trace.z, dtype=float)
    X_raw = surface_trace.x
    Y_raw = surface_trace.y

    # If x/y None, build default 1D arrays
    if X_raw is None or Y_raw is None:
        ny, nx = Z_raw.shape
        X_raw = np.arange(nx)
        Y_raw = np.arange(ny)

    X_raw = np.array(X_raw, dtype=float)
    Y_raw = np.array(Y_raw, dtype=float)

    # Ensure 2D mesh consistent with Plotly semantics:
    # Plotly docs: if x,y are 1D, z must be shape (len(y), len(x)).
    # We enforce that; if z is (len(x), len(y)), we transpose to (len(y), len(x)).
    if X_raw.ndim == 1 and Y_raw.ndim == 1:
        nx = len(X_raw); ny = len(Y_raw)
        if Z_raw.shape == (ny, nx):
            Z = Z_raw
        elif Z_raw.shape == (nx, ny):
            Z = Z_raw.T  # fix orientation to match x=columns, y=rows
        else:
            # Fallback: try to reshape if sizes match
            if Z_raw.size == nx * ny:
                Z = Z_raw.reshape(ny, nx)
            else:
                raise RuntimeError(f"Incompatibilidade entre shapes: Z={Z_raw.shape}, x={nx}, y={ny}")
        Xg, Yg = np.meshgrid(X_raw, Y_raw)  # mesh with shape (ny, nx)
    else:
        # x/y already come as 2D grids
        Xg = np.array(X_raw, dtype=float)
        Yg = np.array(Y_raw, dtype=float)
        Z = Z_raw
        if not (Z.shape == Xg.shape == Yg.shape):
            raise RuntimeError(f"Shapes incompatíveis (esperado iguais): Z={Z.shape}, X={Xg.shape}, Y={Yg.shape}")

    vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))

    # ----- layout with dedicated colorbar axes -----
    fig_m = plt.figure(figsize=(7.0, 5.2), dpi=220, constrained_layout=False)
    gs = GridSpec(1, 20, figure=fig_m)
    ax = fig_m.add_subplot(gs[0, :18], projection="3d")
    cax = fig_m.add_subplot(gs[0, 19])

    surf = ax.plot_surface(Xg, Yg, Z, cmap="viridis", linewidth=0, antialiased=True,
                           shade=False, vmin=vmin, vmax=vmax)

    # Scatter points
    for tr in fig.data:
        if getattr(tr, "type", "") == "scatter3d" and "markers" in (getattr(tr, "mode", "") or ""):
            xs = np.array(tr.x, dtype=float)
            ys = np.array(tr.y, dtype=float)
            zs = np.array(tr.z, dtype=float)
            ms = float(getattr(tr.marker, "size", 5) or 5)
            ax.scatter(xs, ys, zs, s=ms*8, c="red", depthshade=False)

    # ----- axis labels and ranges -----
    scene = getattr(fig.layout, "scene", None) or {}
    def _title(axobj, default):
        try:
            return axobj.title.text if getattr(axobj, "title", None) else default
        except Exception:
            return default
    def _rng(axobj):
        try:
            return axobj.range
        except Exception:
            return None

    xlabel = _title(getattr(scene, "xaxis", {}), "x")
    ylabel = _title(getattr(scene, "yaxis", {}), "y")
    zlabel = _title(getattr(scene, "zaxis", {}), "z")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)

    xr = _rng(getattr(scene, "xaxis", {}))
    yr = _rng(getattr(scene, "yaxis", {}))
    zr = _rng(getattr(scene, "zaxis", {}))
    if xr: ax.set_xlim(xr[0], xr[1])
    if yr: ax.set_ylim(yr[0], yr[1])
    if zr: ax.set_zlim(zr[0], zr[1])

    # ----- camera mapping (tuned) -----
    try:
        cam = getattr(scene, "camera", {}) or {}
        eye = getattr(cam, "eye", {}) or {}
        ex = float(getattr(eye, "x", 1.5)); ey = float(getattr(eye, "y", 1.5)); ez = float(getattr(eye, "z", 1.0))
        az_plotly = math.degrees(math.atan2(ey, ex))
        az_mpl = (180.0 - az_plotly)
        r = (ex**2 + ey**2 + ez**2) ** 0.5
        elev = math.degrees(math.asin(ez / r))
        # offsets from session_state if available
        try:
            import streamlit as st
            az_mpl += float(st.session_state.get("azim_offset", 0.0))
            elev += float(st.session_state.get("elev_offset", 0.0))
        except Exception:
            pass
        ax.view_init(elev=elev, azim=az_mpl)
    except Exception:
        ax.view_init(elev=30, azim=-60)

    cb = fig_m.colorbar(surf, cax=cax)
    cb.set_label(zlabel)

    out = io.BytesIO()
    fig_m.savefig(out, format="png", bbox_inches="tight", dpi=220)
    plt.close(fig_m)
    out.seek(0)
    return out.getvalue()


def generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df, model_type, pezo_option=None):
    from io import BytesIO
    from docx.shared import Inches, Pt
    import re

    doc = Document()
    # Ajuste: Define fonte 12pt para estilos principais
    for style_name in ['Normal', 'List Paragraph', 'Heading 1', 'Heading 2']:
        doc.styles[style_name].font.size = Pt(12)

    doc.add_heading("Relatório de Regressão", level=1)
    doc.add_heading("Configurações", level=2)
    doc.add_paragraph(f"Modelo de regressão: {model_type}", style="List Paragraph")
    if model_type.startswith("Polinomial"):
        doc.add_paragraph(f"Grau polinomial: {degree}", style="List Paragraph")
    elif model_type == "Pezo":
        doc.add_paragraph(f"Pezo – {pezo_option}", style="List Paragraph")
    doc.add_paragraph(f"Tipo de energia: {energy}", style="List Paragraph")
    if degree is not None:
        doc.add_paragraph(f"Grau polinomial: {degree}", style="List Paragraph")

    doc.add_heading("Equação Ajustada", level=2)
    # Exibe a equação completa sem quebras forçadas
    add_formatted_equation(doc, eq_latex.strip("$$"))

    # Indicadores Estatísticos
    doc.add_heading("Indicadores Estatísticos", level=2)
    # Cada indicador em parágrafo estilo List Paragraph
    for line in metrics_txt.strip().split("\n\n"):
        doc.add_paragraph(line, style="List Paragraph")
    # Cálculo de amplitude e extremos
    amplitude = float(df["MR"].max() - df["MR"].min())
    max_mr = float(df["MR"].max())
    min_mr = float(df["MR"].min())

    # Parse RMSE e MAE do metrics_txt
    rmse_match = re.search(r"\*\*RMSE=\*\*\s*([0-9]+(?:\.[0-9]+)?)", metrics_txt)
    mae_match = re.search(r"\*\*MAE:\*\*\s*([0-9]+(?:\.[0-9]+)?)", metrics_txt)
    rmse_val = float(rmse_match.group(1)) if rmse_match else float("nan")
    mae_val = float(mae_match.group(1)) if mae_match else float("nan")

    params = [
        ("Amplitude", f"{amplitude:.4f} MPa"),
        ("MR Máximo", f"{max_mr:.4f} MPa"),
        ("MR Mínimo", f"{min_mr:.4f} MPa"),
        ("Intercepto", f"{intercept:.4f} MPa")
    ]
    for name, val in params:
        doc.add_paragraph(f"{name} = {val}", style="List Paragraph")

    # Avaliação da Qualidade do Ajuste
    doc.add_heading("Avaliação da Qualidade do Ajuste", level=2)
    # Categorias de qualidade conforme interface
    def quality_label(val, thresholds, labels):
        for t, lab in zip(thresholds, labels):
            if val <= t:
                return lab
        return labels[-1]

    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]
    # Cálculo dos indicadores percentuais e interpretação
    qual_nrmse = quality_label(nrmse_range, [0.05, 0.10], labels_nrmse)
    qual_cv    = quality_label(cv_rmse,     [0.10, 0.20], labels_cv)
    qual_mae   = quality_label(mae_pct,     [0.10, 0.20], labels_cv)

    metrics_quality = [
        ("NRMSE", f"{nrmse_range:.2%}", qual_nrmse),
        ("CV(RMSE)", f"{cv_rmse:.2%}", qual_cv),
        ("MAE %", f"{mae_pct:.2%}", qual_mae)
    ]
    for name, val, cat in metrics_quality:
        doc.add_paragraph(f"{name} = {val} → {cat}", style="List Paragraph")
    
    # Inicia tabela na segunda página
    doc.add_page_break()
    add_data_table(doc, df)
    doc.add_heading("Gráfico 3D da Superfície", level=2)
    try:
        img = export_plotly_figure_png(fig, azim_offset_deg=st.session_state.get('azim_offset', 0.0), elev_offset_deg=st.session_state.get('elev_offset', 0.0))
        doc.add_picture(BytesIO(img), width=Inches(6))
    except Exception as e:
        doc.add_paragraph(f"Gráfico 3D não disponível: {e}")
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
    lines.append(f"Tipo de energia: {energy}\\")
    if degree is not None:
        lines.append(f"Grau polinomial: {degree}\\")
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
    lines.append(f"  \\item \\textbf{{NRMSE_range}}: {nrmse_range:.2%}")
    lines.append(f"  \\item \\textbf{{CV(RMSE)}}: {cv_rmse:.2%}")
    lines.append(f"  \\item \\textbf{{MAE \\%}}: {mae_pct:.2%}")
    lines.append(r"\end{itemize}")

    lines.append(f"Intercepto: {intercept:.4f}\\")
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
    lines.append(r"\includegraphics[width=\\linewidth]{surface_plot.png}")
    lines.append(r"\end{document}")

    # gera bytes da figura (fallback Matplotlib)
try:
    img_data = export_plotly_figure_png(fig, azim_offset_deg=st.session_state.get('azim_offset', 0.0),
                                        elev_offset_deg=st.session_state.get('elev_offset', 0.0))
except Exception:
    img_data = None

tex_content = "
".join(lines)
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
pezo_option = None  # inicializa pezo_option para evitar erro quando não for Pezo
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
        eq_latex = f"$$MR = {k1:.4f} (θ^{{{k2:.4f}}}/{Pa_display:.6f}) (σ_d^{{{k3:.4f}}}/{Pa_display:.6f})$$"
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
            const = k1 * Pa_display
            eq_latex = (
                f"$$MR = {const:.4f}(σ₃/{Pa_display:.6f})^{{{k2:.4f}}}(σ_d/{Pa_display:.6f})^{{{k3:.4f}}}$$"
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
            eq_latex = f"$$MR = {k1:.4f} (σ₃^{{{k2:.4f}}}) (σ_d^{{{k3:.4f}}})$$"
            intercept = 0.0

            is_power     = True
            power_params = popt
            model_obj    = pezo_model_nonnorm
            poly_obj     = None

    # --- Saída e Relatório ---
    # Validação do ajuste: impede R² negativo
    if np.isnan(r2) or r2 < 0:
        st.error(f"❌ Não foi possível ajustar o modelo. R\u00b2 = {r2:.4f}.")
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
        "A função de MR é válida apenas para valores de 0,02≤σ₃≤0,14 e 0,02≤σ_d≤0,42 observada a norma DNIT 134/2018-ME e a precisão do equipamento.",
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

    
    # Downloads LaTeX com gráfico e Word (ZIP + Word)
    try:
        # 1) Gera o .tex e a imagem da superfície
        tex_content, img_data = generate_latex_doc(
            eq_latex, r2, r2_adj, rmse, mae,
            mean_MR, std_MR, energy, degree,
            intercept, df, fig
        )

        # 2) Cria e oferece o ZIP com main.tex + surface_plot.png
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w") as zf:
            zf.writestr("main.tex", tex_content)
            # Adiciona imagem somente se gerada com sucesso
            if img_data:
                zf.writestr("surface_plot.png", img_data)
        zip_buf.seek(0)
        st.download_button(
            "Salvar LaTeX",
            data=zip_buf,
            file_name="Relatorio_Regressao.zip",
            mime="application/zip"
        )

        # 3) Tenta converter para Word via pypandoc
        import pypandoc
        pypandoc.download_pandoc('latest')
        docx_bytes = pypandoc.convert_text(tex_content, 'docx', format='latex')
        st.download_button(
            "Converter: Word (OMML)",
            data=docx_bytes,
            file_name="Relatorio_Regressao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        # 4) Fallback: gera Word diretamente do template
        st.warning(f"Não foi possível gerar LaTeX/OMML: {e}")
        buf = generate_word_doc(
            eq_latex, metrics_txt, fig,
            energy, degree, intercept,
            df, model_type, pezo_option
        )
        buf.seek(0)
        st.download_button(
            "Converter: Word",
            data=buf,
            file_name="Relatorio_Regressao.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

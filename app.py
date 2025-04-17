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
import pypandoc

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

def generate_latex_doc(eq_latex, indicators, nrmse_range, qual_nrmse,
                       cv_rmse, qual_cv, mae_pct, qual_mae,
                       df, fig_path, energy, degree, intercept):
    # Monta o documento .tex
    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs,amsmath,graphicx}",
        r"\begin{document}",
        r"\section*{Relatório de Regressão}",
        r"\subsection*{Configurações}",
        f"Tipo de energia: {energy}\\\\",
    ]
    if degree is not None:
        lines.append(f"Grau polinomial: {degree}\\\\")
    lines += [
        r"\subsection*{Equação Ajustada}",
        eq_latex,
        r"\subsection*{Indicadores Estatísticos}",
        r"\begin{itemize}"
    ]
    for name, val, tip in indicators:
        lines.append(f"  \\item \\textbf{{{name}}}: {val} -- {tip}")
    lines += [
        r"\end{itemize}",
        r"\subsection*{Avaliação da Qualidade do Ajuste}",
        r"\begin{itemize}",
        f"  \\item NRMSE\\_range: {nrmse_range:.2%} → {qual_nrmse}",
        f"  \\item CV(RMSE): {cv_rmse:.2%} → {qual_cv}",
        f"  \\item MAE \\%: {mae_pct:.2%} → {qual_mae}",
        r"\end{itemize}",
        r"\newpage",
        r"\subsection*{Dados do Ensaio Triaxial}",
        df.to_latex(index=False),
        r"\subsection*{Gráfico 3D da Superfície}",
        r"\begin{figure}[ht]\centering",
        rf"  \includegraphics[width=\linewidth]{{{fig_path}}}",
        r"\end{figure}",
        r"\end{document}"
    ]
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

# Configurações na barra lateral
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

energy = st.sidebar.selectbox("Energia", ["Normal", "Intermediária", "Modificada"], index=0)

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
        r2_adj = adjusted_r2(r2, len(y), p_feat) if len(y)>p_feat+1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        fnames = poly.get_feature_names_out(["σ₃","σ_d"])
        if fit_int:
            coefs = np.concatenate(([reg.intercept_], reg.coef_))
            eq_latex = build_latex_equation(coefs, reg.intercept_, [""]+fnames.tolist())
            intercept = reg.intercept_
        else:
            eq_latex = build_latex_equation_no_intercept(reg.coef_, fnames)
            intercept = 0.0
        is_power = False
        power_params = None
        model_obj = reg

    # — Potência Composta —
    elif model_type in ("Potência Composta c/Intercepto", "Potência Composta s/Intercepto"):
        def pot_with_int(X_flat,a0,a1,k1,a2,k2,a3,k3):
            s3,sd = X_flat[:,0],X_flat[:,1]
            return a0 + a1*s3**k1 + a2*(s3*sd)**k2 + a3*sd**k3
        def pot_no_int(X_flat,a1,k1,a2,k2,a3,k3):
            s3,sd = X_flat[:,0],X_flat[:,1]
            return a1*s3**k1 + a2*(s3*sd)**k2 + a3*sd**k3

        mean_y, mean_s3 = y.mean(), X[:,0].mean()
        mean_sd, mean_s3sd = X[:,1].mean(), (X[:,0]*X[:,1]).mean()
        p0_with = [mean_y, mean_y/mean_s3,1, mean_y/mean_s3sd,1, mean_y/mean_sd,1]
        p0_no   = [mean_y/mean_s3,1, mean_y/mean_s3sd,1, mean_y/mean_sd,1]

        if model_type=="Potência Composta c/Intercepto":
            fit_func,p0 = pot_with_int,p0_with
        else:
            fit_func,p0 = pot_no_int,p0_no

        popt,_ = curve_fit(fit_func, X, y, p0=p0, maxfev=200000)
        y_pred = fit_func(X, *popt)
        r2 = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2,len(y),len(popt)) if len(y)>len(popt)+1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae  = mean_absolute_error(y, y_pred)

        if model_type=="Potência Composta s/Intercepto":
            a1,k1,a2,k2,a3,k3 = popt
            eq_latex = f"$$MR = {a1:.4f}\\sigma_3^{{{k1:.4f}}} + {a2:.4f}(\\sigma_3\\sigma_d)^{{{k2:.4f}}} + {a3:.4f}\\sigma_d^{{{k3:.4f}}}$$"
            intercept=0.0
        else:
            a0,a1,k1,a2,k2,a3,k3 = popt
            eq_latex = f"$$MR = {a0:.4f} + {a1:.4f}\\sigma_3^{{{k1:.4f}}} + {a2:.4f}(\\sigma_3\\sigma_d)^{{{k2:.4f}}} + {a3:.4f}\\sigma_d^{{{k3:.4f}}}$$"
            intercept=a0

        is_power = True
        power_params = popt
        model_obj = fit_func
        poly = None

    # — Pezo —
    else:
        def pezo_model(X_flat, k1, k2, k3):
            Pa=0.101325
            s3,sd = X_flat[:,0],X_flat[:,1]
            return k1*Pa*(s3/Pa)**k2*(sd/Pa)**k3

        mean_y, mean_s3 = y.mean(), X[:,0].mean()
        mean_sd = X[:,1].mean()
        p0 = [mean_y/(0.101325*(mean_s3/0.101325)*(mean_sd/0.101325)),1,1]
        popt,_=curve_fit(pezo_model,X,y,p0=p0,maxfev=200000)
        y_pred=pezo_model(X,*popt)
        r2=r2_score(y,y_pred)
        r2_adj=adjusted_r2(r2,len(y),len(popt)) if len(y)>len(popt)+1 else r2
        rmse=np.sqrt(mean_squared_error(y,y_pred))
        mae=mean_absolute_error(y,y_pred)
        k1,k2,k3=popt
        const=k1*0.101325
        eq_latex = f"$$MR = {const:.4f}(\\sigma_3/0.101325)^{{{k2:.4f}}}(\\sigma_d/0.101325)^{{{k3:.4f}}}$$"
        intercept=0.0
        is_power=True
        power_params=popt
        model_obj=pezo_model
        poly=None

    # métricas básicas
    mean_MR,std_MR=y.mean(),y.std()
    indicators=[
        ("R²",f"{r2:.6f}",f"{r2*100:.2f}% da variabilidade explicada"),
        ("R² Ajustado",f"{r2_adj:.6f}","penaliza termos extras"),
        ("RMSE",f"{rmse:.4f} MPa","erro sensível a outliers"),
        ("MAE",f"{mae:.4f} MPa","erro absoluto médio"),
        ("Média MR",f"{mean_MR:.4f} MPa","média observada"),
        ("Desv. Padrão MR",f"{std_MR:.4f} MPa","dispersão dos dados"),
    ]

    # qualidade
    mr_min,mr_max=y.min(),y.max()
    amp,mr_mean=mr_max-mr_min,y.mean()
    nrmse_range=rmse/amp if amp>0 else np.nan
    cv_rmse=rmse/mr_mean if mr_mean else np.nan
    mae_pct=mae/mr_mean if mr_mean else np.nan
    labels_nrmse=["Excelente (≤5%)","Bom (≤10%)","Insuficiente (>10%)"]
    labels_cv=["Excelente (≤10%)","Bom (≤20%)","Insuficiente (>20%)"]
    def qual(val,thr,lbls):
        for t,l in zip(thr,lbls):
            if val<=t: return l
        return lbls[-1]
    qual_nrmse,qual_cv,qual_mae=qual(nrmse_range,[0.05,0.10],labels_nrmse),qual(cv_rmse,[0.10,0.20],labels_cv),qual(mae_pct,[0.10,0.20],labels_cv)

    # gera figura e salva PNG
    fig=plot_3d_surface(df,model_obj,poly,"MR",is_power,power_params)
    img_bytes=fig.to_image(format="png")
    fig_path="/tmp/graph.png"
    with open(fig_path,"wb") as f: f.write(img_bytes)

    # gera .tex e oferece downloads
    latex_str=generate_latex_doc(eq_latex,indicators,nrmse_range,qual_nrmse,cv_rmse,qual_cv,mae_pct,qual_mae,df,fig_path,energy,degree,intercept)
    st.download_button("Salvar LaTeX", latex_str, file_name="Relatorio_Regressao.tex", mime="text/plain")

    try:
        docx_bytes=pypandoc.convert_text(latex_str,'docx',format='latex',extra_args=[f"--resource-path={fig_path.rsplit('/',1)[0]}"])
        st.download_button("Converter para Word", docx_bytes, file_name="Relatorio_Regressao.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    except Exception as e:
        st.error(f"Erro ao converter para Word: {e}")


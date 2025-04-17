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

# Funções de construção de equações LaTeX e formatação no Word (inutlizado para brevidade)
def build_latex_equation(coefs, intercept, feature_names): ...
def build_latex_equation_no_intercept(coefs, feature_names): ...
def add_formatted_equation(doc, eq_text): ...
def add_data_table(doc, df): ...
def plot_3d_surface(df, model, poly, energy_col, is_power=False, power_params=None): ...

def interpret_metrics(r2, r2_adj, rmse, mae, y):
    """Gera texto para relatório Word."""
    txt = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
    txt += f"**R² Ajustado:** {r2_adj:.6f}\n\n"
    txt += f"**RMSE:** {rmse:.4f} MPa\n\n"
    txt += f"**MAE:** {mae:.4f} MPa\n\n"
    txt += f"**Média MR:** {y.mean():.4f} MPa\n\n"
    txt += f"**Desvio Padrão MR:** {y.std():.4f} MPa\n\n"
    return txt

# Função para gerar relatório Word

def generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df): ...

# --- Streamlit App ---
st.set_page_config(page_title="Modelos de MR", layout="wide")
st.title("Modelos de Regressão para MR")
st.markdown("Envie um CSV ou XLSX com colunas **σ3**, **σd** e **MR**.")

uploaded = st.file_uploader("Arquivo", type=["csv", "xlsx"])
if not uploaded:
    st.info("Faça upload para continuar.")
    st.stop()

df = (pd.read_csv(uploaded, decimal=",") if uploaded.name.endswith(".csv") else pd.read_excel(uploaded))
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
        "Pezo (não normalizado)",
        "Pezo (original)"
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

    # — Ajuste dos Modelos —
    if model_type in ("Polinomial c/ Intercepto", "Polinomial s/Intercepto"):
        # Polinomial
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = poly.fit_transform(X)
        fit_int = (model_type == "Polinomial c/ Intercepto")
        reg = LinearRegression(fit_intercept=fit_int)
        reg.fit(Xp, y)
        y_pred = reg.predict(Xp)
        r2 = r2_score(y, y_pred)
        p_feat = Xp.shape[1]
        r2_adj = min(adjusted_r2(r2, len(y), p_feat), r2, 1.0) if len(y) > p_feat + 1 else r2
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
    elif model_type.startswith("Potência Composta"):
        # Potência composta com chute inteligente
        def pot_model(X_flat, a0, a1, k1, a2, k2, a3, k3): ...
        # palpite p0 conforme médias\        mean_y = y.mean(); mean_s3 = X[:,0].mean(); mean_sd = X[:,1].mean(); mean_s3sd = (X[:,0]*X[:,1]).mean()
        p0 = [mean_y, mean_y/mean_s3, 1, mean_y/mean_s3sd, 1, mean_y/mean_sd, 1]
        try:
            popt, _ = curve_fit(pot_model, X, y, p0=p0, maxfev=200000)
        except RuntimeError:
            st.error("❌ Não foi possível ajustar o modelo de Potência Composta. Verifique seus dados ou tente outro modelo.")
            st.stop()
        y_pred = pot_model(X, *popt)
        r2 = r2_score(y, y_pred)
        r2_adj = min(adjusted_r2(r2, len(y), len(popt)), r2, 1.0) if len(y)>len(popt)+1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        a0, a1, k1, a2, k2, a3, k3 = popt
        has_int = model_type.endswith("c/Intercepto")
        if not has_int:
            eq_latex = f"$$MR = {a1:.4f}\sigma_3^{{{k1:.4f}}} + {a2:.4f}(\sigma_3\sigma_d)^{{{k2:.4f}}} + {a3:.4f}\sigma_d^{{{k3:.4f}}}$$"
            intercept = 0.0
        else:
            eq_latex = f"$$MR = {a0:.4f} + {a1:.4f}\sigma_3^{{{k1:.4f}}} + {a2:.4f}(\sigma_3\sigma_d)^{{{k2:.4f}}} + {a3:.4f}\sigma_d^{{{k3:.4f}}}$$"
            intercept = a0
        is_power = True; power_params = popt; model_obj = pot_model; poly = None
    else:
        # Pezo
        def pezo_model(X_flat, k1, k2, k3): ...
        # chute inteligente e ajuste via curve_fit
        mean_y = y.mean(); mean_s3 = X[:,0].mean(); mean_sd = X[:,1].mean()
        Pa0 = 0.101325 if model_type=="Pezo (original)" else 1.0
        k1_0 = mean_y/(Pa0*(mean_s3/Pa0)*(mean_sd/Pa0))
        p0 = [k1_0,1.0,1.0]
        try:
            popt,_ = curve_fit(pezo_model, X, y, p0=p0, maxfev=200000)
        except RuntimeError:
            st.error("❌ Não foi possível ajustar o modelo Pezo. Verifique seus dados ou tente outro modelo.")
            st.stop()
        y_pred = pezo_model(X,*popt)
        r2 = r2_score(y,y_pred)
        r2_adj = min(adjusted_r2(r2,len(y),len(popt)),r2,1.0) if len(y)>len(popt)+1 else r2
        rmse = np.sqrt(mean_squared_error(y,y_pred))
        mae = mean_absolute_error(y,y_pred)
        k1,k2,k3 = popt
        Pa_display = Pa0
        eq_latex = f"$$MR = {k1:.4f}\,{Pa_display:.6f}(\sigma_3/{Pa_display:.6f})^{{{k2:.4f}}}(\sigma_d/{Pa_display:.6f})^{{{k3:.4f}}}$$"
        intercept = 0.0
        is_power=True; power_params=popt; model_obj=pezo_model; poly=None

    # — Saída dos Resultados —
    metrics_txt = interpret_metrics(r2, r2_adj, rmse, mae, y)
    fig = plot_3d_surface(df, model_obj, poly, "MR", is_power=is_power, power_params=power_params)

    st.write("### Equação Ajustada")
    st.latex(eq_latex.strip("$$"))

    st.write("### Indicadores Estatísticos")
    # indicadores principais com tooltips
    mean_MR = y.mean()
    std_MR = y.std()
    indicators = [
        ("R²", f"{r2:.6f}", f"Este valor indica que aproximadamente {r2*100:.2f}% da variabilidade dos dados de MR é explicada pelo modelo."),
        ("R² Ajustado", f"{r2_adj:.6f}", "Penaliza o uso de termos extras; similaridade alta com R² indica bom ajuste sem superajuste."),
        ("RMSE", f"{rmse:.4f} MPa", f"Erro quadrático médio: desvio médio de {rmse:.4f} MPa."),
        ("MAE", f"{mae:.4f} MPa", f"Erro absoluto médio: {mae:.4f} MPa."),
        ("Média MR", f"{mean_MR:.4f} MPa", "Média dos valores observados de MR."),
        ("Desvio Padrão MR", f"{std_MR:.4f} MPa", "Dispersão dos valores em torno da média.")
    ]
    for name, val, tip in indicators:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    # métricas de qualidade do ajuste
    mr_min, mr_max = y.min(), y.max()
    amp = mr_max - mr_min
    mr_mean = mean_MR
    nrmse_range = rmse/amp if amp>0 else np.nan
    cv_rmse = rmse/mr_mean if mr_mean else np.nan
    mae_pct  = mae/mr_mean if mr_mean else np.nan
    quality_metrics = [
        ("NRMSE_range", f"{nrmse_range:.2%}", "NRMSE_range = RMSE / (máx(MR) - mín(MR)), erro relativo à amplitude dos dados."),
        ("CV(RMSE)", f"{cv_rmse:.2%}", "CV(RMSE) = RMSE / média(MR), erro relativo à média dos dados."),
        ("MAE %", f"{mae_pct:.2%}", "MAE % = MAE / média(MR), erro absoluto médio relativo à média dos dados.")
    ]
    for name, val, tip in quality_metrics:
        st.markdown(f"**{name}:** {val} <span title=\"{tip}\">ℹ️</span>", unsafe_allow_html=True)

    st.write(f"**Intercepto:** {intercept:.4f}")
    st.markdown(
        "A função de MR é válida apenas para valores de 0,020≤σ₃≤0,14 e "
        "0,02≤$\\sigma_{d}$≤0,42 observada a norma DNIT 134/2018‑ME.",
        unsafe_allow_html=True
    )

    st.write("### Gráfico 3D da Superfície")
    st.plotly_chart(fig, use_container_width=True)

    buf = generate_word_doc(eq_latex, metrics_txt, fig, energy, degree, intercept, df)
    buf.seek(0)
    st.download_button(
        "Salvar Word",
        data=buf,
        file_name="Relatorio_Regressao.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


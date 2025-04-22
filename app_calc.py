#!/usr/bin/env python
# coding: utf-8

import numpy as np

# Constante de pressão atmosférica [MPa]
Pa = 0.101325

# LaTeX names for variables
latex_vars = {
    "σ3": r"\sigma_3",
    "σd": r"\sigma_d",
    "θ": r"\theta",
    "τ_oct": r"\tau_{oct}"
}

# Funções dos modelos clássicos
def _dunlap(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    return k1 * (s3/Pa)**k2

def _hicks(X, k1, k2):
    sd = X[:,1]
    return k1 * sd**k2

def _wtz81(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 2*s3
    return k1 * (theta/Pa)**k2 * (sd/Pa)**k3

def _uzan85(X, k1, k2, k3):
    return _wtz81(X, k1, k2, k3)

def _johnson86(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 2*s3
    return k1 * theta**k2

def _wtz88(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 2*s3
    tau_oct = 0.471 * sd
    return k1 * Pa * (theta/Pa)**k2 * (tau_oct/Pa)**k3

def _tamber(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    tau_oct = 0.471 * sd
    return k1 * (tau_oct/sd)**k2

def _pezo93(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    return k1 * Pa * (s3/Pa)**k2 * (sd/Pa)**k3

def _hopkins01(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    tau_oct = 0.471 * sd
    return k1 * (s3/Pa) * ((tau_oct/Pa) + 1)**k2

def _ni02(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    return k1 * Pa * (s3/Pa)**k2 * ((sd/Pa) + 1)**k3

def _nchrp28a(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 2*s3
    return k1 * Pa * (theta/Pa)**k2 * ((sd/Pa) + 1)**k3

def _nchrp37a(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 2*s3
    tau_oct = 0.471 * sd
    return k1 * Pa * (theta/Pa)**k2 * ((tau_oct/Pa) + 1)**k3

def _ooi1(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 2*s3
    return k1 * Pa * ((theta/Pa) + 1)**k2 * ((sd/Pa) + 1)**k3

def _ooi2(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 2*s3
    tau_oct = 0.471 * sd
    return k1 * Pa * (theta/Pa)**k2 * ((tau_oct/Pa) + 1)**k3

# Metadados dos modelos clássicos
CLASSICOS = {
    "Dunlap (1963)":   {"func": _dunlap, "n_params":2},
    "Hicks (1970)":    {"func": _hicks, "n_params":2},
    "Witczak (1981)":  {"func": _wtz81, "n_params":3},
    "Uzan (1985)":     {"func": _uzan85, "n_params":3},
    "Johnson et al. (1986)": {"func": _johnson86, "n_params":2},
    "Witczak e Uzan (1988)": {"func": _wtz88, "n_params":3},
    "Tam e Brown (1988)":    {"func": _tamber, "n_params":2},
    "Pezo (1993)":    {"func": _pezo93, "n_params":3},
    "Hopkins et al. (2001)": {"func": _hopkins01, "n_params":2},
    "Ni et al. (2002)": {"func": _ni02, "n_params":3},
    "NCHRP1-28A (2004)": {"func": _nchrp28a, "n_params":3},
    "NCHRP1-37A (2004)": {"func": _nchrp37a, "n_params":3},
    "Ooi et al. (1) (2004)": {"func": _ooi1, "n_params":3},
    "Ooi et al. (2) (2004)": {"func": _ooi2, "n_params":3},
}

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import plotly.graph_objs as go

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
    return "$$" + " \\ ".join(lines) + "$$"

def evaluate_quality(y, rmse, mae):
    amp = y.max() - y.min()
    mean_y = y.mean()
    nrmse = rmse / amp if amp > 0 else float('nan')
    cv_rmse = rmse / mean_y if mean_y else float('nan')
    mae_pct = mae / mean_y if mean_y else float('nan')
    labels_nrmse = ["Excelente (≤5%)","Bom (≤10%)","Insuficiente (>10%)"]
    labels_cv = ["Excelente (≤10%)","Bom (≤20%)","Insuficiente (>20%)"]
    def quality_label(val, thresholds, labels):
        for t, lab in zip(thresholds, labels):
            if val <= t: return lab
        return labels[-1]
    return {
        "NRMSE_range": (nrmse, quality_label(nrmse,[0.05,0.10],labels_nrmse),"NRMSE"),
        "CV(RMSE)": (cv_rmse, quality_label(cv_rmse,[0.10,0.20],labels_cv),"CV"),
        "MAE %": (mae_pct, quality_label(mae_pct,[0.10,0.20],labels_cv),"MAE%")
    }

def calcular_modelo(df, model_type, degree):
    X = df[["σ3","σd"]].values
    y = df["MR"].values
    result = {}

    # Clássico
    if model_type in CLASSICOS:
        meta = CLASSICOS[model_type]
        func = meta["func"]
        p0 = [y.mean()] + [1]* (meta["n_params"]-1)
        popt, _ = curve_fit(func, X, y, p0=p0, maxfev=200000)
        y_pred = func(X, *popt)

        # métricas
        r2 = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt))
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        # Monta equação LaTeX per model
        if model_type == "Dunlap (1963)":
            eq = f"$$MR = {popt[0]:.4f} \left(\frac{{{latex_vars['σ3']}}}{{{Pa}}}\right)^{{{popt[1]:.4f}}}$$"
        elif model_type == "Hicks (1970)":
            eq = f"$$MR = {popt[0]:.4f} {latex_vars['σd']}^{{{popt[1]:.4f}}}$$"
        # ... repita elif para cada clássico ...
        else:
            eq = f"$$MR = " + ", ".join(f"{v:.4f}" for v in popt) + "$$"

        result.update({
            "eq_latex": eq, "intercept":0.0,
            "r2":r2,"r2_adj":r2_adj,"rmse":rmse,"mae":mae,
            "mean_MR":y.mean(),"std_MR":y.std(),
            "model_obj":func,"poly_obj":None,
            "is_power":True,"power_params":popt
        })
        result["quality"] = evaluate_quality(y, rmse, mae)
        return result

    # Genéricos seguem idênticos...
    # [mantém todo o código anteriores para genéricos, omitido aqui para brevidade]

# plot_3d_surface permanece igual

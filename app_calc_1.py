#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Constante de pressão atmosférica [MPa]
Pa = 0.101325

# --- Funções dos modelos clássicos ---
def _dunlap(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    return k1 * (s3 / Pa) ** k2

def _hicks(X, k1, k2):
    sd = X[:,1]
    return k1 * sd ** k2

def _wtz81(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3 * s3
    return k1 * (theta / Pa) ** k2 * (sd / Pa) ** k3

def _uzan85(X, k1, k2, k3):
    return _wtz81(X, k1, k2, k3)

def _johnson86(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3 * s3
    return k1 * theta ** k2

def _wtz88(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3 * s3
    tau_oct = 0.471 * sd
    return k1 * Pa * (theta / Pa) ** k2 * (tau_oct / Pa) ** k3

def _tamber(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    tau_oct = 0.471 * sd
    return k1 * (tau_oct / sd) ** k2

def _pezo93(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    return k1 * Pa * (s3 / Pa) ** k2 * (sd / Pa) ** k3

def _hopkins01(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    tau_oct = 0.471 * sd
    return k1 * (s3 / Pa) * ((tau_oct / Pa) + 1) ** k2

def _ni02(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    return k1 * Pa * (s3 / Pa) ** k2 * ((sd / Pa) + 1) ** k3

def _nchrp28a(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3 * s3
    return k1 * Pa * (theta / Pa) ** k2 * ((sd / Pa) + 1) ** k3

def _nchrp37a(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3 * s3
    tau_oct = 0.471 * sd
    return k1 * Pa * (theta / Pa) ** k2 * ((tau_oct / Pa) + 1) ** k3

def _ooi1(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3 * s3
    return k1 * Pa * ((theta / Pa) + 1) ** k2 * ((sd / Pa) + 1) ** k3

def _ooi2(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3 * s3
    tau_oct = 0.471 * sd
    return k1 * Pa * (theta / Pa) ** k2 * ((tau_oct / Pa) + 1) ** k3

CLASSICOS = {
    "Dunlap (1963)": {"func": _dunlap, "n_params": 2},
    "Hicks (1970)": {"func": _hicks, "n_params": 2},
    "Witczak (1981)": {"func": _wtz81, "n_params": 3},
    "Uzan (1985)": {"func": _uzan85, "n_params": 3},
    "Johnson et al. (1986)": {"func": _johnson86, "n_params": 2},
    "Witczak e Uzan (1988)": {"func": _wtz88, "n_params": 3},
    "Tam e Brown (1988)": {"func": _tamber, "n_params": 2},
    "Pezo (1993)": {"func": _pezo93, "n_params": 3},
    "Hopkins et al. (2001)": {"func": _hopkins01, "n_params": 2},
    "Ni et al. (2002)": {"func": _ni02, "n_params": 3},
    "NCHRP1-28A (2004)": {"func": _nchrp28a, "n_params": 3},
    "NCHRP1-37A (2004)": {"func": _nchrp37a, "n_params": 3},
    "Ooi et al. (1) (2004)": {"func": _ooi1, "n_params": 3},
    "Ooi et al. (2) (2004)": {"func": _ooi2, "n_params": 3},
}

def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def build_latex_equation(coefs, intercept, feature_names):
    terms_per_line = 4
    parts = []
    for coef, term in zip(coefs, feature_names):
        sign = " + " if coef >= 0 else " - "
        parts.append(f"{sign}{abs(coef):.4f}{term.replace(" " , "")}")
    lines, curr = [], f"MR = {intercept:.4f}"
    for i, part in enumerate(parts):
        curr += part
        if (i + 1) % terms_per_line == 0:
            lines.append(curr)
            curr = ""
    if curr.strip():
        lines.append(curr)
    return "$$" + " \ ".join(lines) + "$$"


def calculate_modelo_classico(df, model_name):
    X = df[["σ3", "σd"]].values
    y = df["MR"].values
    if model_name not in CLASSICOS:
        raise ValueError(f"Modelo '{model_name}' não encontrado.")
    meta = CLASSICOS[model_name]
    func = meta["func"]
    p0 = [y.mean()] + [1] * (meta["n_params"] - 1)
    popt, _ = curve_fit(func, X, y, p0=p0, maxfev=200000)
    y_pred = func(X, *popt)
    r2 = r2_score(y, y_pred)
    r2_adj = adjusted_r2(r2, len(y), len(popt))
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    # Monta equação LaTeX específica para cada modelo clássico
    if model_name == "Dunlap (1963)":
        eq = f"$$MR = {popt[0]:.4f} \left(\frac{{\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}$$"
    elif model_name == "Hicks (1970)":
        eq = f"$$MR = {popt[0]:.4f} \sigma_d^{{{popt[1]:.4f}}}$$"
    elif model_name in ["Witczak (1981)", "Uzan (1985)"]:
        eq = (
            f"$$MR = {popt[0]:.4f} \left(\frac{{\sigma_d + 3\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{\sigma_d}}{{Pa}}\right)^{{{popt[2]:.4f}}}$$"
        )
    elif model_name == "Johnson et al. (1986)":
        eq = f"$$MR = {popt[0]:.4f} (\sigma_d + 3\sigma_3)^{{{popt[1]:.4f}}}$$"
    elif model_name == "Witczak e Uzan (1988)":
        eq = (
            f"$$MR = {popt[0]:.4f} Pa \left(\frac{{\sigma_d + 3\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{0.471\sigma_d}}{{Pa}}\right)^{{{popt[2]:.4f}}}$$"
        )
    elif model_name == "Tam e Brown (1988)":
        eq = f"$$MR = {popt[0]:.4f} \left(\frac{{0.471\sigma_d}}{{\sigma_d}}\right)^{{{popt[1]:.4f}}}$$"
    elif model_name == "Pezo (1993)":
        eq = (
            f"$$MR = {popt[0]*Pa:.4f} \left(\frac{{\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{\sigma_d}}{{Pa}}\right)^{{{popt[2]:.4f}}}$$"
        )
    elif model_name == "Hopkins et al. (2001)":
        eq = (
            f"$$MR = {popt[0]:.4f} \left(\frac{{\sigma_3}}{{Pa}}\right)"
            f" \left(\frac{{0.471\sigma_d}}{{Pa}} + 1\right)^{{{popt[1]:.4f}}}$$"
        )
    elif model_name == "Ni et al. (2002)":
        eq = (
            f"$$MR = {popt[0]:.4f} Pa \left(\frac{{\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{\sigma_d}}{{Pa}} + 1\right)^{{{popt[2]:.4f}}}$$"
        )
    elif model_name == "NCHRP1-28A (2004)":
        eq = (
            f"$$MR = {popt[0]:.4f} Pa \left(\frac{{\sigma_d + 3\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{\sigma_d}}{{Pa}} + 1\right)^{{{popt[2]:.4f}}}$$"
        )
    elif model_name == "NCHRP1-37A (2004)":
        eq = (
            f"$$MR = {popt[0]:.4f} Pa \left(\frac{{\sigma_d + 3\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{0.471\sigma_d}}{{Pa}} + 1\right)^{{{popt[2]:.4f}}}$$"
        )
    elif model_name == "Ooi et al. (1) (2004)":
        eq = (
            f"$$MR = {popt[0]:.4f} Pa \left(\frac{{\sigma_d + 3\sigma_3}}{{Pa}} + 1\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{\sigma_d}}{{Pa}} + 1\right)^{{{popt[2]:.4f}}}$$"
        )
    elif model_name == "Ooi et al. (2) (2004)":
        eq = (
            f"$$MR = {popt[0]:.4f} Pa \left(\frac{{\sigma_d + 3\sigma_3}}{{Pa}}\right)^{{{popt[1]:.4f}}}"
            f" \left(\frac{{0.471\sigma_d}}{{Pa}} + 1\right)^{{{popt[2]:.4f}}}$$"
        )
    else:
        feature_names = [f"k{i+1}" for i in range(len(popt))]
        eq = build_latex_equation(popt, 0.0, feature_names)

    return {
        "eq_latex": eq,
        "intercept": 0.0,
        "r2": r2,
        "r2_adj": r2_adj,
        "rmse": rmse,
        "mae": mae,
        "mean_MR": y.mean(),
        "std_MR": y.std(),
        "model_obj": func,
        "power_params": popt,
        "is_power": True,
        "quality": {
            "NRMSE": (rmse / (y.max()-y.min()), "", ""),
            "CV(RMSE)": (rmse / y.mean(), "", ""),
            "MAE %": (mae / y.mean(), "", "")
        }
    }

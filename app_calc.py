#!/usr/bin/env python
# coding: utf-8

# --- app_calc.py ---
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import plotly.graph_objs as go

def adjusted_r2(r2, n, p):
    """Retorna R² ajustado."""
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def build_latex_equation(coefs, intercept, feature_names):
    """Monta equação LaTeX para modelo polinomial com intercepto."""
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
    # Separator is a double backslash plus newline for LaTeX
    return "$$" + " \\ 
".join(lines) + "$$"

def build_latex_equation_no_intercept(coefs, feature_names):
    """Monta equação LaTeX para modelo polinomial sem intercepto."""
    return build_latex_equation(np.concatenate(([0.0], coefs)), 0.0, [""] + feature_names)

def quality_label(val, thresholds, labels):
    """Classifica valor numérico segundo thresholds e labels."""
    for t, lab in zip(thresholds, labels):
        if val <= t:
            return lab
    return labels[-1]

def evaluate_quality(y, rmse, mae):
    """Avalia e retorna dict de qualidade do ajuste."""
    amp = y.max() - y.min()
    mean_y = y.mean()
    nrmse = rmse / amp if amp > 0 else float('nan')
    cv_rmse = rmse / mean_y if mean_y else float('nan')
    mae_pct = mae / mean_y if mean_y else float('nan')

    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]

    qual_nrmse = quality_label(nrmse, [0.05, 0.10], labels_nrmse)
    qual_cv     = quality_label(cv_rmse, [0.10, 0.20], labels_cv)
    qual_mae    = quality_label(mae_pct, [0.10, 0.20], labels_cv)

    return {
        "NRMSE_range": (nrmse, qual_nrmse,
                         "NRMSE_range: RMSE normalizado pela amplitude dos valores de MR; indicador associado ao RMSE."),
        "CV(RMSE)":    (cv_rmse, qual_cv,
                         "CV(RMSE): coeficiente de variação do RMSE (RMSE/média MR); indicador associado ao RMSE."),
        "MAE %":       (mae_pct, qual_mae,
                         "MAE %: MAE dividido pela média de MR; indicador associado ao MAE.")
    }

def calcular_modelo(df, model_type, degree):
    """Executa ajuste de modelo e retorna resultados e métricas."""
    X = df[["σ3", "σd"]].values
    y = df["MR"].values
    result = {}

    # Modelo Polinomial
    if model_type.startswith("Polinomial"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = poly.fit_transform(X)
        fit_int = (model_type == "Polinomial c/ Intercepto")
        reg = LinearRegression(fit_intercept=fit_int)
        reg.fit(Xp, y)
        y_pred = reg.predict(Xp)

        r2 = r2_score(y, y_pred)
        p_feat = Xp.shape[1]
        r2_adj = adjusted_r2(r2, len(y), p_feat) if len(y) > p_feat + 1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        fnames = poly.get_feature_names_out(["σ₃", "σ_d"]).tolist()
        if fit_int:
            coefs = np.concatenate(([reg.intercept_], reg.coef_))
            eq = build_latex_equation(coefs, reg.intercept_, [""] + fnames)
            intercept = reg.intercept_
        else:
            eq = build_latex_equation_no_intercept(reg.coef_, fnames)
            intercept = 0.0

        result.update({
            "eq_latex": eq,
            "intercept": intercept,
            "r2": r2,
            "r2_adj": r2_adj,
            "rmse": rmse,
            "mae": mae,
            "mean_MR": y.mean(),
            "std_MR": y.std(),
            "model_obj": reg,
            "poly_obj": poly,
            "is_power": False,
            "power_params": None
        })

    # Modelo Potência Composta
    elif model_type == "Potência Composta":
        def pot(X_flat, a1, k1, a2, k2, a3, k3):
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return a1 * s3**k1 + a2 * (s3 * sd)**k2 + a3 * sd**k3

        p0 = [y.mean()/X[:,0].mean(), 1,
              y.mean()/(X[:,0]*X[:,1]).mean(), 1,
              y.mean()/X[:,1].mean(), 1]
        popt, _ = curve_fit(pot, X, y, p0=p0, maxfev=200000)
        y_pred = pot(X, *popt)

        r2 = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt)) if len(y) > len(popt)+1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        a1, k1, a2, k2, a3, k3 = popt
        eq = (f"$$MR = {a1:.4f}σ₃^{{{k1:.4f}}} + {a2:.4f}(σ₃σ_d)^{{{k2:.4f}}} + {a3:.4f}σ_d^{{{k3:.4f}}}$$")

        result.update({
            "eq_latex": eq,
            "intercept": 0.0,
            "r2": r2,
            "r2_adj": r2_adj,
            "rmse": rmse,
            "mae": mae,
            "mean_MR": y.mean(),
            "std_MR": y.std(),
            "model_obj": pot,
            "poly_obj": None,

... (rest of code) ...

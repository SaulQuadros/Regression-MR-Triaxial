#!/usr/bin/env python
# coding: utf-8

# --- app_calc.py ---
import numpy as np

# Constante de pressão atmosférica [MPa]
Pa = 0.101325

# Funções dos modelos clássicos
def _dunlap(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    return k1 * (s3/Pa)**k2

def _hicks(X, k1, k2):
    sd = X[:,1]
    return k1 * sd**k2

def _wtz81(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3*s3
    return k1 * (theta/Pa)**k2 * (sd/Pa)**k3

def _uzan85(X, k1, k2, k3):
    # mesmo que Witczak (1981)
    return _wtz81(X, k1, k2, k3)

def _johnson86(X, k1, k2):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3*s3
    return k1 * theta**k2

def _wtz88(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3*s3
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
    theta = sd + 3*s3
    return k1 * Pa * (theta/Pa)**k2 * ((sd/Pa) + 1)**k3

def _nchrp37a(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3*s3
    tau_oct = 0.471 * sd
    return k1 * Pa * (theta/Pa)**k2 * ((tau_oct/Pa) + 1)**k3

def _ooi1(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3*s3
    return k1 * Pa * ((theta/Pa) + 1)**k2 * ((sd/Pa) + 1)**k3

def _ooi2(X, k1, k2, k3):
    s3, sd = X[:,0], X[:,1]
    theta = sd + 3*s3
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
    "Hopkins et al. (2001)":{"func": _hopkins01, "n_params":2},
    "Ni et al. (2002)":{"func": _ni02, "n_params":3},
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
    return "$$" + " \\ \n".join(lines) + "$$"

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
    # Verifica se é modelo clássico
    if model_type in CLASSICOS:
        meta = CLASSICOS[model_type]
        func = meta["func"]
        Xf = X  # X contém [σ3, σd]
        # chute inicial de parâmetros
        if meta["n_params"] == 2:
            p0 = [y.mean(), 1]
        else:
            p0 = [y.mean(), 1, 1]
        popt, _ = curve_fit(func, Xf, y, p0=p0, maxfev=200000)
        y_pred = func(Xf, *popt)

        # métricas
        r2 = r2_score(y, y_pred)
        p = len(popt)
        r2_adj = adjusted_r2(r2, len(y), p) if len(y) > p+1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        # monta equação genérica em LaTeX
        params = ", ".join([f"{v:.4f}" for v in popt])
        eq = f"$$MR = {{{params}}}$$"
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
            "model_obj": func,
            "poly_obj": None,
            "is_power": False,
            "power_params": None
        })
        result["quality"] = evaluate_quality(y, rmse, mae)
        return result

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
        eq = f"$$MR = {a1:.4f}σ₃^{{{k1:.4f}}} + {a2:.4f}(σ₃σ_d)^{{{k2:.4f}}} + {a3:.4f}σ_d^{{{k3:.4f}}}$$"

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
            "is_power": True,
            "power_params": popt
        })

    # Modelo Pezo
    else:
        def pezo(X_flat, k1, k2, k3):
            Pa = 0.101325
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return k1 * Pa * (s3/Pa)**k2 * (sd/Pa)**k3

        p0 = [y.mean()/(0.101325*(X[:,0]/0.101325).mean()*(X[:,1]/0.101325).mean()), 1, 1]
        popt, _ = curve_fit(pezo, X, y, p0=p0, maxfev=200000)
        y_pred = pezo(X, *popt)

        r2 = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt)) if len(y) > len(popt)+1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        const = popt[0] * 0.101325
        eq = f"$$MR = {const:.4f}(σ₃/0.101325)^{{{popt[1]:.4f}}}(σ_d/0.101325)^{{{popt[2]:.4f}}}$$"

        result.update({
            "eq_latex": eq,
            "intercept": 0.0,
            "r2": r2,
            "r2_adj": r2_adj,
            "rmse": rmse,
            "mae": mae,
            "mean_MR": y.mean(),
            "std_MR": y.std(),
            "model_obj": pezo,
            "poly_obj": None,
            "is_power": True,
            "power_params": popt
        })

    # Avaliação da Qualidade do Ajuste
    result["quality"] = evaluate_quality(y, result["rmse"], result["mae"])
    return result

def interpret_metrics(r2, r2_adj, rmse, mae, y):
    txt = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
    txt += f"**R² Ajustado:** {r2_adj:.6f}\n\n"
    txt += f"**RMSE:** {rmse:.4f} MPa\n\n"
    txt += f"**MAE:** {mae:.4f} MPa\n\n"
    txt += f"**Média MR:** {y.mean():.4f} MPa\n\n"
    txt += f"**Desvio Padrão MR:** {y.std():.4f} MPa\n\n"
    return txt

def plot_3d_surface(df, model, poly, energy_col, is_power=False, power_params=None):
    import numpy as _np
    import plotly.graph_objs as go
    s3 = _np.linspace(df["σ3"].min(), df["σ3"].max(), 30)
    sd = _np.linspace(df["σd"].min(), df["σd"].max(), 30)
    s3g, sdg = _np.meshgrid(s3, sd)
    Xg = _np.c_[s3g.ravel(), sdg.ravel()]
    MRg = (model(Xg, *power_params) if is_power else model.predict(poly.transform(Xg)))
    MRg = MRg.reshape(s3g.shape)
    fig = go.Figure(data=[go.Surface(x=s3g, y=sdg, z=MRg)])
    fig.add_trace(go.Scatter3d(
        x=df["σ3"], y=df["σd"], z=df[energy_col],
        mode='markers', marker=dict(size=5, color='red'), name='Dados'
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

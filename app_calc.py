#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import plotly.graph_objs as go

# Constante de pressão atmosférica [MPa]
Pa = 0.101325

# --- Funções dos modelos clássicos ---
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
    "Ni et al. (2002)":   {"func": _ni02, "n_params":3},
    "NCHRP1-28A (2004)": {"func": _nchrp28a, "n_params":3},
    "NCHRP1-37A (2004)": {"func": _nchrp37a, "n_params":3},
    "Ooi et al. (1) (2004)": {"func": _ooi1, "n_params":3},
    "Ooi et al. (2) (2004)": {"func": _ooi2, "n_params":3},
}

def adjusted_r2(r2, n, p):
    """Retorna R² ajustado."""
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

def build_latex_equation(coefs, intercept, feature_names):
    terms_per_line = 4
    parts = []
    for coef, term in zip(coefs, feature_names):
        sign = " + " if coef >= 0 else " - "
        parts.append(f"{sign}{abs(coef):.4f}{term.replace(' ', '')}")
    lines, curr = [], f"MR = {intercept:.4f}"
    for i, part in enumerate(parts):
        curr += part
        if (i + 1) % terms_per_line == 0:
            lines.append(curr)
            curr = ""
    if curr.strip(): lines.append(curr)
    return "$$" + " \\ ".join(lines) + "$$"

def build_latex_equation_no_intercept(coefs, feature_names):
    return build_latex_equation(np.concatenate(([0.0], coefs)), 0.0, [""] + feature_names)

def evaluate_quality(y, rmse, mae):
    amp = y.max() - y.min()
    mean_y = y.mean()
    nrmse = rmse / amp if amp > 0 else float('nan')
    cv_rmse = rmse / mean_y if mean_y else float('nan')
    mae_pct = mae / mean_y if mean_y else float('nan')
    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]
    qual_nrmse = labels_nrmse[0] if nrmse <= 0.05 else labels_nrmse[1] if nrmse <= 0.10 else labels_nrmse[2]
    qual_cv    = labels_cv[0]    if cv_rmse <= 0.10 else labels_cv[1]    if cv_rmse <= 0.20 else labels_cv[2]
    qual_mae   = labels_cv[0]    if mae_pct <= 0.10 else labels_cv[1]    if mae_pct <= 0.20 else labels_cv[2]
    return {
        "NRMSE_range": (nrmse, qual_nrmse, "NRMSE normalizado pela amplitude dos MR."),
        "CV(RMSE)":    (cv_rmse,  qual_cv,   "CV(RMSE): RMSE dividido pela média MR."),
        "MAE %":       (mae_pct,   qual_mae,  "MAE %: MAE dividido pela média MR.")
    }

def calcular_modelo(df, model_type, degree):
    # Variáveis de entrada e resposta
    X = df[["σ3","σd"]].values
    y = df["MR"].values

    # Modelos clássicos
    if model_type in CLASSICOS:
        meta = CLASSICOS[model_type]
        func = meta["func"]
        p0 = [y.mean()] + [1]*(meta["n_params"]-1)
        popt, _ = curve_fit(func, X, y, p0=p0, maxfev=200000)
        y_pred = func(X, *popt)
        r2 = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt))
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        params = ", ".join(f"{v:.4f}" for v in popt)
        eq = f"$$MR = ({params})$$"
        return {
            "eq_latex": eq, "intercept":0.0,
            "r2":r2, "r2_adj":r2_adj, "rmse":rmse,
            "mae":mae, "mean_MR":y.mean(), "std_MR":y.std(),
            "model_obj":func, "poly_obj":None,
            "is_power":True, "power_params":popt,
            "quality": evaluate_quality(y, rmse, mae)
        }

    # Modelos genéricos (polinomiais)
    if model_type.startswith("Polinomial"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = poly.fit_transform(X)
        fit_int = (model_type == "Polinomial c/ Intercepto")
        reg = LinearRegression(fit_intercept=fit_int)
        reg.fit(Xp, y)
        y_pred = reg.predict(Xp)
        r2 = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), Xp.shape[1])
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        fnames = poly.get_feature_names_out(["σ₃","σ_d"]).tolist()
        coefs = np.concatenate(([reg.intercept_], reg.coef_)) if fit_int else reg.coef_
        eq = build_latex_equation(coefs, reg.intercept_ if fit_int else 0.0, [""]+fnames)
        return {
            "eq_latex":eq, "intercept":reg.intercept_,
            "r2":r2,"r2_adj":r2_adj,"rmse":rmse,
            "mae":mae,"mean_MR":y.mean(),"std_MR":y.std(),
            "model_obj":reg,"poly_obj":poly,
            "is_power":False,"power_params":None,
            "quality":evaluate_quality(y,rmse,mae)
        }

    # Potência composta
    if model_type == "Potência Composta":
        def pot(X_flat, a1, k1, a2, k2, a3, k3):
            s3, sd = X_flat[:,0], X_flat[:,1]
            return a1*s3**k1 + a2*(s3*sd)**k2 + a3*sd**k3
        p0 = [y.mean()/X[:,0].mean(),1,y.mean()/(X[:,0]*X[:,1]).mean(),1,y.mean()/X[:,1].mean(),1]
        popt, _ = curve_fit(pot, X, y, p0=p0, maxfev=200000)
        y_pred = pot(X, *popt)
        r2 = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt))
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        eq = f"$$MR = {popt[0]:.4f}\sigma_3^{{{popt[1]:.4f}}} + {popt[2]:.4f}(\sigma_3\sigma_d)^{{{popt[3]:.4f}}} + {popt[4]:.4f}\sigma_d^{{{popt[5]:.4f}}}$$"
        return {
            "eq_latex":eq,"intercept":0.0,"r2":r2,"r2_adj":r2_adj,
            "rmse":rmse,"mae":mae,"mean_MR":y.mean(),"std_MR":y.std(),
            "model_obj":pot,"poly_obj":None,
            "is_power":True,"power_params":popt,
            "quality":evaluate_quality(y,rmse,mae)
        }

    # Modelo Pezo
    def pezo(X_flat, k1, k2, k3):
        s3, sd = X_flat[:,0], X_flat[:,1]
        return k1*Pa*(s3/Pa)**k2*(sd/Pa)**k3
    p0 = [y.mean()/(Pa*(X[:,0]/Pa).mean()*(X[:,1]/Pa).mean()),1,1]
    popt, _ = curve_fit(pezo, X, y, p0=p0, maxfev=200000)
    y_pred = pezo(X, *popt)
    r2 = r2_score(y, y_pred)
    r2_adj = adjusted_r2(r2, len(y), len(popt))
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    eq = f"$$MR = {popt[0]*Pa:.4f}(\sigma_3/Pa)^{{{popt[1]:.4f}}}(\sigma_d/Pa)^{{{popt[2]:.4f}}}$$"
    return {
        "eq_latex":eq,"intercept":0.0,"r2":r2,"r2_adj":r2_adj,
        "rmse":rmse,"mae":mae,"mean_MR":y.mean(),"std_MR":y.std(),
        "model_obj":pezo,"poly_obj":None,
        "is_power":True,"power_params":popt,
        "quality":evaluate_quality(y,rmse,mae)
    }

def interpret_metrics(r2, r2_adj, rmse, mae, y):
    txt  = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
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
    fig.add_trace(go.Scatter3d(x=df["σ3"], y=df["σd"], z=df[energy_col],
                               mode='markers', marker=dict(size=5, color='red'), name='Dados'))
    fig.update_layout(scene=dict(
        xaxis_title='σ₃ (MPa)', yaxis_title='σ_d (MPa)', zaxis_title='MR (MPa)'
    ), margin=dict(l=0,r=0,b=0,t=30))
    return fig

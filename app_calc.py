#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# --- app_calc.py ---
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import plotly.graph_objs as go

def adjusted_r2(r2, n, p):
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

# constrói equações LaTeX

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

def calcular_modelo(df, model_type, degree):
    X = df[["σ3", "σd"]].values
    y = df["MR"].values
    
    # Polinomial
    if model_type in ("Polinomial c/ Intercepto", "Polinomial s/Intercepto"):
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
        mae  = mean_absolute_error(y, y_pred)

        fnames = poly.get_feature_names_out(["σ₃", "σ_d"])  # noqa
        if fit_int:
            coefs = np.concatenate(([reg.intercept_], reg.coef_))
            feature_names = [""] + fnames.tolist()
            eq_latex = build_latex_equation(coefs, reg.intercept_, feature_names)
            intercept = reg.intercept_
        else:
            eq_latex = build_latex_equation_no_intercept(reg.coef_, fnames)
            intercept = 0.0

        return {
            "eq_latex": eq_latex,
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
        }
    
    # Potência Composta
    elif model_type == "Potência Composta":
        def pot_no_int(X_flat, a1, k1, a2, k2, a3, k3):
            s3, sd = X_flat[:, 0], X_flat[:, 1]
            return a1 * s3**k1 + a2 * (s3 * sd)**k2 + a3 * sd**k3

        mean_y = y.mean()
        p0 = [mean_y/X[:,0].mean(),1, mean_y/(X[:,0]*X[:,1]).mean(),1, mean_y/X[:,1].mean(),1]
        popt, _ = curve_fit(pot_no_int, X, y, p0=p0, maxfev=200000)
        y_pred = pot_no_int(X, *popt)

        r2     = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt)) if len(y) > len(popt)+1 else r2
        rmse   = np.sqrt(mean_squared_error(y, y_pred))
        mae    = mean_absolute_error(y, y_pred)

        a1,k1,a2,k2,a3,k3 = popt
        eq_latex = f"$$MR = {a1:.4f}σ₃^{{{k1:.4f}}} + {a2:.4f}(σ₃σ_d)^{{{k2:.4f}}} + {a3:.4f}σ_d^{{{k3:.4f}}}$$"

        return {
            "eq_latex": eq_latex,
            "intercept": 0.0,
            "r2": r2,
            "r2_adj": r2_adj,
            "rmse": rmse,
            "mae": mae,
            "mean_MR": y.mean(),
            "std_MR": y.std(),
            "model_obj": pot_no_int,
            "poly_obj": None,
            "is_power": True,
            "power_params": popt
        }
    
    # Modelo Pezo
    else:
        def pezo_model(X_flat, k1, k2, k3):
            Pa = 0.101325
            s3, sd = X_flat[:,0], X_flat[:,1]
            return k1 * Pa * (s3/Pa)**k2 * (sd/Pa)**k3

        mean_y   = y.mean()
        k1_0     = mean_y/(0.101325*(X[:,0]/0.101325).mean()*(X[:,1]/0.101325).mean())
        popt, _  = curve_fit(pezo_model, X, y, p0=[k1_0,1,1], maxfev=200000)
        y_pred   = pezo_model(X, *popt)

        r2     = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt)) if len(y) > len(popt)+1 else r2
        rmse   = np.sqrt(mean_squared_error(y, y_pred))
        mae    = mean_absolute_error(y, y_pred)

        const   = popt[0] * 0.101325
        eq_latex = f"$$MR = {const:.4f}(σ₃/0.101325)^{{{popt[1]:.4f}}}(σ_d/0.101325)^{{{popt[2]:.4f}}}$$"

        return {
            "eq_latex": eq_latex,
            "intercept": 0.0,
            "r2": r2,
            "r2_adj": r2_adj,
            "rmse": rmse,
            "mae": mae,
            "mean_MR": y.mean(),
            "std_MR": y.std(),
            "model_obj": pezo_model,
            "poly_obj": None,
            "is_power": True,
            "power_params": popt
        }

def interpret_metrics(r2, r2_adj, rmse, mae, y):
    txt = f"**R²:** {r2:.6f} (~{r2*100:.2f}% explicado)\n\n"
    txt += f"**R² Ajustado:** {r2_adj:.6f}\n\n"
    txt += f"**RMSE:** {rmse:.4f} MPa\n\n"
    txt += f"**MAE:** {mae:.4f} MPa\n\n"
    txt += f"**Média MR:** {y.mean():.4f} MPa\n\n"
    txt += f"**Desvio Padrão MR:** {y.std():.4f} MPa\n\n"
    return txt

def plot_3d_surface(df, model, poly, energy_col, is_power=False, power_params=None):
    import numpy as np
    import plotly.graph_objs as go
    s3 = np.linspace(df["σ3"].min(), df["σ3"].max(), 30)
    sd = np.linspace(df["σd"].min(), df["σd"].max(), 30)
    s3g, sdg = np.meshgrid(s3, sd)
    Xg = np.c_[s3g.ravel(), sdg.ravel()]
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
        ), margin=dict(l=0,r=0,b=0,t=30)
    )
    return fig


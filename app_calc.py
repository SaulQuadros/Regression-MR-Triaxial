#!/usr/bin/env python
# coding: utf-8

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
    """
    Monta a equação LaTeX dividindo em até 4 termos por linha e garantindo
    que nunca haja sinais duplicados como "+ -".
    """
    # Começa listando intercepto
    terms = [f"MR = {intercept:.4f}"]
    # Adiciona cada coeficiente com sinal correto
    for coef, name in zip(coefs, feature_names):
        if name == "":
            continue
        if coef >= 0:
            terms.append(f"+ {coef:.4f}{name}")
        else:
            terms.append(f"- {abs(coef):.4f}{name}")
    # Agrupa em linhas de até 4 termos
    lines = []
    for i in range(0, len(terms), 4):
        lines.append(" ".join(terms[i:i+4]))
    # Junta com quebras LaTeX \ 
    body = " \\ 
".join(lines)
    return "$$" + body + "$$"

def build_latex_equation_no_intercept(coefs, feature_names):
    """Chama build_latex_equation com intercepto zero."""
    full_coefs = np.concatenate(([0.0], coefs))
    return build_latex_equation(full_coefs, 0.0, [""] + feature_names)

def quality_label(val, thresholds, labels):
    for t, lab in zip(thresholds, labels):
        if val <= t:
            return lab
    return labels[-1]

def evaluate_quality(y, rmse, mae):
    """Avalia a qualidade do ajuste e retorna dicionário com valores e tooltips."""
    amp = y.max() - y.min()
    mean_y = y.mean()
    nrmse = rmse / amp if amp > 0 else float('nan')
    cv_rmse = rmse / mean_y if mean_y else float('nan')
    mae_pct = mae / mean_y if mean_y else float('nan')

    labels_nrmse = ["Excelente (≤5%)", "Bom (≤10%)", "Insuficiente (>10%)"]
    labels_cv    = ["Excelente (≤10%)", "Bom (≤20%)", "Insuficiente (>20%)"]

    return {
        "NRMSE":    (nrmse, quality_label(nrmse, [0.05,0.10], labels_nrmse),
                     "NRMSE: RMSE normalizado pela amplitude dos valores de MR."),
        "CV(RMSE)": (cv_rmse, quality_label(cv_rmse, [0.10,0.20], labels_cv),
                     "CV(RMSE): coeficiente de variação do RMSE."),
        "MAE %":    (mae_pct, quality_label(mae_pct, [0.10,0.20], labels_cv),
                     "MAE %: MAE dividido pela média de MR.")
    }

def calcular_modelo(df, model_type, degree):
    """Executa ajuste de modelo e retorna dicionário com resultados e métricas."""
    X = df[["σ3","σd"]].values
    y = df["MR"].values
    result = {}

    # — Modelo Polinomial —
    if model_type in ("Polinomial c/ Intercepto", "Polinomial s/Intercepto"):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = poly.fit_transform(X)
        fit_int = (model_type == "Polinomial c/ Intercepto")
        reg = LinearRegression(fit_intercept=fit_int)
        reg.fit(Xp, y)
        y_pred = reg.predict(Xp)

        r2 = r2_score(y, y_pred)
        p = Xp.shape[1]
        r2_adj = adjusted_r2(r2, len(y), p) if len(y) > p+1 else r2
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae  = mean_absolute_error(y, y_pred)

        fnames = poly.get_feature_names_out(["σ₃","σ_d"]).tolist()
        if fit_int:
            coefs = np.concatenate(([reg.intercept_], reg.coef_))
            eq_latex = build_latex_equation(coefs, reg.intercept_, [""] + fnames)
            intercept = reg.intercept_
        else:
            eq_latex = build_latex_equation_no_intercept(reg.coef_, fnames)
            intercept = 0.0

        result.update({
            "eq_latex":    eq_latex,
            "intercept":   intercept,
            "r2":          r2,
            "r2_adj":      r2_adj,
            "rmse":        rmse,
            "mae":         mae,
            "mean_MR":     y.mean(),
            "std_MR":      y.std(),
            "model_obj":   reg,
            "poly_obj":    poly,
            "is_power":    False,
            "power_params": None
        })

    # — Potência Composta —
    elif model_type == "Potência Composta":
        def pot(Xf,a1,k1,a2,k2,a3,k3):
            s3,sd = Xf[:,0],Xf[:,1]
            return a1*s3**k1 + a2*(s3*sd)**k2 + a3*sd**k3

        popt, _ = curve_fit(pot, X, y,
            p0=[y.mean()/X[:,0].mean(),1,
                y.mean()/(X[:,0]*X[:,1]).mean(),1,
                y.mean()/X[:,1].mean(),1],
            maxfev=200000)
        y_pred = pot(X, *popt)

        r2     = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt)) if len(y) > len(popt)+1 else r2
        rmse   = np.sqrt(mean_squared_error(y, y_pred))
        mae    = mean_absolute_error(y, y_pred)

        eq_latex = (f"$$MR = {popt[0]:.4f}σ₃^{{{popt[1]:.4f}}} + "
                    f"{popt[2]:.4f}(σ₃σ_d)^{{{popt[3]:.4f}}} + "
                    f"{popt[4]:.4f}σ_d^{{{popt[5]:.4f}}}$$")

        result.update({
            "eq_latex":    eq_latex,
            "intercept":   0.0,
            "r2":          r2,
            "r2_adj":      r2_adj,
            "rmse":        rmse,
            "mae":         mae,
            "mean_MR":     y.mean(),
            "std_MR":      y.std(),
            "model_obj":   pot,
            "poly_obj":    None,
            "is_power":    True,
            "power_params": popt
        })

    # — Pezo —
    else:
        def pezo(Xf,k1,k2,k3):
            Pa=0.101325
            s3,sd = Xf[:,0],Xf[:,1]
            return k1*Pa*(s3/Pa)**k2*(sd/Pa)**k3

        popt, _ = curve_fit(pezo, X, y,
            p0=[y.mean()/(0.101325*(X[:,0]/0.101325).mean()*
                     (X[:,1]/0.101325).mean()),1,1],
            maxfev=200000)
        y_pred = pezo(X, *popt)

        r2     = r2_score(y, y_pred)
        r2_adj = adjusted_r2(r2, len(y), len(popt)) if len(y) > len(popt)+1 else r2
        rmse   = np.sqrt(mean_squared_error(y, y_pred))
        mae    = mean_absolute_error(y, y_pred)

        eq_latex = (f"$$MR = {popt[0]*0.101325:.4f}"
                    f"(σ₃/0.101325)^{{{popt[1]:.4f}}}"
                    f"(σ_d/0.101325)^{{{popt[2]:.4f}}}$$")

        result.update({
            "eq_latex":    eq_latex,
            "intercept":   0.0,
            "r2":          r2,
            "r2_adj":      r2_adj,
            "rmse":        rmse,
            "mae":         mae,
            "mean_MR":     y.mean(),
            "std_MR":      y.std(),
            "model_obj":   pezo,
            "poly_obj":    None,
            "is_power":    True,
            "power_params": popt
        })

    result["quality"] = evaluate_quality(y, result["rmse"], result["mae"])
    return result

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
    fig.add_trace(go.Scatter3d(
        x=df["σ3"], y=df["σd"], z=df[energy_col],
        mode='markers', marker=dict(size=5, color='red'), name='Dados'
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title=r"$\sigma_3$ (MPa)",
            yaxis_title=r"$\sigma_d$ (MPa)",
            zaxis_title="MR (MPa)"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

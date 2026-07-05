import numpy as np
import pandas as pd

from models import MODELS_MAP, Pezo1993NonNormalizedModel
from utils.metrics import calculate_metrics
from utils.fitting import count_model_parameters, fit_logarithmic_model

PHYSICAL_PA = 0.101325

# Modelos do grupo "gerais" ficam fora da busca pelo melhor modelo testado.
GENERAL_MODELS = {
    "Polinomial c/ Intercepto",
    "Polinomial s/Intercepto",
    "Potência Composta (Genérico)",
}

# Rótulos dos métodos de ajuste testados por modelo.
FIT_METHODS = ("natural", "log")


def resolve_pa(model, pa_value):
    """Pa físico para modelos com termo '+1'; caso contrário, o Pa escolhido."""
    if getattr(model, "requires_physical_pa", False):
        return PHYSICAL_PA
    return pa_value


def _candidate_factories():
    """(rótulo, factory) dos modelos testados + variante não normalizada do Pezo."""
    candidates = [
        (name, factory)
        for name, factory in MODELS_MAP.items()
        if name not in GENERAL_MODELS
    ]
    candidates.append(("Pezo (1993) - Não normalizada", Pezo1993NonNormalizedModel))
    return candidates


def _n_params(model):
    if getattr(model, "_params", None) is not None:
        return len(model._params)
    if getattr(model, "_coefs", None) is not None:
        return len(model._coefs)
    return 0


def _fit_one(factory, method, X, y, pa_value):
    """Ajusta uma instância nova pelo método dado. Retorna dict ou None se falhar."""
    model = factory()
    if hasattr(model, "Pa"):
        model.Pa = resolve_pa(model, pa_value)
    try:
        if method == "log":
            fit_logarithmic_model(model, X, y)
        else:
            model.fit(X, y)
        y_pred = model.predict(X)
    except Exception:
        return None
    if not np.all(np.isfinite(y_pred)):
        return None
    metrics = calculate_metrics(y, y_pred, _n_params(model))
    if not np.isfinite(metrics.get("r2_adj", np.nan)):
        return None
    return {
        "model": model,
        "metrics": metrics,
        "method": method,
        "pa": resolve_pa(model, pa_value),
        "y_pred": y_pred,
    }


def evaluate_all_models(X, y, pa_value):
    """Ajusta todos os modelos testados (ambos os métodos, mantendo o melhor de
    cada), remove duplicatas e ordena por R² ajustado (desc), RMSE (asc)."""
    results = []
    seen = set()
    for name, factory in _candidate_factories():
        fits = [f for f in (_fit_one(factory, m, X, y, pa_value) for m in FIT_METHODS) if f]
        if not fits:
            continue
        best = max(fits, key=lambda f: (f["metrics"]["r2_adj"], -f["metrics"]["rmse"]))

        # dedupe: modelos que produzem a MESMA predição são matematicamente
        # idênticos para o ranking (ex.: Ooi (2) ≡ NCHRP 1-37A).
        dedupe_key = tuple(np.round(best["y_pred"], 6).tolist())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        best["name"] = name
        best["coefficients"] = best["model"].get_coefficients()
        results.append(best)

    results.sort(key=lambda r: (-r["metrics"]["r2_adj"], r["metrics"]["rmse"]))
    return results


def _format_coefficients(coefficients):
    return "; ".join(f"{label}={value:.4f}" for label, value in coefficients)


def build_ranking_dataframe(results):
    """DataFrame formatado (strings) para exibição no app e nos relatórios."""
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Modelo": r["name"],
            "Coeficientes": _format_coefficients(r["coefficients"]),
            "R²": f"{m['r2']:.4f}",
            "R²aj": f"{m['r2_adj']:.4f}",
            "RMSE (MPa)": f"{m['rmse']:.4f}",
            "MAE (MPa)": f"{m['mae']:.4f}",
            "NRMSE": f"{m['nrmse_range']:.2%}",
            "CV(RMSE)": f"{m['cv_rmse']:.2%}",
            "MAE%": f"{m['mae_pct']:.2%}",
            "Método": r["method"],
            "Pa": f"{r['pa']:g}",
        })
    return pd.DataFrame(rows)

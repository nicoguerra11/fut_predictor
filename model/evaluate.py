"""
evaluate.py
Evaluación del modelo: LOO-CV, Ranked Probability Score y calibration plot.
"""

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

from model.train import load_matches, build_team_index, prepare_arrays, build_model

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
POSTERIOR_PATH = MODEL_DIR / "posterior.nc"
REPORTS_DIR = BASE_DIR / "reports"


def loo_cv(idata: az.InferenceData | None = None) -> az.ELPDData:
    """
    Calcula Leave-One-Out Cross-Validation usando ArviZ (PSIS-LOO).

    Args:
        idata: InferenceData ya cargado. Si None, lo carga desde disco.

    Returns:
        ELPDData con las métricas LOO.
    """
    if idata is None:
        if not POSTERIOR_PATH.exists():
            raise FileNotFoundError(f"No se encontró posterior en {POSTERIOR_PATH}")
        idata = az.from_netcdf(str(POSTERIOR_PATH))

    log.info("Calculando PSIS-LOO...")
    loo_result = az.loo(idata, pointwise=True)
    log.info(f"LOO-CV:\n{loo_result}")

    # Advertir sobre observaciones de alto k_hat (diagnóstico de Pareto)
    if hasattr(loo_result, "pareto_k"):
        k_hat = loo_result.pareto_k
        bad_k = (k_hat > 0.7).sum()
        if bad_k > 0:
            log.warning(
                f"{bad_k} observaciones con k̂ > 0.7 — el modelo puede tener "
                "problemas con esos partidos (posiblemente resultados atípicos)."
            )
    return loo_result


def ranked_probability_score(
    probs: np.ndarray, outcomes: np.ndarray
) -> float:
    """
    Calcula el Ranked Probability Score (RPS) para predicciones de 3 resultados.

    Args:
        probs: Array (n, 3) con probabilidades [P(local), P(empate), P(visitante)].
        outcomes: Array (n,) con resultado real: 0=local, 1=empate, 2=visitante.

    Returns:
        RPS medio (menor es mejor).
    """
    n = len(outcomes)
    cumprobs = np.cumsum(probs, axis=1)  # (n, 3)

    # One-hot del resultado real
    one_hot = np.zeros((n, 3))
    one_hot[np.arange(n), outcomes] = 1.0
    cum_outcomes = np.cumsum(one_hot, axis=1)

    rps_per_match = np.mean((cumprobs - cum_outcomes) ** 2, axis=1)
    return float(np.mean(rps_per_match))


def _naive_baseline_probs(df: pd.DataFrame) -> np.ndarray:
    """
    Baseline naive: distribución empírica global de resultados
    (misma probabilidad para todos los partidos).
    """
    total = len(df)
    p_home = float((df["goles_local"] > df["goles_visitante"]).sum() / total)
    p_draw = float((df["goles_local"] == df["goles_visitante"]).sum() / total)
    p_away = float((df["goles_local"] < df["goles_visitante"]).sum() / total)
    return np.tile([p_home, p_draw, p_away], (total, 1))


def evaluate_rps(
    idata: az.InferenceData | None = None,
    data_path: Path | None = None,
) -> dict[str, float]:
    """
    Compara el RPS del modelo bayesiano contra el baseline naive.

    Returns:
        Dict con 'rps_modelo' y 'rps_baseline'.
    """
    if idata is None:
        if not POSTERIOR_PATH.exists():
            raise FileNotFoundError(f"No se encontró posterior en {POSTERIOR_PATH}")
        idata = az.from_netcdf(str(POSTERIOR_PATH))

    df = load_matches(data_path)
    teams, team_to_idx = build_team_index(df)
    arrays = prepare_arrays(df, team_to_idx)

    # Extraer goles simulados de la posterior predictiva
    ppc = idata.posterior_predictive
    n_chains = ppc.sizes["chain"]
    n_draws = ppc.sizes["draw"]

    goals_home_ppc = ppc["goles_local_obs"].values.reshape(n_chains * n_draws, -1)
    goals_away_ppc = ppc["goles_visitante_obs"].values.reshape(n_chains * n_draws, -1)

    # Para cada partido, calcular P(local), P(empate), P(visitante) via simulaciones
    p_home_model = np.mean(goals_home_ppc > goals_away_ppc, axis=0)
    p_draw_model = np.mean(goals_home_ppc == goals_away_ppc, axis=0)
    p_away_model = np.mean(goals_home_ppc < goals_away_ppc, axis=0)

    probs_model = np.column_stack([p_home_model, p_draw_model, p_away_model])

    # Resultado real
    outcomes = np.where(
        arrays["goals_home"] > arrays["goals_away"], 0,
        np.where(arrays["goals_home"] == arrays["goals_away"], 1, 2),
    )

    rps_modelo = ranked_probability_score(probs_model, outcomes)
    rps_baseline = ranked_probability_score(_naive_baseline_probs(df), outcomes)

    log.info(f"RPS modelo:    {rps_modelo:.4f}")
    log.info(f"RPS baseline:  {rps_baseline:.4f}")
    log.info(f"Mejora vs baseline: {(rps_baseline - rps_modelo) / rps_baseline:.1%}")

    return {"rps_modelo": rps_modelo, "rps_baseline": rps_baseline}


def calibration_plot(
    idata: az.InferenceData | None = None,
    data_path: Path | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Genera el calibration plot: probabilidades predichas vs frecuencia observada.
    Divide las predicciones en 10 bins y compara con la línea perfecta.
    """
    if idata is None:
        if not POSTERIOR_PATH.exists():
            raise FileNotFoundError(f"No se encontró posterior en {POSTERIOR_PATH}")
        idata = az.from_netcdf(str(POSTERIOR_PATH))

    df = load_matches(data_path)
    teams, team_to_idx = build_team_index(df)
    arrays = prepare_arrays(df, team_to_idx)

    ppc = idata.posterior_predictive
    n_chains = ppc.sizes["chain"]
    n_draws = ppc.sizes["draw"]

    goals_home_ppc = ppc["goles_local_obs"].values.reshape(n_chains * n_draws, -1)
    goals_away_ppc = ppc["goles_visitante_obs"].values.reshape(n_chains * n_draws, -1)

    p_home = np.mean(goals_home_ppc > goals_away_ppc, axis=0)
    actual_home_win = (arrays["goals_home"] > arrays["goals_away"]).astype(float)

    # Binning en deciles
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(p_home, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 9)

    observed_freq = np.array(
        [
            actual_home_win[bin_indices == b].mean() if (bin_indices == b).sum() > 0 else np.nan
            for b in range(10)
        ]
    )
    counts = np.array([(bin_indices == b).sum() for b in range(10)])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibración perfecta")

    valid = ~np.isnan(observed_freq)
    scatter = ax.scatter(
        bin_centers[valid],
        observed_freq[valid],
        s=counts[valid] * 8,
        c=counts[valid],
        cmap="viridis",
        zorder=5,
        label="Bins (tamaño = n partidos)",
    )
    plt.colorbar(scatter, ax=ax, label="Partidos en el bin")

    ax.set_xlabel("Probabilidad predicha P(victoria local)", fontsize=12)
    ax.set_ylabel("Frecuencia observada", fontsize=12)
    ax.set_title("Calibration Plot — Modelo Poisson Jerárquico Bayesiano", fontsize=13)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        log.info(f"Calibration plot guardado en {save_path}")

    return fig


def full_evaluation(
    idata: az.InferenceData | None = None,
    data_path: Path | None = None,
) -> dict:
    """
    Ejecuta todas las evaluaciones y retorna un resumen.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Evaluación completa del modelo ===")

    loo_result = loo_cv(idata)
    rps_scores = evaluate_rps(idata, data_path)
    calibration_plot(
        idata,
        data_path,
        save_path=REPORTS_DIR / "calibration_plot.png",
    )

    return {
        "loo_elpd": float(loo_result.elpd_loo),
        "loo_se": float(loo_result.se),
        **rps_scores,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resultados = full_evaluation()
    print("\n=== Resultados de evaluación ===")
    for k, v in resultados.items():
        print(f"  {k}: {v:.4f}")

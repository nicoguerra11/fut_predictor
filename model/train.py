"""
train.py
Entrena el modelo de Poisson jerárquico bayesiano para predicción de partidos
del Mundial 2026 usando PyMC 5.

Modelo:
    goles_ij ~ Poisson(λ_ij)
    log(λ_ij) = μ + ataque_i - defensa_j + α·Δranking_ij + γ·es_local

    Δranking_ij = (ranking_rival - ranking_propio) / 50
"""

import logging
import os
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model"

POSTERIOR_PATH = MODEL_DIR / "posterior.nc"
TEAMS_PATH = MODEL_DIR / "teams.csv"

# Hiperparámetros del sampler — configurables via env vars para free tier
DRAWS = int(os.getenv("MCMC_DRAWS", "2000"))
TUNE = int(os.getenv("MCMC_TUNE", "1000"))
CHAINS = int(os.getenv("MCMC_CHAINS", "4"))
TARGET_ACCEPT = 0.9


def load_matches(path: Path | None = None) -> pd.DataFrame:
    """Carga y valida el dataset de partidos procesado."""
    csv_path = path or PROCESSED_DIR / "matches.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Ejecutá primero data/fetch_data.py"
        )

    df = pd.read_csv(csv_path)
    required = {
        "equipo_local", "equipo_visitante",
        "goles_local", "goles_visitante",
        "ranking_local", "ranking_visitante",
        "fase",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")

    # Filtrar solo partidos finalizados (con marcador entero)
    df = df.dropna(subset=["goles_local", "goles_visitante"]).copy()
    df["goles_local"] = df["goles_local"].astype(int)
    df["goles_visitante"] = df["goles_visitante"].astype(int)

    log.info(f"Dataset cargado: {len(df)} partidos, {df['equipo_local'].nunique()} equipos")
    return df


def build_team_index(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Construye el índice de equipos únicos ordenado alfabéticamente."""
    all_teams = sorted(
        set(df["equipo_local"].unique()) | set(df["equipo_visitante"].unique())
    )
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    return all_teams, team_to_idx


def prepare_arrays(
    df: pd.DataFrame, team_to_idx: dict[str, int]
) -> dict[str, np.ndarray]:
    """Convierte el DataFrame a arrays numpy para PyMC."""
    home_idx = df["equipo_local"].map(team_to_idx).to_numpy()
    away_idx = df["equipo_visitante"].map(team_to_idx).to_numpy()

    # Δranking normalizado: (ranking_rival - ranking_propio) / 50
    delta_rank_home = (
        (df["ranking_visitante"] - df["ranking_local"]) / 50.0
    ).to_numpy()
    delta_rank_away = (
        (df["ranking_local"] - df["ranking_visitante"]) / 50.0
    ).to_numpy()

    # Ventaja de local: solo aplica en fase de grupos
    is_home_advantage = (df["fase"] == "grupos").astype(float).to_numpy()

    goals_home = df["goles_local"].to_numpy()
    goals_away = df["goles_visitante"].to_numpy()

    return {
        "home_idx": home_idx,
        "away_idx": away_idx,
        "delta_rank_home": delta_rank_home,
        "delta_rank_away": delta_rank_away,
        "is_home_advantage": is_home_advantage,
        "goals_home": goals_home,
        "goals_away": goals_away,
    }


def build_model(arrays: dict[str, np.ndarray], n_teams: int) -> pm.Model:
    """
    Construye el modelo PyMC.

    Priors:
        μ ~ Normal(0, 1)               — media global de goles (log scale)
        α ~ Normal(0, 0.5)             — efecto del ranking FIFA
        γ ~ Normal(0, 1)               — ventaja de local (solo grupos)
        σ_ataque ~ HalfNormal(1)
        σ_defensa ~ HalfNormal(1)
        ataque_i ~ Normal(0, σ_ataque) — efecto jerárquico de ataque por selección
        defensa_i ~ Normal(0, σ_defensa) — efecto jerárquico de defensa por selección
    """
    with pm.Model() as model:
        # ── Hiperpriors ────────────────────────────────────────────────────────
        sigma_ataque = pm.HalfNormal("sigma_ataque", sigma=1.0)
        sigma_defensa = pm.HalfNormal("sigma_defensa", sigma=1.0)

        # ── Priors globales ────────────────────────────────────────────────────
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        alpha = pm.Normal("alpha", mu=0.0, sigma=0.5)
        gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)

        # ── Efectos por equipo (jerárquico) ────────────────────────────────────
        ataque = pm.Normal("ataque", mu=0.0, sigma=sigma_ataque, shape=n_teams)
        defensa = pm.Normal("defensa", mu=0.0, sigma=sigma_defensa, shape=n_teams)

        # ── Función de enlace log ──────────────────────────────────────────────
        # log(λ_local) = μ + ataque_local - defensa_visitante + α·Δranking_local + γ·es_local
        log_lambda_home = (
            mu
            + ataque[arrays["home_idx"]]
            - defensa[arrays["away_idx"]]
            + alpha * arrays["delta_rank_home"]
            + gamma * arrays["is_home_advantage"]
        )
        # log(λ_visitante) = μ + ataque_visitante - defensa_local + α·Δranking_visitante
        log_lambda_away = (
            mu
            + ataque[arrays["away_idx"]]
            - defensa[arrays["home_idx"]]
            + alpha * arrays["delta_rank_away"]
        )

        lambda_home = pm.math.exp(log_lambda_home)
        lambda_away = pm.math.exp(log_lambda_away)

        # ── Verosimilitud ──────────────────────────────────────────────────────
        pm.Poisson("goles_local_obs", mu=lambda_home, observed=arrays["goals_home"])
        pm.Poisson("goles_visitante_obs", mu=lambda_away, observed=arrays["goals_away"])

    return model


def sample_posterior(model: pm.Model) -> az.InferenceData:
    """Ejecuta MCMC con NUTS y retorna el InferenceData."""
    log.info(f"Iniciando sampling: {CHAINS} cadenas × {DRAWS} draws (tune={TUNE})")
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            return_inferencedata=True,
            progressbar=True,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    return idata


def save_artifacts(
    idata: az.InferenceData, teams: list[str], posterior_path: Path | None = None
) -> None:
    """Serializa la posterior (NetCDF) y el índice de equipos (CSV)."""
    out_path = posterior_path or POSTERIOR_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idata.to_netcdf(str(out_path))
    log.info(f"Posterior guardada en {out_path}")

    teams_df = pd.DataFrame({"equipo": teams, "idx": range(len(teams))})
    teams_df.to_csv(TEAMS_PATH, index=False)
    log.info(f"Índice de equipos guardado en {TEAMS_PATH}")


def train(data_path: Path | None = None, posterior_path: Path | None = None) -> az.InferenceData:
    """Entrena el modelo completo y guarda los artefactos."""
    df = load_matches(data_path)
    teams, team_to_idx = build_team_index(df)
    arrays = prepare_arrays(df, team_to_idx)
    model = build_model(arrays, n_teams=len(teams))
    idata = sample_posterior(model)

    # Guardar nombre de equipos como coordenada en el InferenceData
    idata.posterior = idata.posterior.assign_coords({"teams": ("teams", teams)})

    save_artifacts(idata, teams, posterior_path)

    # Diagnósticos básicos
    summary = az.summary(idata, var_names=["mu", "alpha", "gamma", "sigma_ataque", "sigma_defensa"])
    log.info(f"\nResumen de parámetros globales:\n{summary}")

    r_hat_max = float(summary["r_hat"].max())
    if r_hat_max > 1.05:
        log.warning(f"R̂ máximo = {r_hat_max:.3f} > 1.05 — considerar más muestras o reparametrización")
    else:
        log.info(f"Convergencia OK (R̂ máximo = {r_hat_max:.3f})")

    return idata


def retrain(new_data_path: Path | None = None) -> az.InferenceData:
    """
    Rreentrena el modelo con datos actualizados.
    Útil cuando hay nuevos resultados del Mundial 2026.
    Hace exactamente lo mismo que train() pero con un mensaje de log claro.
    """
    log.info("Iniciando reentrenamiento con datos actualizados...")
    return train(data_path=new_data_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    idata = train()
    print("\n✓ Modelo entrenado exitosamente")
    print(f"  Posterior guardada en: {POSTERIOR_PATH}")
    print(f"  Equipos indexados en: {TEAMS_PATH}")

"""
train.py
Entrena el modelo de Poisson jerárquico para predicción de partidos.

Modelo:
    goles_ij ~ Poisson(λ_ij)
    log(λ_ij) = μ + ataque_i - defensa_j + α·Δranking_ij + γ·es_local

Método: MAP con L-BFGS-B + bootstrap para cuantificar incertidumbre.
Sin compilación C — funciona en cualquier servidor.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model"

POSTERIOR_PATH = MODEL_DIR / "posterior.npz"
TEAMS_PATH = MODEL_DIR / "teams.csv"

N_BOOTSTRAP = int(os.getenv("BOOTSTRAP_SAMPLES", "1000"))


def load_matches(path: Path | None = None) -> pd.DataFrame:
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
    df = df.dropna(subset=["goles_local", "goles_visitante"]).copy()
    df["goles_local"] = df["goles_local"].astype(int)
    df["goles_visitante"] = df["goles_visitante"].astype(int)
    log.info(f"Dataset cargado: {len(df)} partidos, {df['equipo_local'].nunique()} equipos")
    return df


def build_team_index(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    all_teams = sorted(set(df["equipo_local"]) | set(df["equipo_visitante"]))
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    return all_teams, team_to_idx


def build_arrays(
    df: pd.DataFrame, team_to_idx: dict[str, int]
) -> tuple[np.ndarray, ...]:
    home_idx = df["equipo_local"].map(team_to_idx).to_numpy()
    away_idx = df["equipo_visitante"].map(team_to_idx).to_numpy()
    delta_home = ((df["ranking_visitante"] - df["ranking_local"]) / 50.0).to_numpy()
    delta_away = ((df["ranking_local"] - df["ranking_visitante"]) / 50.0).to_numpy()
    is_home = (df["fase"] == "grupos").astype(float).to_numpy()
    goals_home = df["goles_local"].to_numpy()
    goals_away = df["goles_visitante"].to_numpy()
    return home_idx, away_idx, delta_home, delta_away, is_home, goals_home, goals_away


def neg_log_posterior(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    delta_home: np.ndarray,
    delta_away: np.ndarray,
    is_home: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
) -> float:
    """Negativo del log-posterior (log-verosimilitud Poisson + prior gaussiano)."""
    mu = params[0]
    alpha = params[1]
    gamma = params[2]
    ataque = params[3: 3 + n_teams]
    defensa = params[3 + n_teams: 3 + 2 * n_teams]

    # Prior gaussiano (regularización L2)
    log_prior = -0.5 * (
        mu**2
        + (alpha / 0.5) ** 2
        + gamma**2
        + np.sum(ataque**2)
        + np.sum(defensa**2)
    )

    # Log-verosimilitud Poisson (log(λ) versión estable)
    log_lam_h = (
        mu + ataque[home_idx] - defensa[away_idx]
        + alpha * delta_home + gamma * is_home
    )
    log_lam_a = (
        mu + ataque[away_idx] - defensa[home_idx]
        + alpha * delta_away
    )
    log_lam_h = np.clip(log_lam_h, -10, 10)
    log_lam_a = np.clip(log_lam_a, -10, 10)

    ll = np.sum(goals_home * log_lam_h - np.exp(log_lam_h))
    ll += np.sum(goals_away * log_lam_a - np.exp(log_lam_a))

    return -(log_prior + ll)


def fit_map(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    delta_home: np.ndarray,
    delta_away: np.ndarray,
    is_home: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    n_teams: int,
) -> np.ndarray:
    """Encuentra el MAP del modelo con L-BFGS-B."""
    n_params = 3 + 2 * n_teams
    x0 = np.zeros(n_params)
    x0[0] = np.log(max(1.0, np.mean(np.concatenate([goals_home, goals_away]))))

    result = minimize(
        neg_log_posterior,
        x0=x0,
        args=(n_teams, home_idx, away_idx, delta_home, delta_away, is_home, goals_home, goals_away),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        log.debug(f"Optimización: {result.message}")
    return result.x


def bootstrap_posterior(
    df: pd.DataFrame,
    team_to_idx: dict[str, int],
    n_teams: int,
    n_bootstrap: int,
) -> np.ndarray:
    """Bootstrap paramétrico — cada muestra es un MAP sobre un resampleo del dataset."""
    rng = np.random.default_rng(42)
    samples: list[np.ndarray] = []
    failed = 0

    for i in range(n_bootstrap):
        idxs = rng.integers(0, len(df), size=len(df))
        boot_df = df.iloc[idxs]
        arrays = build_arrays(boot_df, team_to_idx)
        try:
            params = fit_map(*arrays, n_teams)
            samples.append(params)
        except Exception:
            failed += 1
        if (i + 1) % 200 == 0:
            log.info(f"  Bootstrap: {i + 1}/{n_bootstrap} ({failed} fallidos)")

    log.info(f"Bootstrap completo: {len(samples)} muestras válidas ({failed} fallidas)")
    return np.array(samples)


def save_artifacts(
    samples: np.ndarray,
    teams: list[str],
    posterior_path: Path | None = None,
) -> None:
    out_path = posterior_path or POSTERIOR_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_teams = len(teams)
    np.savez_compressed(
        str(out_path),
        mu=samples[:, 0],
        alpha=samples[:, 1],
        gamma=samples[:, 2],
        ataque=samples[:, 3: 3 + n_teams],
        defensa=samples[:, 3 + n_teams: 3 + 2 * n_teams],
    )
    log.info(f"Posterior guardada en {out_path} ({len(samples)} muestras bootstrap)")

    teams_df = pd.DataFrame({"equipo": teams, "idx": range(n_teams)})
    teams_df.to_csv(TEAMS_PATH, index=False)
    log.info(f"Equipos indexados en {TEAMS_PATH}")


def train(
    data_path: Path | None = None,
    posterior_path: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    df = load_matches(data_path)
    teams, team_to_idx = build_team_index(df)
    n_teams = len(teams)

    log.info("Ajustando MAP con todos los datos...")
    all_arrays = build_arrays(df, team_to_idx)
    map_params = fit_map(*all_arrays, n_teams)
    log.info(f"  μ={map_params[0]:.3f}  α={map_params[1]:.3f}  γ={map_params[2]:.3f}")

    log.info(f"Iniciando bootstrap ({N_BOOTSTRAP} muestras)...")
    samples = bootstrap_posterior(df, team_to_idx, n_teams, N_BOOTSTRAP)

    save_artifacts(samples, teams, posterior_path)
    return samples, teams


def retrain(new_data_path: Path | None = None) -> tuple[np.ndarray, list[str]]:
    log.info("Iniciando reentrenamiento con datos actualizados...")
    return train(data_path=new_data_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples, teams = train()
    print(f"\n✓ Modelo entrenado: {len(samples)} muestras bootstrap, {len(teams)} equipos")
    print(f"  Posterior: {POSTERIOR_PATH}")
    print(f"  Equipos:   {TEAMS_PATH}")

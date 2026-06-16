"""
predict.py
Carga la posterior entrenada y genera predicciones para un partido dado.

Uso:
    from model.predict import Predictor
    pred = Predictor()
    resultado = pred.predict("Argentina", "Brasil", es_local=True)
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
POSTERIOR_PATH = MODEL_DIR / "posterior.npz"
TEAMS_PATH = MODEL_DIR / "teams.csv"

N_SIMULATIONS = 4000
CREDIBLE_INTERVAL = 0.89  # 89% como recomienda McElreath


@dataclass
class MatchPrediction:
    """Resultado completo de la predicción para un partido."""

    team_home: str
    team_away: str
    prob_home: float
    prob_draw: float
    prob_away: float
    expected_goals_home: float
    expected_goals_away: float
    credible_interval_home: tuple[int, int]
    credible_interval_away: tuple[int, int]
    most_likely_score_home: int
    most_likely_score_away: int
    goals_dist_home: list[float]
    goals_dist_away: list[float]
    scorelines: list[dict]


class Predictor:
    """
    Predictor de partidos basado en la posterior bayesiana entrenada.
    Carga los artefactos una sola vez en memoria (patrón singleton por instancia).
    """

    def __init__(
        self,
        posterior_path: Path | None = None,
        teams_path: Path | None = None,
    ) -> None:
        self._posterior_path = posterior_path or POSTERIOR_PATH
        self._teams_path = teams_path or TEAMS_PATH
        self._samples: dict[str, np.ndarray] | None = None
        self._team_to_idx: dict[str, int] = {}
        self._teams: list[str] = []
        self._default_idx: int = 0

    def _load(self) -> None:
        """Carga lazy de la posterior y el índice de equipos."""
        if self._samples is not None:
            return

        if not self._posterior_path.exists():
            raise FileNotFoundError(
                f"No se encontró la posterior en {self._posterior_path}. "
                "Ejecutá model/train.py primero."
            )
        if not self._teams_path.exists():
            raise FileNotFoundError(
                f"No se encontró el índice de equipos en {self._teams_path}."
            )

        log.info(f"Cargando posterior desde {self._posterior_path}...")
        data = np.load(str(self._posterior_path))
        self._samples = {
            "mu": data["mu"],
            "alpha": data["alpha"],
            "gamma": data["gamma"],
            "ataque": data["ataque"],
            "defensa": data["defensa"],
        }

        teams_df = pd.read_csv(self._teams_path)
        self._teams = teams_df["equipo"].tolist()
        self._team_to_idx = dict(zip(teams_df["equipo"], teams_df["idx"]))
        self._default_idx = len(self._teams) // 2
        n_samples = len(data["mu"])
        log.info(f"Posterior cargada: {len(self._teams)} equipos, {n_samples} muestras bootstrap")

    def get_teams(self) -> list[str]:
        """Retorna la lista de equipos conocidos por el modelo."""
        self._load()
        return list(self._teams)

    def _get_idx(self, team: str) -> int:
        """Retorna el índice del equipo, usando la media global si es desconocido."""
        idx = self._team_to_idx.get(team)
        if idx is None:
            log.warning(
                f"Equipo '{team}' no encontrado en el índice. "
                "Usando prior global (media del índice)."
            )
            return self._default_idx
        return idx

    def _get_samples(self) -> dict[str, np.ndarray]:
        """Retorna una copia de las muestras de la posterior."""
        assert self._samples is not None
        return dict(self._samples)

    def predict(
        self,
        team_home: str,
        team_away: str,
        es_local: bool = True,
        ranking_home: int | None = None,
        ranking_away: int | None = None,
        n_simulations: int = N_SIMULATIONS,
    ) -> MatchPrediction:
        """
        Simula un partido desde la posterior predictiva.

        Args:
            team_home: Nombre del equipo local (o equipo A en partido neutral).
            team_away: Nombre del equipo visitante (o equipo B).
            es_local: True si team_home juega en casa (aplica ventaja local).
                      False para partidos en sede neutral.
            ranking_home: Ranking FIFA del equipo local (opcional; si None usa mediana).
            ranking_away: Ranking FIFA del equipo visitante (opcional).
            n_simulations: Número de simulaciones desde la posterior.

        Returns:
            MatchPrediction con probabilidades, goles esperados e intervalos de credibilidad.
        """
        self._load()

        home_idx = self._get_idx(team_home)
        away_idx = self._get_idx(team_away)

        samples = self._get_samples()

        # Si tenemos menos muestras que n_simulations, samplear con reemplazo
        n_total = len(samples["mu"])
        rng = np.random.default_rng(seed=42)
        if n_simulations != n_total:
            idxs = rng.choice(n_total, size=n_simulations, replace=True)
            samples = {k: v[idxs] if v.ndim == 1 else v[idxs] for k, v in samples.items()}

        # Δranking normalizado
        r_home = ranking_home or 50
        r_away = ranking_away or 50
        delta_home = (r_away - r_home) / 50.0
        delta_away = (r_home - r_away) / 50.0
        home_advantage = 1.0 if es_local else 0.0

        # log(λ) para cada muestra
        log_lambda_home = (
            samples["mu"]
            + samples["ataque"][:, home_idx]
            - samples["defensa"][:, away_idx]
            + samples["alpha"] * delta_home
            + samples["gamma"] * home_advantage
        )
        log_lambda_away = (
            samples["mu"]
            + samples["ataque"][:, away_idx]
            - samples["defensa"][:, home_idx]
            + samples["alpha"] * delta_away
        )

        lambda_home = np.exp(np.clip(log_lambda_home, -10, 10))
        lambda_away = np.exp(np.clip(log_lambda_away, -10, 10))

        # Simular goles desde Poisson
        rng2 = np.random.default_rng(seed=0)
        simulated_home = rng2.poisson(lambda_home)
        simulated_away = rng2.poisson(lambda_away)

        # Probabilidades de resultado
        prob_home = float(np.mean(simulated_home > simulated_away))
        prob_draw = float(np.mean(simulated_home == simulated_away))
        prob_away = float(np.mean(simulated_home < simulated_away))

        # Goles esperados
        expected_home = float(np.mean(simulated_home))
        expected_away = float(np.mean(simulated_away))

        # Intervalo de credibilidad 89%
        alpha_tail = (1.0 - CREDIBLE_INTERVAL) / 2
        ci_home = (
            int(np.quantile(simulated_home, alpha_tail)),
            int(np.quantile(simulated_home, 1 - alpha_tail)),
        )
        ci_away = (
            int(np.quantile(simulated_away, alpha_tail)),
            int(np.quantile(simulated_away, 1 - alpha_tail)),
        )

        # Resultado más probable (moda conjunta)
        from collections import Counter
        score_counts = Counter(zip(simulated_home.tolist(), simulated_away.tolist()))
        most_likely = score_counts.most_common(1)[0][0]

        # Distribución de goles (0 a 8 para la UI)
        max_goals = 9
        dist_home = [float(np.mean(simulated_home == g)) for g in range(max_goals)]
        dist_away = [float(np.mean(simulated_away == g)) for g in range(max_goals)]

        # Grid de probabilidades 6×6 (marcadores exactos 0-0 a 5-5)
        goals_range = np.arange(6)
        pmf_home_grid = scipy_poisson.pmf(goals_range[None, :], lambda_home[:, None])
        pmf_away_grid = scipy_poisson.pmf(goals_range[None, :], lambda_away[:, None])
        grid = np.einsum("ni,nj->ij", pmf_home_grid, pmf_away_grid) / len(lambda_home)
        total_grid = grid.sum()
        scorelines = sorted(
            [
                {"home": int(i), "away": int(j), "prob": round(float(grid[i, j] / total_grid), 4)}
                for i in range(6)
                for j in range(6)
            ],
            key=lambda x: -x["prob"],
        )

        return MatchPrediction(
            team_home=team_home,
            team_away=team_away,
            prob_home=round(prob_home, 4),
            prob_draw=round(prob_draw, 4),
            prob_away=round(prob_away, 4),
            expected_goals_home=round(expected_home, 2),
            expected_goals_away=round(expected_away, 2),
            credible_interval_home=ci_home,
            credible_interval_away=ci_away,
            most_likely_score_home=most_likely[0],
            most_likely_score_away=most_likely[1],
            goals_dist_home=dist_home,
            goals_dist_away=dist_away,
            scorelines=scorelines,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = Predictor()
    result = predictor.predict("Argentina", "Brasil", es_local=True)
    print(f"\n{result.team_home} vs {result.team_away}")
    print(f"  P(local)  = {result.prob_home:.1%}")
    print(f"  P(empate) = {result.prob_draw:.1%}")
    print(f"  P(visit.) = {result.prob_away:.1%}")
    print(f"  Goles esperados: {result.expected_goals_home:.2f} – {result.expected_goals_away:.2f}")
    print(f"  Resultado más probable: {result.most_likely_score_home}-{result.most_likely_score_away}")

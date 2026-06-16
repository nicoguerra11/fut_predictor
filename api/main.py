"""
main.py
API FastAPI para predicción de partidos del Mundial 2026.
La posterior se carga una sola vez al arrancar (lifespan event).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    StandingsEntry,
    StandingsResponse,
    TeamsResponse,
)
from model.predict import Predictor

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Grupos del Mundial 2026 — sorteo oficial 11 de junio de 2026 ─────────────
WC_2026_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


# ── Estado global compartido ──────────────────────────────────────────────────
class AppState:
    predictor: Predictor | None = None
    model_loaded: bool = False
    last_trained: str = "build"  # timestamp ISO o "build" si viene del deploy
    retrain_running: bool = False


state = AppState()
_retrain_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga la posterior al arrancar; libera al apagar."""
    log.info("Iniciando API — cargando modelo bayesiano...")
    predictor = Predictor()
    try:
        predictor._load()
        state.predictor = predictor
        state.model_loaded = True
        log.info("Modelo cargado correctamente")
    except FileNotFoundError as exc:
        log.warning(f"Posterior no encontrada: {exc}")
        log.warning("La API arrancará pero /predict devolverá 503 hasta que se entrene el modelo")
        state.predictor = predictor
        state.model_loaded = False
    yield
    log.info("API detenida")


app = FastAPI(
    title="World Cup 2026 Predictor",
    description="Predicción de partidos del Mundial 2026 con modelo de Poisson jerárquico bayesiano",
    version="1.0.0",
    lifespan=lifespan,
)

# Orígenes permitidos para CORS (Vercel en prod + localhost en dev)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model() -> Predictor:
    """Lanza 503 si el modelo no está cargado."""
    if not state.model_loaded or state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está disponible. "
                "Ejecutá model/train.py para generar la posterior."
            ),
        )
    return state.predictor


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utilidades"])
async def health() -> HealthResponse:
    """Verifica el estado de la API y si el modelo está cargado."""
    teams_count = 0
    if state.model_loaded and state.predictor is not None:
        try:
            teams_count = len(state.predictor.get_teams())
        except Exception:
            pass
    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded,
        teams_available=teams_count,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predice el resultado de un partido dado.

    - **team_home**: equipo local (o equipo A en partido neutral)
    - **team_away**: equipo visitante (o equipo B)
    - **neutral**: True si el partido se disputa en sede neutral
    """
    predictor = _require_model()
    try:
        result = predictor.predict(
            team_home=request.team_home,
            team_away=request.team_away,
            es_local=not request.neutral,
            ranking_home=request.ranking_home,
            ranking_away=request.ranking_away,
        )
    except Exception as exc:
        log.exception(f"Error en predicción para {request.team_home} vs {request.team_away}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictResponse(
        team_home=result.team_home,
        team_away=result.team_away,
        prob_home=result.prob_home,
        prob_draw=result.prob_draw,
        prob_away=result.prob_away,
        expected_goals_home=result.expected_goals_home,
        expected_goals_away=result.expected_goals_away,
        credible_interval_home=list(result.credible_interval_home),
        credible_interval_away=list(result.credible_interval_away),
        most_likely_score_home=result.most_likely_score_home,
        most_likely_score_away=result.most_likely_score_away,
        goals_dist_home=result.goals_dist_home,
        goals_dist_away=result.goals_dist_away,
        scorelines=result.scorelines,
    )


@app.get("/teams", response_model=TeamsResponse, tags=["Equipos"])
async def get_teams() -> TeamsResponse:
    """Retorna la lista de equipos disponibles en el modelo."""
    predictor = _require_model()
    teams = predictor.get_teams()
    return TeamsResponse(teams=sorted(teams), total=len(teams))


@app.get("/standings", response_model=StandingsResponse, tags=["Tabla"])
async def get_standings() -> StandingsResponse:
    """
    Retorna la tabla de posiciones por grupo con predicciones
    para los partidos restantes.

    Nota: los resultados reales se deben cargar desde data/processed/matches.csv.
    Los partidos pendientes se predicen con el modelo bayesiano.
    """
    predictor = _require_model()

    standings: list[StandingsEntry] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for group_name, teams in WC_2026_GROUPS.items():
        # Intentar predecir todos los partidos del grupo para estimar la tabla
        team_points: dict[str, float] = {t: 0.0 for t in teams}

        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1:]:
                try:
                    result = predictor.predict(team_a, team_b, es_local=False)
                    # Puntos esperados: victoria=3, empate=1
                    team_points[team_a] += 3 * result.prob_home + 1 * result.prob_draw
                    team_points[team_b] += 3 * result.prob_away + 1 * result.prob_draw
                except Exception as exc:
                    log.warning(f"No se pudo predecir {team_a} vs {team_b}: {exc}")

        # Ordenar por puntos predichos descendente
        sorted_teams = sorted(teams, key=lambda t: team_points[t], reverse=True)
        total_teams = len(sorted_teams)

        for rank, team in enumerate(sorted_teams):
            # Probabilidad simple de clasificar: top 2 pasan directo
            # (simplificado; en el Mundial 2026 pasan los 2 primeros + mejores terceros)
            prob_advance = max(0.0, min(1.0, (total_teams - rank) / total_teams))

            standings.append(
                StandingsEntry(
                    group=group_name,
                    team=team,
                    played=0,
                    wins=0,
                    draws=0,
                    losses=0,
                    goals_for=0,
                    goals_against=0,
                    points=0,
                    predicted_points=round(team_points[team], 2),
                    prob_advance=round(prob_advance, 3),
                )
            )

    return StandingsResponse(standings=standings, last_updated=timestamp)


# ── Reentrenamiento en vivo ───────────────────────────────────────────────────

def _do_retrain() -> None:
    """
    Descarga el CSV actualizado de martj42 (incluye resultados del WC 2026
    que se hayan jugado) y reentrena el modelo Laplace en ~15 segundos.
    Actualiza el estado global; el próximo /predict usa el modelo nuevo.
    """
    try:
        log.info("=== Reentrenamiento iniciado ===")
        state.retrain_running = True

        from data.fetch_data import fetch_all_data
        from model.train import train

        fetch_all_data()
        train()

        new_predictor = Predictor()
        new_predictor._load()
        state.predictor = new_predictor
        state.model_loaded = True
        state.last_trained = datetime.now(timezone.utc).isoformat()
        log.info(f"=== Reentrenamiento completado: {state.last_trained} ===")
    except Exception:
        log.exception("Error durante el reentrenamiento")
    finally:
        state.retrain_running = False


@app.post("/admin/retrain", tags=["Admin"])
async def retrain(
    background_tasks: BackgroundTasks,
    x_admin_key: str = Header(default=""),
) -> dict:
    """
    Vuelve a descargar los resultados del CSV de martj42 (que se actualiza
    con los partidos del Mundial 2026 mientras se juegan) y reentrena el
    modelo en background (~15s). Las predicciones mejoran automáticamente
    con cada resultado nuevo.

    Protegido con la variable de entorno ADMIN_KEY (si no está seteada,
    acepta cualquier request — solo para desarrollo).
    """
    admin_key = os.getenv("ADMIN_KEY", "")
    if admin_key and x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Clave de administrador incorrecta")
    if state.retrain_running:
        return {"status": "already_running", "message": "Ya hay un reentrenamiento en curso"}

    background_tasks.add_task(_do_retrain)
    return {
        "status": "started",
        "message": "Reentrenando con los últimos resultados del Mundial 2026. Listo en ~15 segundos.",
    }


@app.get("/admin/retrain/status", tags=["Admin"])
async def retrain_status() -> dict:
    """Estado del modelo: cuándo se entrenó y cuántos partidos incorporó."""
    teams_count = 0
    if state.model_loaded and state.predictor is not None:
        try:
            teams_count = len(state.predictor.get_teams())
        except Exception:
            pass
    return {
        "model_loaded": state.model_loaded,
        "retrain_running": state.retrain_running,
        "last_trained": state.last_trained,
        "teams_in_model": teams_count,
    }

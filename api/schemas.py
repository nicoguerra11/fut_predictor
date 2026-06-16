"""
schemas.py
Modelos Pydantic para request/response de la API de predicción del Mundial.
"""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Cuerpo del request POST /predict."""

    team_home: str = Field(..., description="Nombre del equipo local (o equipo A)", examples=["Argentina"])
    team_away: str = Field(..., description="Nombre del equipo visitante (o equipo B)", examples=["Brasil"])
    neutral: bool = Field(
        default=False,
        description="True si el partido se juega en sede neutral (sin ventaja local)",
    )
    ranking_home: int | None = Field(
        default=None,
        ge=1,
        description="Ranking FIFA del equipo local (opcional; usa mediana si se omite)",
    )
    ranking_away: int | None = Field(
        default=None,
        ge=1,
        description="Ranking FIFA del equipo visitante (opcional)",
    )

    @field_validator("team_home", "team_away")
    @classmethod
    def no_empty_team(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("El nombre del equipo no puede estar vacío")
        return v


class PredictResponse(BaseModel):
    """Respuesta del endpoint POST /predict."""

    team_home: str
    team_away: str
    prob_home: float = Field(..., ge=0, le=1, description="Probabilidad de victoria local")
    prob_draw: float = Field(..., ge=0, le=1, description="Probabilidad de empate")
    prob_away: float = Field(..., ge=0, le=1, description="Probabilidad de victoria visitante")
    expected_goals_home: float = Field(..., ge=0, description="Goles esperados equipo local")
    expected_goals_away: float = Field(..., ge=0, description="Goles esperados equipo visitante")
    credible_interval_home: list[int] = Field(
        ..., min_length=2, max_length=2,
        description="Intervalo de credibilidad 89% goles locales [lower, upper]",
    )
    credible_interval_away: list[int] = Field(
        ..., min_length=2, max_length=2,
        description="Intervalo de credibilidad 89% goles visitantes [lower, upper]",
    )
    most_likely_score_home: int = Field(..., ge=0)
    most_likely_score_away: int = Field(..., ge=0)
    goals_dist_home: list[float] = Field(
        ..., description="Distribución de probabilidad de 0 a 8 goles (equipo local)"
    )
    goals_dist_away: list[float] = Field(
        ..., description="Distribución de probabilidad de 0 a 8 goles (equipo visitante)"
    )
    scorelines: list[dict] = Field(
        ..., description="36 marcadores exactos (0-0 a 5-5) ordenados por probabilidad descendente"
    )


class TeamInfo(BaseModel):
    """Información básica de un equipo disponible."""

    name: str


class TeamsResponse(BaseModel):
    """Lista de equipos disponibles en el modelo."""

    teams: list[str]
    total: int


class HealthResponse(BaseModel):
    """Estado de la API."""

    status: str
    model_loaded: bool
    teams_available: int


class StandingsEntry(BaseModel):
    """Fila de la tabla de posiciones con predicciones."""

    group: str
    team: str
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int
    predicted_points: float = Field(
        ..., description="Puntos esperados incluyendo partidos pendientes"
    )
    prob_advance: float = Field(
        ..., ge=0, le=1, description="Probabilidad estimada de pasar a la siguiente fase"
    )


class StandingsResponse(BaseModel):
    """Tabla de posiciones con predicciones."""

    standings: list[StandingsEntry]
    last_updated: str

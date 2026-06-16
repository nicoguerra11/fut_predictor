"""
fetch_data.py
Descarga resultados históricos de partidos internacionales competitivos desde:
  - GitHub CSV de martj42/international_results (fuente primaria, ~50k partidos)
    Incluye: Mundiales, clasificatorias, Euros, Copa América, AFCON, etc.
  - Ranking FIFA: dict hardcodeado de los 48 clasificados al Mundial 2026.

Guarda el resultado en data/processed/matches.csv.
"""

import io
import logging
import os
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INTERNATIONAL_RESULTS_CSV = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Solo torneos competitivos oficiales — excluye amistosos para mantener calidad
COMPETITIVE_TOURNAMENTS = {
    # Mundiales
    "FIFA World Cup",
    "FIFA World Cup qualification",
    # Europa
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    # Sudamérica
    "Copa América",
    "Copa América Centenario",
    "CONMEBOL-UEFA Cup of Champions",
    # África
    "Africa Cup of Nations",
    "Africa Cup of Nations qualification",
    # Asia
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    # CONCACAF
    "CONCACAF Gold Cup",
    "CONCACAF Nations League",
    "CONCACAF Championship",
    # Oceanía
    "OFC Nations Cup",
    "OFC Nations Cup qualification",
}

# Año de inicio — datos más recientes dan mejor señal para WC 2026
START_YEAR = 2018

# Rankings FIFA aproximados de los 48 clasificados al Mundial 2026
# Sorteo oficial: 11 de junio de 2026
_FALLBACK_FIFA_RANKINGS: dict[str, int] = {
    "Argentina": 1,   "France": 2,      "Spain": 3,       "England": 4,
    "Brazil": 5,      "Belgium": 6,     "Portugal": 7,    "Netherlands": 8,
    "Colombia": 10,   "United States": 11, "Croatia": 12, "Germany": 13,
    "Morocco": 14,    "Japan": 15,      "Mexico": 16,     "Uruguay": 17,
    "Switzerland": 19, "Sweden": 20,   "South Korea": 22, "Scotland": 24,
    "Australia": 25,  "Austria": 26,   "Turkey": 28,      "Senegal": 29,
    "Norway": 31,     "Iran": 33,      "Czechia": 37,     "Tunisia": 38,
    "Egypt": 40,      "Canada": 42,    "Ecuador": 44,     "Panama": 48,
    "Paraguay": 51,   "Saudi Arabia": 55, "Algeria": 58,
    "Bosnia and Herzegovina": 60, "Ivory Coast": 61, "Ghana": 63,
    "Cape Verde": 66, "South Africa": 68, "Iraq": 69,    "Uzbekistan": 72,
    "DR Congo": 74,   "Qatar": 76,     "Curaçao": 85,    "Jordan": 88,
    "New Zealand": 92, "Haiti": 112,
}

# Normalización de nombres: unifica variantes hacia el nombre usado en teams.js
_NAME_NORM: dict[str, str] = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea DPR": "North Korea",
    "Trinidad and Tobago": "Trinidad & Tobago",
    # DR Congo: el CSV de martj42 puede usar "Congo DR"
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    # Curaçao: variante sin tilde
    "Curacao": "Curaçao",
    # Ivory Coast: el CSV usa el nombre francés con apóstrofe Unicode
    "Côte d’Ivoire": "Ivory Coast",       # ASCII apostrophe
    "Côte d’Ivoire": "Ivory Coast",  # Unicode right single quote
    "Cote d’Ivoire": "Ivory Coast",
}


def normalize_name(name: str) -> str:
    """Normaliza el nombre de una selección. Unifica variantes de apóstrofe."""
    # Normalizar apóstrofes Unicode a ASCII antes de buscar en el dict
    normalized = name.replace("’", "'").replace("ʼ", "'")
    return _NAME_NORM.get(normalized, normalized)


# ── Fuente principal: GitHub CSV ──────────────────────────────────────────────

def fetch_competitive_matches(client: httpx.Client) -> pd.DataFrame:
    """
    Descarga el CSV completo de resultados internacionales y filtra a torneos
    competitivos desde START_YEAR. Cubre todos los equipos del WC 2026.
    """
    log.info("Descargando resultados internacionales desde GitHub CSV...")
    try:
        resp = client.get(INTERNATIONAL_RESULTS_CSV, timeout=120)
        resp.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise RuntimeError(f"No se pudo descargar el CSV de resultados: {e}") from e

    raw_path = RAW_DIR / "international_results.csv"
    raw_path.write_bytes(resp.content)
    log.info(f"  CSV descargado: {len(resp.content) / 1024:.0f} KB")

    df = pd.read_csv(io.StringIO(resp.text))
    # Columnas: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Filtrar: torneos competitivos + años recientes
    mask = (
        df["tournament"].isin(COMPETITIVE_TOURNAMENTS)
        & (df["year"] >= START_YEAR)
    )
    df_filtered = df[mask].copy()
    df_filtered = df_filtered.dropna(subset=["home_score", "away_score"])

    df_filtered["equipo_local"] = df_filtered["home_team"].apply(normalize_name)
    df_filtered["equipo_visitante"] = df_filtered["away_team"].apply(normalize_name)
    df_filtered["goles_local"] = df_filtered["home_score"].astype(int)
    df_filtered["goles_visitante"] = df_filtered["away_score"].astype(int)
    df_filtered["fecha"] = df_filtered["date"]
    df_filtered["torneo"] = df_filtered["tournament"]

    # es_local: True si el equipo_local juega en su propio estadio (clasificatorias, etc.)
    # En sedes neutrales (Mundiales, finales) = False para ambos equipos.
    df_filtered["es_local"] = ~df_filtered["neutral"].fillna(False)

    n_total = len(df_filtered)
    n_teams = len(set(df_filtered["equipo_local"]) | set(df_filtered["equipo_visitante"]))
    n_tournaments = df_filtered["tournament"].nunique()
    log.info(f"  → {n_total} partidos competitivos | {n_teams} equipos | {n_tournaments} torneos")
    log.info(f"     Rango: {df_filtered['fecha'].min()} — {df_filtered['fecha'].max()}")

    return df_filtered[[
        "fecha", "equipo_local", "equipo_visitante",
        "goles_local", "goles_visitante",
        "torneo", "es_local", "year",
    ]]


# ── Ranking FIFA ──────────────────────────────────────────────────────────────

def fetch_fifa_ranking(client: httpx.Client) -> pd.DataFrame:
    """Usa el ranking hardcodeado de los 48 clasificados al WC 2026."""
    log.info("Usando ranking FIFA hardcodeado para los 48 clasificados al WC 2026.")
    return pd.DataFrame(
        [{"equipo": k, "ranking_fifa": v} for k, v in _FALLBACK_FIFA_RANKINGS.items()]
    )


def merge_with_ranking(matches_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ranking_local y ranking_visitante al dataframe de partidos."""
    ranking_lookup = dict(zip(ranking_df["equipo"], ranking_df["ranking_fifa"]))
    default_rank = int(ranking_df["ranking_fifa"].median())

    def get_rank(team: str) -> int:
        return ranking_lookup.get(team, ranking_lookup.get(normalize_name(team), default_rank))

    matches_df = matches_df.copy()
    matches_df["ranking_local"] = matches_df["equipo_local"].apply(get_rank)
    matches_df["ranking_visitante"] = matches_df["equipo_visitante"].apply(get_rank)
    return matches_df


# ── Orquestador ───────────────────────────────────────────────────────────────

def fetch_all_data() -> pd.DataFrame:
    """Orquesta la descarga completa y guarda matches.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        matches_df = fetch_competitive_matches(client)
        ranking_df = fetch_fifa_ranking(client)

    matches_df = merge_with_ranking(matches_df, ranking_df)

    final_cols = [
        "fecha", "equipo_local", "equipo_visitante",
        "goles_local", "goles_visitante",
        "ranking_local", "ranking_visitante",
        "torneo", "es_local",
    ]
    matches_df = matches_df[final_cols].sort_values("fecha").reset_index(drop=True)

    out = PROCESSED_DIR / "matches.csv"
    matches_df.to_csv(out, index=False)
    log.info(f"Dataset guardado en {out} ({len(matches_df)} partidos)")
    return matches_df


if __name__ == "__main__":
    df = fetch_all_data()
    n_teams = len(set(df["equipo_local"]) | set(df["equipo_visitante"]))
    print(df.head(10).to_string())
    print(f"\nTotal: {len(df)} partidos")
    print(f"Equipos únicos: {n_teams}")
    print(f"Torneos: {sorted(df['torneo'].unique())}")

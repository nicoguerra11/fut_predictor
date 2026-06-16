"""
fetch_data.py
Descarga resultados históricos de Mundiales desde dos fuentes:
  1. GitHub CSV de martj42/international_results (sin API key) — fuente primaria
  2. football-data.org API (opcional, requiere FOOTBALL_DATA_API_KEY) — suplemento

Ranking FIFA: GitHub CSV de stefanoltmann/fifa-ranking con dict hardcodeado como fallback.
Guarda el resultado en data/processed/matches.csv.
"""

import io
import json
import logging
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FOOTBALL_API_BASE = "https://api.football-data.org/v4"
FIFA_RANKING_CSV = (
    "https://raw.githubusercontent.com/stefanoltmann/fifa-ranking/main/ranking.csv"
)
INTERNATIONAL_RESULTS_CSV = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

WORLD_CUP_YEARS = [2014, 2018, 2022]

PHASE_MAP = {
    "GROUP_STAGE": "grupos",
    "LAST_16": "octavos",
    "QUARTER_FINALS": "cuartos",
    "SEMI_FINALS": "semifinales",
    "THIRD_PLACE": "tercer_puesto",
    "FINAL": "final",
}

# Rankings FIFA aproximados de los 48 clasificados al Mundial 2026 (fallback)
_FALLBACK_FIFA_RANKINGS: dict[str, int] = {
    "Argentina": 1, "France": 2, "Spain": 3, "England": 4,
    "Brazil": 5, "Belgium": 6, "Portugal": 7, "Netherlands": 8,
    "Italy": 9, "Colombia": 10, "United States": 11, "Croatia": 12,
    "Germany": 13, "Morocco": 14, "Mexico": 15, "Uruguay": 16,
    "Japan": 17, "Switzerland": 18, "Iran": 22, "South Korea": 24,
    "Australia": 25, "Poland": 26, "Ukraine": 28, "Senegal": 29,
    "Serbia": 33, "Tunisia": 35, "Nigeria": 36, "Egypt": 38,
    "Peru": 39, "Canada": 40, "Chile": 43, "Ecuador": 44,
    "Cameroon": 45, "Panama": 47, "Costa Rica": 52, "Jamaica": 53,
    "Saudi Arabia": 55, "Algeria": 56, "Ivory Coast": 59, "South Africa": 66,
    "Uzbekistan": 70, "Venezuela": 71, "Honduras": 81, "Bolivia": 85,
    "Bahrain": 86, "Guatemala": 90, "New Zealand": 92, "Indonesia": 130,
}

# Normalización de nombres de selecciones entre fuentes
_NAME_NORM: dict[str, str] = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea DPR": "North Korea",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Ivory Coast": "Côte d'Ivoire",
    "Côte d'Ivoire": "Ivory Coast",
}


def normalize_name(name: str) -> str:
    """Normaliza nombres de selecciones para unificar fuentes."""
    return _NAME_NORM.get(name, name)


# ── Fuente primaria: GitHub CSV ───────────────────────────────────────────────

def fetch_from_github_csv(client: httpx.Client) -> pd.DataFrame:
    """
    Descarga el CSV de resultados internacionales de martj42/international_results
    y filtra los partidos del Mundial 2014, 2018 y 2022.
    """
    log.info("Descargando resultados desde GitHub CSV (fuente primaria)...")
    try:
        resp = client.get(INTERNATIONAL_RESULTS_CSV, timeout=60)
        resp.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise RuntimeError(f"No se pudo descargar el CSV de resultados: {e}") from e

    raw_path = RAW_DIR / "international_results.csv"
    raw_path.write_bytes(resp.content)

    df = pd.read_csv(io.StringIO(resp.text))
    # Columnas: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

    df["year"] = pd.to_datetime(df["date"]).dt.year
    wc = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["year"].isin(WORLD_CUP_YEARS))
    ].copy()

    wc = wc.dropna(subset=["home_score", "away_score"])
    wc["equipo_local"] = wc["home_team"].apply(normalize_name)
    wc["equipo_visitante"] = wc["away_team"].apply(normalize_name)
    wc["goles_local"] = wc["home_score"].astype(int)
    wc["goles_visitante"] = wc["away_score"].astype(int)
    wc["fecha"] = wc["date"]
    wc["torneo"] = "Mundial " + wc["year"].astype(str)
    # En el Mundial todos los partidos son en sede neutral (no hay ventaja local)
    wc["fase"] = "eliminacion"

    log.info(f"  → {len(wc)} partidos del Mundial desde GitHub CSV")
    return wc[["fecha", "equipo_local", "equipo_visitante",
               "goles_local", "goles_visitante", "torneo", "fase", "year"]]


# ── Fuente secundaria: football-data.org (opcional) ──────────────────────────

def fetch_wc_matches_api(year: int, client: httpx.Client) -> list[dict]:
    """
    Descarga partidos desde football-data.org. Retorna lista vacía si falla
    (403, 404, sin API key, etc.) para no interrumpir el proceso.
    """
    if not API_KEY:
        return []

    url = f"{FOOTBALL_API_BASE}/competitions/WC/matches"
    headers = {"X-Auth-Token": API_KEY}
    log.info(f"Intentando football-data.org para {year}...")
    try:
        resp = client.get(url, params={"season": year}, headers=headers, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.warning(f"  football-data.org {year}: HTTP {e.response.status_code} — usando GitHub CSV")
        return []
    except httpx.RequestError as e:
        log.warning(f"  football-data.org {year}: error de red — usando GitHub CSV")
        return []

    matches = resp.json().get("matches", [])
    log.info(f"  → {len(matches)} partidos desde API para {year}")
    return matches


def parse_api_matches(raw_matches: list[dict], year: int) -> list[dict]:
    """Convierte la respuesta JSON de la API al formato interno."""
    rows = []
    for m in raw_matches:
        if m.get("status") not in ("FINISHED",):
            continue
        full_time = m.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        if home_goals is None or away_goals is None:
            continue
        rows.append({
            "fecha": m.get("utcDate", "")[:10],
            "equipo_local": normalize_name(m.get("homeTeam", {}).get("name", "")),
            "equipo_visitante": normalize_name(m.get("awayTeam", {}).get("name", "")),
            "goles_local": int(home_goals),
            "goles_visitante": int(away_goals),
            "torneo": f"Mundial {year}",
            "fase": PHASE_MAP.get(m.get("stage", ""), "grupos"),
            "year": year,
        })
    return rows


# ── Ranking FIFA ─────────────────────────────────────────────────────────────

def fetch_fifa_ranking(client: httpx.Client) -> pd.DataFrame:
    """
    Descarga el ranking FIFA desde el CSV público en GitHub.
    Si la descarga falla, usa el dict hardcodeado como fallback.
    """
    log.info("Descargando ranking FIFA desde GitHub...")
    try:
        resp = client.get(FIFA_RANKING_CSV, timeout=30)
        resp.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        log.warning(f"No se pudo descargar el ranking FIFA ({e}). Usando rankings hardcodeados.")
        return pd.DataFrame(
            [{"equipo": k, "ranking_fifa": v} for k, v in _FALLBACK_FIFA_RANKINGS.items()]
        )

    raw_path = RAW_DIR / "fifa_ranking_raw.csv"
    raw_path.write_bytes(resp.content)

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        log.warning(f"No se pudo parsear el CSV de ranking FIFA ({e}). Usando fallback.")
        return pd.DataFrame(
            [{"equipo": k, "ranking_fifa": v} for k, v in _FALLBACK_FIFA_RANKINGS.items()]
        )

    df.columns = [c.strip().lower() for c in df.columns]
    name_col = next((c for c in df.columns if any(k in c for k in ("team", "name", "country"))), None)
    rank_col = next((c for c in df.columns if "rank" in c), None)

    if not name_col or not rank_col:
        log.warning("No se identificaron columnas de ranking. Usando fallback.")
        return pd.DataFrame(
            [{"equipo": k, "ranking_fifa": v} for k, v in _FALLBACK_FIFA_RANKINGS.items()]
        )

    df_ranking = (
        df[[name_col, rank_col]]
        .dropna()
        .rename(columns={name_col: "equipo", rank_col: "ranking_fifa"})
    )
    df_ranking["ranking_fifa"] = pd.to_numeric(df_ranking["ranking_fifa"], errors="coerce")
    df_ranking = df_ranking.dropna(subset=["ranking_fifa"])
    df_ranking["ranking_fifa"] = df_ranking["ranking_fifa"].astype(int)

    log.info(f"  → {len(df_ranking)} equipos en el ranking FIFA")
    return df_ranking


# ── Merge ranking ─────────────────────────────────────────────────────────────

def merge_with_ranking(matches_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ranking_local y ranking_visitante al dataframe de partidos."""
    ranking_lookup = dict(zip(ranking_df["equipo"], ranking_df["ranking_fifa"]))
    default_rank = int(ranking_df["ranking_fifa"].median())

    def get_rank(team: str) -> int:
        return ranking_lookup.get(team, ranking_lookup.get(normalize_name(team), default_rank))

    matches_df["ranking_local"] = matches_df["equipo_local"].apply(get_rank)
    matches_df["ranking_visitante"] = matches_df["equipo_visitante"].apply(get_rank)
    return matches_df


# ── Orquestador ───────────────────────────────────────────────────────────────

def fetch_all_data() -> pd.DataFrame:
    """Orquesta la descarga completa y guarda matches.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        # 1. Fuente primaria: GitHub CSV
        matches_df = fetch_from_github_csv(client)

        # 2. Suplemento opcional: football-data.org API
        if API_KEY:
            api_rows: list[dict] = []
            for year in WORLD_CUP_YEARS:
                raw = fetch_wc_matches_api(year, client)
                if raw:
                    api_rows.extend(parse_api_matches(raw, year))
                time.sleep(6)  # respetar rate limit

            if api_rows:
                api_df = pd.DataFrame(api_rows)
                # Combinar: preferir datos de API (más detallados) sobre GitHub CSV
                years_from_api = set(api_df["year"].unique())
                github_filtered = matches_df[~matches_df["year"].isin(years_from_api)]
                matches_df = pd.concat([github_filtered, api_df], ignore_index=True)
                log.info(f"Combinado: {len(matches_df)} partidos totales (API + GitHub CSV)")

        # 3. Ranking FIFA
        ranking_df = fetch_fifa_ranking(client)

    matches_df = merge_with_ranking(matches_df, ranking_df)

    final_cols = [
        "fecha", "equipo_local", "equipo_visitante",
        "goles_local", "goles_visitante",
        "ranking_local", "ranking_visitante",
        "torneo", "fase",
    ]
    matches_df = matches_df[final_cols].sort_values("fecha").reset_index(drop=True)

    out = PROCESSED_DIR / "matches.csv"
    matches_df.to_csv(out, index=False)
    log.info(f"Dataset guardado en {out} ({len(matches_df)} partidos)")
    return matches_df


if __name__ == "__main__":
    df = fetch_all_data()
    print(df.head(10).to_string())
    print(f"\nTotal: {len(df)} partidos")
    print(f"Equipos únicos: {df['equipo_local'].nunique()}")
    print(f"Mundiales: {df['torneo'].unique()}")

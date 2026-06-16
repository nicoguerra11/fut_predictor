"""
fetch_data.py
Descarga resultados históricos de Mundiales (2014-2026) desde football-data.org
y el ranking FIFA actual desde GitHub. Guarda el resultado en data/processed/matches.csv.
"""

import os
import time
import logging
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY:
    raise ValueError("Falta FOOTBALL_DATA_API_KEY en el archivo .env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FOOTBALL_API_BASE = "https://api.football-data.org/v4"
FIFA_RANKING_CSV = (
    "https://raw.githubusercontent.com/stefanoltmann/fifa-ranking/main/ranking.csv"
)

# Años de Mundiales disponibles en la API (2014, 2018, 2022) + 2026
WORLD_CUP_YEARS = [2014, 2018, 2022, 2026]

# Mapeo de fases (en inglés → español)
PHASE_MAP = {
    "GROUP_STAGE": "grupos",
    "LAST_16": "octavos",
    "QUARTER_FINALS": "cuartos",
    "SEMI_FINALS": "semifinales",
    "THIRD_PLACE": "tercer_puesto",
    "FINAL": "final",
}


def fetch_wc_matches(year: int, client: httpx.Client) -> list[dict]:
    """Descarga todos los partidos de un Mundial específico desde football-data.org."""
    url = f"{FOOTBALL_API_BASE}/competitions/WC/matches"
    params = {"season": year}
    headers = {"X-Auth-Token": API_KEY}

    log.info(f"Descargando partidos del Mundial {year}...")
    try:
        response = client.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            log.warning(f"No se encontraron datos para el Mundial {year} (404).")
            return []
        raise RuntimeError(f"Error HTTP al consultar el año {year}: {e}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Error de red al consultar el año {year}: {e}") from e

    data = response.json()
    matches = data.get("matches", [])
    log.info(f"  → {len(matches)} partidos encontrados para {year}")
    return matches


def parse_matches(raw_matches: list[dict], year: int) -> list[dict]:
    """Convierte la respuesta JSON de la API al formato interno."""
    rows = []
    for m in raw_matches:
        status = m.get("status", "")
        if status not in ("FINISHED", "IN_PLAY", "PAUSED"):
            continue  # ignorar partidos no jugados aún

        score = m.get("score", {})
        full_time = score.get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        # Si el partido no tiene marcador completo (ej: futuro o en curso sin goles), saltar
        if home_goals is None or away_goals is None:
            continue

        home_team = m.get("homeTeam", {}).get("name", "")
        away_team = m.get("awayTeam", {}).get("name", "")
        stage = m.get("stage", "")
        date = m.get("utcDate", "")[:10]  # YYYY-MM-DD

        rows.append(
            {
                "fecha": date,
                "equipo_local": home_team,
                "equipo_visitante": away_team,
                "goles_local": int(home_goals),
                "goles_visitante": int(away_goals),
                "torneo": f"Mundial {year}",
                "fase": PHASE_MAP.get(stage, stage.lower()),
                "year": year,
            }
        )
    return rows


# Rankings FIFA aproximados de los 48 clasificados al Mundial 2026 (fallback)
_FALLBACK_FIFA_RANKINGS: dict[str, int] = {
    "Argentina": 1, "France": 2, "Spain": 3, "England": 4,
    "Brazil": 5, "Belgium": 6, "Portugal": 7, "Netherlands": 8,
    "Italy": 9, "Colombia": 10, "United States": 11, "Croatia": 12,
    "Germany": 13, "Morocco": 14, "Mexico": 15, "Uruguay": 16,
    "Japan": 17, "Switzerland": 18, "Iran": 22, "South Korea": 24,
    "Australia": 25, "Poland": 26, "Senegal": 29, "Serbia": 33,
    "Egypt": 38, "Peru": 39, "Canada": 40, "Chile": 43,
    "Ecuador": 44, "Cameroon": 45, "Panama": 47, "Costa Rica": 52,
    "Jamaica": 53, "Saudi Arabia": 55, "Algeria": 56, "Ivory Coast": 59,
    "South Africa": 66, "Uzbekistan": 70, "Venezuela": 71,
    "Tunisia": 35, "Nigeria": 36, "Ukraine": 28,
    "Bolivia": 85, "Bahrain": 86, "Guatemala": 90, "New Zealand": 92,
    "Indonesia": 130, "Honduras": 81,
}


def fetch_fifa_ranking(client: httpx.Client) -> pd.DataFrame:
    """
    Descarga el ranking FIFA desde el CSV público en GitHub.
    Si la descarga falla, usa el dict hardcodeado como fallback.
    """
    log.info("Descargando ranking FIFA desde GitHub...")
    try:
        response = client.get(FIFA_RANKING_CSV, timeout=30)
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        log.warning(f"No se pudo descargar el ranking FIFA ({e}). Usando rankings hardcodeados.")
        df_fallback = pd.DataFrame(
            [{"equipo": k, "ranking_fifa": v} for k, v in _FALLBACK_FIFA_RANKINGS.items()]
        )
        return df_fallback

    raw_path = RAW_DIR / "fifa_ranking_raw.csv"
    raw_path.write_bytes(response.content)

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo parsear el CSV del ranking FIFA: {e}") from e

    # El CSV puede tener distintas columnas según la fuente; normalizamos
    df.columns = [c.strip().lower() for c in df.columns]

    # Intentamos detectar columnas de nombre de equipo y ranking
    name_candidates = [c for c in df.columns if "team" in c or "name" in c or "country" in c]
    rank_candidates = [c for c in df.columns if "rank" in c]

    if not name_candidates or not rank_candidates:
        raise ValueError(
            f"No se pudieron identificar columnas de equipo/ranking en el CSV. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    name_col = name_candidates[0]
    rank_col = rank_candidates[0]

    # Quedarse con el ranking más reciente si hay múltiples filas por equipo
    df_ranking = (
        df[[name_col, rank_col]]
        .dropna()
        .rename(columns={name_col: "equipo", rank_col: "ranking_fifa"})
        .astype({"ranking_fifa": int})
    )

    # Si hay columna de fecha, quedarse con la más reciente
    date_candidates = [c for c in df.columns if "date" in c or "fecha" in c]
    if date_candidates:
        date_col = date_candidates[0]
        df_with_date = df[[name_col, rank_col, date_col]].dropna()
        df_with_date = df_with_date.rename(
            columns={name_col: "equipo", rank_col: "ranking_fifa", date_col: "fecha_ranking"}
        )
        df_with_date["fecha_ranking"] = pd.to_datetime(
            df_with_date["fecha_ranking"], errors="coerce"
        )
        df_ranking = (
            df_with_date.sort_values("fecha_ranking")
            .groupby("equipo")
            .last()
            .reset_index()[["equipo", "ranking_fifa"]]
            .astype({"ranking_fifa": int})
        )

    log.info(f"  → {len(df_ranking)} equipos en el ranking FIFA")
    return df_ranking


def normalize_team_name(name: str) -> str:
    """Normaliza nombres de selecciones para mejorar el matching con el ranking FIFA."""
    replacements = {
        "Korea Republic": "South Korea",
        "Korea DPR": "North Korea",
        "IR Iran": "Iran",
        "USA": "United States",
        "Türkiye": "Turkey",
        "Czech Republic": "Czechia",
        "Republic of Ireland": "Ireland",
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
        "Trinidad and Tobago": "Trinidad & Tobago",
    }
    return replacements.get(name, name)


def merge_with_ranking(matches_df: pd.DataFrame, ranking_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas ranking_local y ranking_visitante al dataframe de partidos."""
    ranking_lookup = dict(zip(ranking_df["equipo"], ranking_df["ranking_fifa"]))
    default_rank = int(ranking_df["ranking_fifa"].median())

    matches_df["equipo_local_norm"] = matches_df["equipo_local"].apply(normalize_team_name)
    matches_df["equipo_visitante_norm"] = matches_df["equipo_visitante"].apply(
        normalize_team_name
    )

    matches_df["ranking_local"] = matches_df["equipo_local_norm"].map(ranking_lookup)
    matches_df["ranking_visitante"] = matches_df["equipo_visitante_norm"].map(ranking_lookup)

    # Para equipos sin ranking conocido, usar la mediana
    missing_local = matches_df["ranking_local"].isna().sum()
    missing_away = matches_df["ranking_visitante"].isna().sum()
    if missing_local > 0:
        log.warning(
            f"{missing_local} equipos locales sin ranking FIFA; usando mediana ({default_rank})"
        )
    if missing_away > 0:
        log.warning(
            f"{missing_away} equipos visitantes sin ranking FIFA; usando mediana ({default_rank})"
        )

    matches_df["ranking_local"] = matches_df["ranking_local"].fillna(default_rank).astype(int)
    matches_df["ranking_visitante"] = (
        matches_df["ranking_visitante"].fillna(default_rank).astype(int)
    )

    return matches_df.drop(columns=["equipo_local_norm", "equipo_visitante_norm"])


def fetch_all_data() -> pd.DataFrame:
    """Orquesta la descarga completa y devuelve el DataFrame final."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_matches: list[dict] = []

    with httpx.Client() as client:
        for year in WORLD_CUP_YEARS:
            raw_matches = fetch_wc_matches(year, client)
            if raw_matches:
                parsed = parse_matches(raw_matches, year)
                all_matches.extend(parsed)
                # Guardar raw para debugging
                raw_path = RAW_DIR / f"wc_{year}_raw.json"
                import json
                raw_path.write_text(json.dumps(raw_matches, ensure_ascii=False, indent=2))
            # Respetar rate limit de la API gratuita (10 req/min)
            time.sleep(6)

        ranking_df = fetch_fifa_ranking(client)

    if not all_matches:
        raise RuntimeError(
            "No se descargó ningún partido. Verificá la API key y la conexión."
        )

    matches_df = pd.DataFrame(all_matches)
    matches_df = merge_with_ranking(matches_df, ranking_df)

    # Columnas finales según spec
    final_cols = [
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "goles_local",
        "goles_visitante",
        "ranking_local",
        "ranking_visitante",
        "torneo",
        "fase",
    ]
    matches_df = matches_df[final_cols].sort_values("fecha").reset_index(drop=True)

    output_path = PROCESSED_DIR / "matches.csv"
    matches_df.to_csv(output_path, index=False)
    log.info(f"Dataset guardado en {output_path} ({len(matches_df)} partidos)")

    return matches_df


if __name__ == "__main__":
    df = fetch_all_data()
    print(df.head(10).to_string())
    print(f"\nTotal de partidos: {len(df)}")
    print(f"Equipos únicos: {df['equipo_local'].nunique()}")
    print(f"Mundiales incluidos: {df['torneo'].unique()}")

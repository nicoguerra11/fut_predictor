# World Cup 2026 Predictor

Predicción de partidos del Mundial 2026 usando un modelo de **Poisson jerárquico bayesiano** implementado con PyMC 5.

## Modelo estadístico

```
goles_ij ~ Poisson(λ_ij)
log(λ_ij) = μ + ataque_i - defensa_j + α·Δranking_ij + γ·es_local

Δranking_ij = (ranking_rival - ranking_propio) / 50
```

**Priors:**
- `ataque_i ~ Normal(0, σ_ataque)` — efecto jerárquico de ataque por selección
- `defensa_i ~ Normal(0, σ_defensa)` — efecto jerárquico de defensa por selección
- `μ ~ Normal(0, 1)` — media global de goles (escala log)
- `α ~ Normal(0, 0.5)` — efecto del ranking FIFA
- `γ ~ Normal(0, 1)` — ventaja de local (solo fase de grupos)
- `σ_ataque, σ_defensa ~ HalfNormal(1)`

## Estructura

```
world-cup-predictor/
├── data/
│   ├── fetch_data.py       ← descarga datos de football-data.org + ranking FIFA
│   ├── raw/                ← JSONs crudos de la API
│   └── processed/
│       └── matches.csv     ← dataset limpio para entrenar
├── model/
│   ├── train.py            ← entrena el modelo y guarda posterior.nc
│   ├── predict.py          ← carga posterior y simula partidos
│   └── evaluate.py         ← LOO-CV, RPS, calibration plot
├── api/
│   ├── main.py             ← FastAPI (endpoints: /health, /predict, /teams, /standings)
│   └── schemas.py          ← modelos Pydantic
├── frontend/               ← React + Vite (deploy en Vercel)
├── notebooks/
│   └── eda.ipynb
├── requirements.txt
├── render.yaml             ← deploy backend en Render
└── vercel.json             ← deploy frontend en Vercel
```

## Setup rápido

### 1. Configurar variables de entorno

```bash
cp .env.example .env
# Editá .env con tu API key de football-data.org (gratuita en https://www.football-data.org/)
```

### 2. Instalar dependencias Python

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Descargar datos

```bash
python data/fetch_data.py
```

### 4. Entrenar el modelo

```bash
python model/train.py
# Tarda ~10-20 min dependiendo del hardware (PyMC con 4 cadenas × 2000 draws)
```

### 5. Levantar la API

```bash
uvicorn api.main:app --reload --port 8000
# Documentación interactiva: http://localhost:8000/docs
```

### 6. Levantar el frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
# App en: http://localhost:5173
```

## Reentrenamiento

Cuando haya nuevos resultados del Mundial 2026:

```bash
# 1. Actualizar el dataset
python data/fetch_data.py

# 2. Reentrenar
python -c "from model.train import retrain; retrain()"
```

## Evaluación del modelo

```bash
python model/evaluate.py
# Genera reports/calibration_plot.png y muestra LOO-CV + RPS
```

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado de la API y del modelo |
| POST | `/predict` | Predice un partido dado dos equipos |
| GET | `/teams` | Lista de equipos disponibles |
| GET | `/standings` | Tabla de grupos con predicciones |

### Ejemplo `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"team_home": "Argentina", "team_away": "Brasil", "neutral": false}'
```

```json
{
  "prob_home": 0.4312,
  "prob_draw": 0.2801,
  "prob_away": 0.2887,
  "expected_goals_home": 1.42,
  "expected_goals_away": 1.18,
  "credible_interval_home": [0, 3],
  "credible_interval_away": [0, 3],
  "most_likely_score_home": 1,
  "most_likely_score_away": 1
}
```

## Deploy

### Backend (Render)
El archivo `render.yaml` configura un Web Service gratuito en Render.
Seteá la env var `FOOTBALL_DATA_API_KEY` en el dashboard de Render.

### Frontend (Vercel)
```bash
cd frontend
vercel --prod
# Configurá VITE_API_URL apuntando a la URL de tu servicio en Render
```

## Limitaciones conocidas

- La API gratuita de football-data.org tiene rate limit de 10 req/min.
- El modelo asume independencia entre goles de ambos equipos (no captura correlación).
- Equipos sin historial en los datos usan la media global como prior.
- Las predicciones de fase de grupos no toman en cuenta el sorteo real del Mundial 2026.

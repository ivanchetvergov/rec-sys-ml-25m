# RecSys

Two-stage movie recommender system with full stack implementation:

- Offline ML pipeline (feature store, training, evaluation, artifacts)
- Online FastAPI backend (personal recommendations, similar items, user actions)
- Next.js frontend (catalog, recommendations, profile, review flows)

## 1. Scope

The project solves three tasks end-to-end:

1. Build recommendation artifacts from MovieLens data.
2. Serve recommendations and user workflows through API.
3. Provide web UI for interaction and feedback collection.

## 2. Architecture

Data flow:

1. Raw CSV -> preprocessing -> feature store (`data/processed/feature_store`).
2. Train iALS + CatBoost ranker -> artifacts (`data/models/two_stage_ranker`).
3. Build movie catalog and similarity index (`data/processed/movies.parquet`, `data/processed/similarity_index.parquet`).
4. Backend loads artifacts at startup and serves requests.
5. Frontend consumes API and writes user feedback (`watched`, `watchlist`, `reviews`) to Postgres.

Key docs:

- `docs/ARCHITECTURE.md`
- `docs/ml-pipeline.md`
- `docs/backend.md`
- `docs/frontend.md`

## 3. Repository Layout

```text
.
├── backend/                    # FastAPI app, routers, services, migrations
├── frontend/                   # Next.js app
├── src/                        # ML pipeline, models, training scripts
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Feature store, movies catalog, similarity index
│   └── models/                 # Trained model artifacts
├── docs/                       # Technical documentation
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## 4. Requirements

- Python 3.11+
- Node.js 18+
- Docker + Docker Compose (for full web stack)
- `make`

Notes:

- `Makefile` contains hardcoded Python paths. Before using it, update `PYTHON` and `PIP` to your local environment.
- Raw MovieLens files must be present under `data/raw/ml-25m`.

## 5. Quick Start

### Option A: Full stack in Docker (recommended)

1. Prepare artifacts:

```bash
make preprocess
make train-ranker
make build-similarity
```

1. Start web stack:

```bash
make web
```

1. Open:

- Frontend via nginx: `http://localhost`
- API docs: `http://localhost/api/docs`

1. Stop:

```bash
make web-down
```

### Option B: Run backend/frontend locally

1. Install Python dependencies:

```bash
make install
```

1. Start backend:

```bash
make backend
```

1. Start frontend (separate terminal):

```bash
make frontend
```

## 6. Core Workflows

### 6.1 Data and artifacts

```bash
make preprocess
make extract-movies
make train-ranker
make build-similarity
```

### 6.2 Model baselines and experiments

```bash
make train-popularity
make train-cf
make train-als
make train-ranker
```

Sample runs:

```bash
make train-popularity-sample
make train-cf-sample
make train-als-sample
make train-ranker-sample
```

### 6.3 MLflow

```bash
make mlflow-ui
```

Default UI: `http://localhost:5000`

## 7. Backend API Surface

Main route groups under `/api`:

- `movies`: popular, personal, search, similar, details
- `auth`: register, login, me
- `watched`
- `watchlist`
- `reviews`
- `users` (public profile + privacy)
- `admin` (stats)

Entry point: `backend/app/main.py`

## 8. Recommendation Runtime Behavior

- Primary model: two-stage (`iALS` candidates -> `CatBoost` rerank)
- Online adaptation: user fold-in and post-ranking from fresh interactions
- Degradation mode: `popularity_fallback` if model/artifacts unavailable

Model marker in API responses:

- `two_stage`
- `two_stage_live_foldin`
- `popularity_fallback`

## 9. Data Lifecycle Summary

1. Raw interactions enter preprocessing.
2. Temporal split and no-leakage features are built.
3. Training artifacts are produced and versioned.
4. Backend serves online recommendations from loaded artifacts.
5. User feedback goes to Postgres and influences online ranking immediately.
6. Full model refresh happens on next retraining cycle.

## 10. Troubleshooting

1. Empty personal recommendations: check model artifacts in `data/models/two_stage_ranker` and backend logs for fallback warnings.

1. Similar movies unavailable: rebuild index with `make build-similarity`.

1. Backend fails at startup: validate Postgres connectivity and verify migrations in `backend/migrations` were applied.

1. Frontend cannot reach API: check `NEXT_PUBLIC_API_URL` and nginx routing in Docker setup.

## 11. Cleanup

```bash
make clean
```

This removes processed feature store and MLflow artifacts.

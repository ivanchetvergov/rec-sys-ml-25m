# RecSys – Architecture Overview

```
Browser
  │
  ▼
┌──────────────────────────────────────────┐
│  Nginx :80                               │  reverse proxy
│  /api/*  →  backend:8000                 │
│  /*      →  frontend:3000                │
└──────────────────────────────────────────┘
       │                     │
       ▼                     ▼
┌─────────────┐      ┌─────────────────┐
│  FastAPI    │      │  Next.js 14     │
│  (backend)  │      │  (frontend)     │
│  :8000      │      │  :3000          │
└──────┬──────┘      └─────────────────┘
       │
   ┌───┼─────────────────────────────────┐
   │   │                                 │
   ▼   ▼                                 ▼
PostgreSQL  Redis               ClickHouse
:5432       :6379               :8123
(users,     (cache,             (analytics:
 movies,     task broker)        user_activity,
 interactions)                   model_metrics)
                │
                ▼
         Celery Worker
         (background tasks:
          cache refresh,
          TMDb sync,
          model refit)

ML Layer (outside containers, host or mounted volume):
  src/models/popularity_based.py   ← PopularityRecommender
  data/processed/feature_store/    ← Parquet splits (train/val/test)
  mlruns/                          ← MLflow experiment tracking
```

## Request lifecycle (recommendation)

```
GET /api/v1/recommendations/home
 │
 ├─ JWT middleware (deps.py) → decode token → load User from PG
 │
 ├─ MLService._ensure_initialized()
 │     ├─ PopularityRecommender.load()   (from disk / feature store)
 │     └─ _build_content_matrix()        (genre cosine from parquet)
 │
 ├─ get_home_recommendations()
 │     ├─ get_popular()       → DB fallback if model not loaded
 │     ├─ _collab_recs()      → stub (returns popular, swap for ALS)
 │     └─ _content_recs()     → cosine similarity on genre matrix
 │
 └─ JSON [RecommendationSection, ...]
```

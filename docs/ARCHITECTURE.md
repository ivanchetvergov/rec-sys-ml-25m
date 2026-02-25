# Архитектура системы

## Схема слоёв

```
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND  (Next.js 14 · TypeScript · Tailwind)             │
│                                                             │
│  src/app/page.tsx                   ← главная страница (SSR)│
│  src/app/movies/[id]/page.tsx       ← страница фильма       │
│  src/app/movies/[id]/               │
│    MoviePageInteractive.tsx         ← рейтинг/отзыв/похожие │
│  src/components/MovieCard.tsx       ← карточка с постером   │
│  src/lib/api.ts                     ← HTTP клиент           │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP REST  (nginx → порт 8000)
┌─────────────────────▼───────────────────────────────────────┐
│  BACKEND  (FastAPI · Python 3.11 · Pydantic v2)             │
│                                                             │
│  app/main.py                         ← точка входа, CORS    │
│  app/routers/movies.py               ← /api/movies/...      │
│  app/services/                                              │
│    popularity_service.py   ← популярные / каталог           │
│    recommender_service.py  ← персональные (TwoStage)        │
│    similarity_service.py   ← похожие фильмы (ALS cosine)    │
│    tmdb_service.py         ← постеры / детали (TMDB API)    │
│  app/schemas.py                      ← Pydantic ответы      │
└─────────────────────┬───────────────────────────────────────┘
                      │ читает parquet напрямую (без БД)
┌─────────────────────▼───────────────────────────────────────┐
│  PROCESSED DATA  (Parquet файлы)                            │
│                                                             │
│  data/processed/                                            │
│  ├── movies.parquet              (~17K строк · ~1 MB)       │
│  ├── similarity_index.parquet    (~17K строк · ~3 MB)       │
│  └── feature_store/                                         │
│       └── ml_v_20260215_184134/                             │
│            ├── train.parquet  (17.4M строк · 56 колонок)    │
│            ├── val.parquet    (3.7M строк)                  │
│            └── test.parquet   (3.7M строк)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ производится из
┌─────────────────────▼───────────────────────────────────────┐
│  ML PIPELINE  (Python · pandas · implicit · CatBoost)       │
│                                                             │
│  src/pipeline/preprocess_pipeline.py  ← ETL оркестратор    │
│  src/pipeline/extract_movies.py       ← movies.parquet      │
│  src/pipeline/build_similarity_index.py ← ALS cosine index │
│  src/models/                                                │
│    popularity_based.py     ← PopularityRecommender (baseline)│
│    als_recommender.py      ← ALSRecommender (Stage 1)       │
│    catboost_ranker.py      ← CatBoostRanker  (Stage 2)      │
│    two_stage_recommender.py← production pipeline            │
│  src/training/                                              │
│    train_popularity.py     ← MLflow: popularity_baseline    │
│    train_als.py            ← MLflow: als_candidate_generator│
│    train_ranker.py         ← MLflow: two_stage_ranker       │
│  src/evaluation/metrics.py ← P@K, R@K, nDCG@K, MAP@K       │
└─────────────────────┬───────────────────────────────────────┘
                      │ исходные данные
┌─────────────────────▼───────────────────────────────────────┐
│  RAW DATA  (MovieLens 25M)                                  │
│                                                             │
│  data/raw/ml-25m/                                           │
│  ├── ratings.csv    (25M записей: userId, movieId, rating)  │
│  ├── movies.csv     (62K фильмов: movieId, title, genres)   │
│  ├── links.csv      (IMDb / TMDb идентификаторы)            │
│  └── tags.csv       (пользовательские теги)                 │
└─────────────────────────────────────────────────────────────┘
```

## Поток данных — запрос на главную страницу

```
браузер
  │  GET /                         (Next.js SSR)
  ▼
Next.js page.tsx
  │  fetchPopularMovies(limit=40)
  │  GET http://backend:8000/api/movies/popular?limit=40
  ▼
FastAPI  /api/movies/popular
  │  Depends(get_popularity_service)
  ▼
PopularityService.get_popular(limit=40)
  │  первый вызов: читает movies.parquet (~17K строк, ~1 MB)
  │  результат кешируется в памяти через @lru_cache
  ▼
list[dict] → PopularMoviesResponse → JSON → MovieCard × 40
```

## Поток данных — персональные рекомендации

```
браузер
  │  GET /api/movies/personal?user_id=123&limit=20
  ▼
FastAPI  /api/movies/personal
  │  Depends(get_recommender_service)
  ▼
RecommenderService.get_personal_recs(user_id=123, n=20)
  │
  ├─ [Stage 1] ALSRecommender → top-300 кандидатов   ~1-2 ms
  ├─ [Stage 2] CatBoostRanker → переранжировать       ~5-10 ms
  └─ [Fallback] PopularityService  (cold-start / ошибка)
  ▼
PersonalRecsResponse (model="two_stage" | "popularity_fallback")
```

## Поток данных — похожие фильмы

```
браузер (MoviePageInteractive.tsx)
  │  fetchSimilarMovies(movie.id)
  │  GET /api/movies/{id}/similar?limit=24
  ▼
FastAPI  /api/movies/{id}/similar
  │  Depends(get_similarity_service)
  ▼
SimilarityService.get_similar_ids(movie_id, n=24)
  │  O(1) lookup в предзагруженном dict
  │  similarity_index.parquet — собирается make build-similarity
  ▼
PopularityService.get_movie(mid) × 24  (обогащение метаданными)
  ▼
SimilarMoviesResponse (model="als_cosine") → 24 карточки

## Дизайн-решения

### Нет базы данных на текущем этапе

Feature store содержит агрегированные метрики (`movie_avg_rating`, `movie_popularity`). Читается из parquet при старте (~400 ms, кешируется). БД добавится когда появятся пользователи и взаимодействия в реальном времени.

### Лёгкий каталог `movies.parquet`

Вместо чтения полного `train.parquet` (17.4M строк, 56 колонок, ~500 MB) при каждом старте бэкенд загружает `movies.parquet` (~17K строк, ~1 MB) — выжимку уникальных фильмов. Генерируется командой `make extract-movies`.

### Pre-computed similarity index

Item-item косинусное сходство по ALS-векторам вычисляется **оффлайн** (`make build-similarity`) один раз после обучения модели. Бэкенд читает 3 MB parquet при старте и отвечает за O(1) — нет зависимостей от `implicit` или CatBoost в рантайме.

### In-memory кеш через `@lru_cache`

Все четыре сервиса (`PopularityService`, `RecommenderService`, `SimilarityService`, `TMDBService`) создаются один раз при первом запросе и живут весь цикл приложения.

### Next.js ISR (Incremental Static Regeneration)

`fetchPopularMovies()` использует `next: { revalidate: 3600 }` — страница кешируется на 1 час. Персональные рекомендации и похожие фильмы используют `cache: 'no-store'`.

### Временной сплит обучающей выборки

Данные разделены **по времени** (70% / 15% / 15%) — модель обучается на прошлом и предсказывает будущее.

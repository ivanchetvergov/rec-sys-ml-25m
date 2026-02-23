# Архитектура системы

## Схема слоёв

```
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND  (Next.js 14 · TypeScript · Tailwind)             │
│                                                             │
│  src/app/page.tsx              ← главная страница (SSR)     │
│  src/components/MovieCard.tsx  ← карточка фильма            │
│  src/lib/api.ts                ← HTTP клиент                │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP REST  (порт 8000)
┌─────────────────────▼───────────────────────────────────────┐
│  BACKEND  (FastAPI · Python 3.11 · Pydantic v2)             │
│                                                             │
│  app/main.py                         ← точка входа, CORS    │
│  app/routers/movies.py               ← /api/movies/...      │
│  app/services/popularity_service.py  ← бизнес-логика        │
│  app/schemas.py                      ← Pydantic ответы      │
└─────────────────────┬───────────────────────────────────────┘
                      │ читает parquet напрямую (без БД)
┌─────────────────────▼───────────────────────────────────────┐
│  FEATURE STORE  (Parquet файлы)                             │
│                                                             │
│  data/processed/feature_store/                              │
│  └── ml_v_20260215_184134/                                  │
│       ├── train.parquet   (17.4M строк · 56 колонок)        │
│       ├── val.parquet     (3.7M строк)                      │
│       ├── test.parquet    (3.7M строк)                      │
│       └── metadata.json                                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ производится из
┌─────────────────────▼───────────────────────────────────────┐
│  ML PIPELINE  (Python · pandas · scikit-learn)              │
│                                                             │
│  src/pipeline/preprocess_pipeline.py  ← оркестратор 5 шагов│
│  src/data_loader.py                   ← загрузка CSV        │
│  src/preprocessor.py                  ← фильтрация          │
│  src/feature_engineer.py              ← 56 признаков        │
│  src/data_splitter.py                 ← temporal split      │
│  src/feature_store.py                 ← сохранение parquet  │
│  src/models/popularity_based.py       ← PopularityRecommender│
│  src/training/train_popularity.py     ← MLflow запуск       │
│  src/evaluation/metrics.py            ← P@K, R@K, nDCG@K   │
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
  │  GET http://localhost:8000/api/movies/popular?limit=40
  ▼
FastAPI  /api/movies/popular
  │  Depends(get_popularity_service)
  ▼
PopularityService.get_popular(limit=40)
  │  первый вызов: читает train.parquet (7 колонок)
  │  результат кешируется в памяти через @lru_cache
  │  повторные вызовы: срез DataFrame за < 1ms
  ▼
list[dict] → Pydantic PopularMoviesResponse
  │
  ▼
JSON ответ → MovieCard × 40 → HTML страница
```

## Дизайн-решения

### Нет базы данных на текущем этапе

Feature store уже содержит агрегированные метрики по фильмам (`movie_avg_rating`, `movie_num_ratings`, `movie_popularity`). Читать их из parquet быстро (~400ms при старте) и дёшево. БД добавится когда появятся пользователи и взаимодействия.

### In-memory кеш через `@lru_cache`

`PopularityService` создаётся один раз при первом запросе и живёт весь цикл приложения. Повторные запросы к `/popular` отдаются из памяти.

### Next.js ISR (Incremental Static Regeneration)

`fetch` с `next: { revalidate: 3600 }` — страница кешируется на 1 час, rebuild происходит в фоне.

### Временной сплит обучающей выборки

Данные разделены **по времени** (70% / 15% / 15%), а не случайно — предотвращает утечку данных из будущего в прошлое.

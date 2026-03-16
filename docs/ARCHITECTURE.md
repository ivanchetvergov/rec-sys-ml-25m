# Архитектура системы

## 1. Обзор

RecSys состоит из трех основных контуров:

1. ML контур (offline): подготовка данных, обучение, построение артефактов.
2. Backend контур (online): FastAPI API, загрузка артефактов, персонализация и профильные функции.
3. Frontend контур: Next.js UI, пользовательские взаимодействия и визуализация рекомендаций.

## 2. Высокоуровневая схема

```text
MovieLens 25M CSV
      |
      v
Preprocess + Feature Engineering
      |
      v
Feature Store (parquet)
      |
      +--> Train two-stage model (iALS + CatBoost)
      |         |
      |         v
      |    data/models/two_stage_ranker
      |
      +--> Extract movies catalogue
      |         |
      |         v
      |    data/processed/movies.parquet
      |
      +--> Build item-item similarity index
                |
                v
           data/processed/similarity_index.parquet


FastAPI startup:
  - run SQL migrations
  - warm PopularityService
  - warm RecommenderService
  - warm SimilarityService

Requests:
  Frontend -> Nginx -> FastAPI -> Services -> parquet + postgres
```

## 3. Контуры данных

### 3.1 Offline data flow

1. src/pipeline/preprocess_pipeline.py
   - load raw ratings/movies/links
   - preprocessing
   - temporal split (train/val/test)
   - no-leakage feature engineering
   - feature store save
2. src/training/train_ranker.py
   - train iALS stage
   - build ranker dataset
   - train CatBoost ranker
   - evaluate val/test
   - log MLflow + save artifacts
3. src/pipeline/extract_movies.py
   - creates light movie catalogue for backend
4. src/pipeline/build_similarity_index.py
   - builds cosine similar items from ALS factors

### 3.2 Online recommendation flow

GET /api/movies/personal?user_id=...

1. RecommenderService loads two-stage pipeline once.
2. iALS retrieves candidate pool.
3. CatBoost reranks candidates.
4. Online refinements:
   - seen filtering from watched/reviews
   - similar-item boost from recent positives
   - genre affinity boost
   - optional user fold-in recomputation with recent interactions
5. Enrichment by movie metadata.
6. Response model marker:
   - two_stage / two_stage_live_foldin / popularity_fallback

## 4. Слой backend

### 4.1 API маршруты

- movies: popular, personal, search, similar, details, movie by id
- auth: register, login, me
- watched: list/add/delete/export
- watchlist: list/add/delete
- reviews: list/upsert/delete/movie public reviews
- users: public profile, me privacy get/put
- admin: overview/daily/top/rating distribution/users

### 4.2 Сервисы

- PopularityService: каталог и популярность
- RecommenderService: two-stage inference + online post-ranking
- SimilarityService: item-item index lookup
- TMDBService: external metadata

### 4.3 Storage стратегия

- Postgres: users, watchlist, watched, reviews, daily_activity
- Parquet artifacts: model + catalogue + similarity index
- In-memory singleton services через @lru_cache

## 5. Слой frontend

Основные UX-потоки:

1. Главная (/): hero + trending + personalized + catalog.
2. Фильм (/movies/[id]): rating/review/watchlist/watched/similar/community reviews.
3. Профиль (/profile): watched/watchlist/reviews + profile privacy toggle.
4. Публичный профиль (/profile/[userId]): доступ с учетом privacy.
5. Auth (/login, /register) + multi-account switch в header.

## 6. Ключевые бизнес-правила

1. Сохранение review (rating 1..5) автоматически:
   - добавляет фильм в watched
   - удаляет фильм из watchlist
2. Публичный профиль закрывается при is_profile_private = true.
3. Unauthorized (401) на клиенте сбрасывает session state.

## 7. Производительность

1. Warm-up на startup снижает latency первого запроса.
2. Similarity lookup O(1) по предвычисленному индексу.
3. Персональные рекомендации не кешируются (no-store).
4. Popular catalogue может revalidate на frontend.

## 8. Надежность

1. Миграции запускаются на старте backend.
2. В users privacy роуте есть defensive DDL guard:
   - ALTER TABLE users ADD COLUMN IF NOT EXISTS is_profile_private ...
3. Есть отдельная миграция hardening для privacy колонки.

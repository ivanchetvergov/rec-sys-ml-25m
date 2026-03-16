# Backend (FastAPI)

## 1. Назначение

Backend предоставляет REST API для:

1. Каталога фильмов и поиска.
2. Персональных рекомендаций.
3. Похожих фильмов.
4. Auth и user-centric функций (watched, watchlist, reviews, profile).
5. Admin аналитики.

## 2. Точка входа

Файл: `backend/app/main.py`

На startup:

1. Запускаются SQL миграции.
2. Прогреваются сервисы:
   - PopularityService
   - RecommenderService
   - SimilarityService

Подключенные роутеры (`/api` префикс):

- movies
- auth
- watchlist
- reviews
- watched
- admin
- users

## 3. Структура backend

```text
backend/app
  main.py
  database.py
  core/
    db.py
    security.py
  routers/
    movies.py
    auth.py
    watchlist.py
    watched.py
    reviews.py
    users.py
    admin.py
  services/
    popularity_service.py
    recommender_service.py
    similarity_service.py
    tmdb_service.py
```

## 4. База данных и миграции

Миграции: `backend/migrations/*.sql`

Текущие ключевые таблицы:

1. users
2. watchlist
3. reviews
4. watched
5. daily_activity

Privacy колонка:

- `users.is_profile_private BOOLEAN NOT NULL DEFAULT FALSE`

Защита от рассинхрона:

- в users router выполняется defensive `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` перед privacy-операциями.

## 5. Безопасность

Файл: `backend/app/routers/auth.py`

1. JWT bearer auth.
2. OAuth2PasswordBearer для `/api/auth/login`.
3. `get_current_user` dependency.
4. `require_admin` dependency.

Валидация:

- login length 3..64
- password min 6

## 6. Роутеры

### 6.1 Movies

Файл: `backend/app/routers/movies.py`

- GET `/api/movies/popular`
- GET `/api/movies/personal`
- GET `/api/movies/search`
- GET `/api/movies/{movie_id}/similar`
- GET `/api/movies/{movie_id}/details`
- GET `/api/movies/{movie_id}`

### 6.2 Auth

Файл: `backend/app/routers/auth.py`

- POST `/api/auth/register`
- POST `/api/auth/login`
- GET `/api/auth/me`

### 6.3 Watchlist

Файл: `backend/app/routers/watchlist.py`

- GET `/api/watchlist`
- POST `/api/watchlist`
- DELETE `/api/watchlist/{movie_id}`

### 6.4 Watched

Файл: `backend/app/routers/watched.py`

- GET `/api/watched`
- POST `/api/watched`
- DELETE `/api/watched/{movie_id}`
- GET `/api/watched/export`

### 6.5 Reviews

Файл: `backend/app/routers/reviews.py`

- GET `/api/reviews`
- POST `/api/reviews`
- GET `/api/reviews/movie/{movie_id}`
- DELETE `/api/reviews/{movie_id}`

Бизнес-правила в upsert review:

1. review/rating -> auto watched upsert
2. review/rating -> auto remove from watchlist

### 6.6 Users

Файл: `backend/app/routers/users.py`

- GET `/api/users/{user_id}/profile`
- GET `/api/users/me/privacy`
- PUT `/api/users/me/privacy`

Поведение:

- public profile возвращает 403, если профиль private.

### 6.7 Admin

Файл: `backend/app/routers/admin.py`

- GET `/api/admin/stats/overview`
- GET `/api/admin/stats/daily`
- GET `/api/admin/stats/top-movies`
- GET `/api/admin/stats/rating-distribution`
- GET `/api/admin/stats/users`

Все admin endpoints требуют admin роль.

## 7. Сервисы

### 7.1 PopularityService

- Загружает и кеширует `data/processed/movies.parquet`.
- Отдает популярные фильмы, movie lookup, search.

### 7.2 RecommenderService

- Загружает two-stage model из `data/models/two_stage_ranker`.
- Персональные рекомендации.
- Online пост-ранжирование и fold-in логика.
- Fallback на popularity при недоступности модели.

### 7.3 SimilarityService

- Загружает `data/processed/similarity_index.parquet`.
- Быстрый lookup похожих по movieId.

### 7.4 TMDBService

- Получает poster/overview/tagline/runtime/rating по tmdb id.

## 8. Ошибки и деградация

1. Нет модели -> `popularity_fallback` в personal endpoint.
2. Нет similarity index -> model=`not_available` в similar endpoint.
3. Нет TMDB details -> details endpoint возвращает null-поля, но не падает.

## 9. Operational notes

1. При изменении схемы users privacy использовать миграции + defensive guard.
2. После retrain обновлять similarity index.
3. Для диагностики использовать `/api/docs` и backend logs.

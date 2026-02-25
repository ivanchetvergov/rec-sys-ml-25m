# Backend

FastAPI приложение, отдающее рекомендации из feature store.

## Запуск

```bash
# Локально (нужен Python-окружение с зависимостями)
make backend
# или
cd backend && uvicorn app.main:app --reload --port 8000

# Swagger UI
open http://localhost:8000/api/docs
```

## Структура

```
backend/
├── requirements.txt
├── Dockerfile
└── app/
    ├── main.py              ← FastAPI приложение, CORS, startup warm-up
    ├── schemas.py           ← Pydantic модели запросов и ответов
    ├── routers/
    │   └── movies.py        ← все эндпоинты /api/movies/...
    └── services/
        ├── popularity_service.py   ← каталог, популярные, metadata lookup
        ├── recommender_service.py  ← PersonalRecs через TwoStageRecommender
        ├── similarity_service.py   ← похожие фильмы из similarity_index.parquet
        └── tmdb_service.py         ← постеры / детали через TMDB API
```

## Зависимости (`requirements.txt`)

```
fastapi==0.111.0          ← веб-фреймворк
uvicorn[standard]==0.29.0 ← ASGI сервер
pandas==2.2.2             ← чтение parquet
pyarrow==16.0.0           ← parquet engine
pydantic==2.7.1           ← валидация данных
pydantic-settings==2.3.0  ← конфигурация из .env
python-dotenv==1.0.1      ← .env файл
```

---

## `app/main.py` — точка входа

```python
app = FastAPI(title="RecSys API", version="0.1.0", docs_url="/api/docs")
```

**CORS** настроен для `http://localhost:3000` (Next.js в dev режиме).
В production нужно поменять на реальный домен.

Зарегистрированные роутеры:

| Префикс | Модуль | Описание |
|---------|--------|----------|
| `/api/movies` | `routers/movies.py` | весь каталог, рекомендации, похожие, детали |
| `/api/health` | `main.py` | health check |

**Startup warm-up** — при старте приложения `_warm_up()` прогревает все четыре сервиса:

```python
@app.on_event("startup")
async def _warm_up():
    get_popularity_service()._ensure_movies_loaded()   # movies.parquet
    get_recommender_service()._ensure_loaded()          # TwoStageRecommender
    get_similarity_service()._ensure_loaded()           # similarity_index.parquet
```

Все сервисы загружаются один раз при старте, последующие запросы обслуживаются из памяти за < 1 ms.

---

## `app/schemas.py` — Pydantic схемы

### `Movie`

```python
class Movie(BaseModel):
    id: int                            # MovieLens movieId
    title: str                         # "Toy Story (1995)"
    genres: Optional[str]              # "Adventure|Animation" или None
    year: Optional[int]                # 1995
    avg_rating: Optional[float]        # 3.92
    num_ratings: Optional[int]         # 49695
    popularity_score: Optional[float]  # 15.84
    tmdb_id: Optional[int]             # для постеров
    imdb_id: Optional[str]             # "tt0114709"
```

### `PopularMoviesResponse`

```python
class PopularMoviesResponse(BaseModel):
    total_returned: int
    offset: int
    total_available: Optional[int]
    movies: list[Movie]
```

### `MovieDetails` (TMDB-обогащённые данные)

```python
class MovieDetails(BaseModel):
    id: int
    title: str
    overview: Optional[str]       # описание от TMDB
    poster_url: Optional[str]     # https://image.tmdb.org/...
    backdrop_url: Optional[str]
    tagline: Optional[str]
    runtime: Optional[int]        # минуты
    tmdb_rating: Optional[float]
    tmdb_votes: Optional[int]
    release_date: Optional[str]
```

### `PersonalRec` / `PersonalRecsResponse`

```python
class PersonalRec(BaseModel):
    id: int; score: float; title: ...; genres: ...; ...

class PersonalRecsResponse(BaseModel):
    user_id: int
    model: str          # "two_stage" | "popularity_fallback"
    total_returned: int
    movies: list[PersonalRec]
```

### `SimilarMoviesResponse`

```python
class SimilarMoviesResponse(BaseModel):
    movie_id: int
    model: str          # "als_cosine" | "not_available"
    total_returned: int
    movies: list[Movie]
```

---

## Сервисы

### `PopularityService`

Читает `data/processed/movies.parquet` — лёгкий файл (~17K строк, ~1 MB), извлечённый из feature store командой `make extract-movies`.

**Основные методы:**

| Метод | Описание |
|-------|----------|
| `get_popular(limit, offset)` | Топ-N фильмов по popularity score |
| `get_movie(movie_id)` | Поиск одного фильма по id |
| `get_tmdb_id(movie_id)` | tmdbId для запросов к TMDB |
| `total_count()` | Общее число фильмов (пагинация) |

### `RecommenderService`

Загружает `TwoStageRecommender` из `data/models/two_stage_ranker/`. При недоступности модели или холодном старте пользователя — fallback на `PopularityService`.

**Метод:** `get_personal_recs(user_id, n, pop_fallback)` → `(movies, model_name)`

### `SimilarityService`

Читает `data/processed/similarity_index.parquet` (3 MB). Строит `dict[movieId → list[movieId]]` для O(1) lookup.

**Метод:** `get_similar_ids(movie_id, n=20)` → `list[int]`
**Флаг:** `available` — True если ALS-индекс загружен (False = parquet не найден → пустой ответ, не ошибка)

### `TMDBService`

Проксирует запросы к `api.themoviedb.org`. Требует `TMDB_API_KEY` в `.env`.

**Метод:** `get_movie_details(tmdb_id)` → `dict` с `poster_url`, `overview`, `tagline`, `runtime`, `tmdb_rating`

---

## `app/routers/movies.py` — эндпоинты

| Метод | Путь | Сервис | Описание |
|-------|------|--------|----------|
| GET | `/api/movies/popular` | PopularityService | Топ по popularity score |
| GET | `/api/movies/personal` | RecommenderService | Персональные рекомендации |
| GET | `/api/movies/{id}/similar` | SimilarityService + PopularityService | Похожие фильмы (ALS) |
| GET | `/api/movies/{id}/details` | TMDBService | TMDB-обогащённые детали |
| GET | `/api/movies/{id}` | PopularityService | Один фильм по id |

> **Порядок маршрутов важен:** `/popular` и `/personal` должны быть зарегистрированы **до** `/{movie_id}`, иначе FastAPI трактует слово "popular" как movie_id.

---

## Добавление нового эндпоинта — пример

```python
# app/routers/movies.py

@router.get("/search")
def search_movies(
    q: str = Query(min_length=1),
    limit: int = Query(20, le=100),
    service: PopularityService = Depends(get_popularity_service),
):
    all_movies = service.get_popular(limit=20_000)
    q_lower = q.lower()
    results = [m for m in all_movies if q_lower in m["title"].lower()]
    return {"query": q, "results": results[:limit]}
```

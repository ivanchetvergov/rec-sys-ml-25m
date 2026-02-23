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
    ├── main.py              ← FastAPI приложение, CORS, роуты
    ├── schemas.py           ← Pydantic модели запросов и ответов
    ├── routers/
    │   └── movies.py        ← эндпоинты /api/movies/...
    └── services/
        └── popularity_service.py  ← бизнес-логика популярности
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
| `/api/movies` | `routers/movies.py` | каталог фильмов |
| `/api/health` | `main.py` | health check |

---

## `app/schemas.py` — Pydantic схемы

### `Movie`

```python
class Movie(BaseModel):
    id: int                        # MovieLens movieId
    title: str                     # "Toy Story (1995)"
    genres: Optional[str]          # "Adventure|Animation|Children" или None
    year: Optional[int]            # 1995
    avg_rating: Optional[float]    # 3.92
    num_ratings: Optional[int]     # 49695
    popularity_score: Optional[float]  # 15.8432
```

### `PopularMoviesResponse`

```python
class PopularMoviesResponse(BaseModel):
    total_returned: int     # сколько фильмов в этом ответе
    offset: int             # смещение (для пагинации)
    movies: list[Movie]     # список фильмов
```

---

## `app/services/popularity_service.py` — бизнес-логика

### Класс `PopularityService`

Читает feature store parquet, извлекает уникальные фильмы, сортирует по `movie_popularity`.

**Инициализация:**

```python
service = PopularityService(
    feature_store_path=Path("data/processed/feature_store"),  # опционально
    dataset_tag="ml_v_20260215_184134"
)
```

**Метод `get_popular(limit, offset) → list[dict]`:**

- Первый вызов: читает `train.parquet`, загружает только нужные 7 колонок, дедуплицирует по `movieId`, сортирует по `movie_popularity` DESC. Занимает ~400ms.
- Последующие вызовы: возвращает срез из `self._movies` за < 1ms.
- `limit` ограничен максимум 100.

**Колонки, загружаемые из parquet:**

```
movieId, title, genres, year,
movie_avg_rating, movie_num_ratings, movie_popularity
```

Остальные 49 колонок из 56 не читаются — экономия памяти.

### Синглтон через `@lru_cache`

```python
@lru_cache(maxsize=1)
def get_popularity_service() -> PopularityService:
    return PopularityService()
```

Используется как FastAPI Dependency. Создаётся один раз при первом HTTP запросе.

---

## `app/routers/movies.py` — роутер

```
GET /api/movies/popular
```

Использует `Depends(get_popularity_service)` — FastAPI сам управляет lifetime синглтона.

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
    all_movies = service.get_popular(limit=1000)
    q_lower = q.lower()
    results = [m for m in all_movies if q_lower in m["title"].lower()]
    return {"query": q, "results": results[:limit]}
```

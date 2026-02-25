# REST API Reference

Base URL: `http://localhost:8000`
Swagger UI: `http://localhost:8000/api/docs`
ReDoc: `http://localhost:8000/api/redoc`

---

## Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/api/health` | Health check |
| GET | `/api/movies/popular` | Топ популярных фильмов |
| GET | `/api/movies/personal` | Персональные рекомендации (iALS + CatBoost) |
| GET | `/api/movies/{id}` | Один фильм по id |
| GET | `/api/movies/{id}/similar` | Похожие фильмы (ALS cosine) |
| GET | `/api/movies/{id}/details` | TMDB-обогащённые детали (постер, обзор) |

---

### `GET /api/health`

Проверка работоспособности сервера.

**Ответ `200`:**

```json
{ "status": "ok" }
```

---

### `GET /api/movies/popular`

Возвращает список самых популярных фильмов, отсортированных по popularity score.

**Формула:** `popularity = avg_rating × log(1 + num_ratings)`

#### Query параметры

| Параметр | Тип | По умолчанию | Ограничения | Описание |
|----------|-----|--------------|-------------|----------|
| `limit` | int | `20` | 1 ≤ x ≤ 100 | Количество фильмов |
| `offset` | int | `0` | ≥ 0 | Смещение для пагинации |

#### Примеры запросов

```bash
# Топ-20 (по умолчанию)
curl http://localhost:8000/api/movies/popular

# Топ-40
curl http://localhost:8000/api/movies/popular?limit=40

# Страница 2 (позиции 21–40)
curl http://localhost:8000/api/movies/popular?limit=20&offset=20
```

#### Ответ `200 OK`

```json
{
  "total_returned": 2,
  "offset": 0,
  "movies": [
    {
      "id": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "year": 1994,
      "avg_rating": 4.43,
      "num_ratings": 97999,
      "popularity_score": 16.1084
    },
    {
      "id": 296,
      "title": "Pulp Fiction (1994)",
      "genres": "Comedy|Crime|Drama|Thriller",
      "year": 1994,
      "avg_rating": 4.20,
      "num_ratings": 92406,
      "popularity_score": 15.8432
    }
  ]
}
```

#### Поля ответа

| Поле | Тип | Nullable | Описание |
|------|-----|----------|----------|
| `total_returned` | int | — | Количество фильмов в этом ответе |
| `offset` | int | — | Переданное смещение |
| `movies[].id` | int | — | MovieLens `movieId` |
| `movies[].title` | string | — | Название с годом: `"Toy Story (1995)"` |
| `movies[].genres` | string | ✓ | Жанры через `\|`: `"Action\|Drama"`, `null` если не указаны |
| `movies[].year` | int | ✓ | Год выхода, `null` если не удалось извлечь |
| `movies[].avg_rating` | float | ✓ | Средний рейтинг [0.5, 5.0], округлён до 2 знаков |
| `movies[].num_ratings` | int | ✓ | Количество оценок от пользователей |
| `movies[].popularity_score` | float | ✓ | Популярность, округлена до 4 знаков |

#### Пагинация

```bash
# Получить первые 100 позиций (максимум за один запрос — 100)
curl "http://localhost:8000/api/movies/popular?limit=100&offset=0"
curl "http://localhost:8000/api/movies/popular?limit=100&offset=100"
curl "http://localhost:8000/api/movies/popular?limit=100&offset=200"
```

#### Коды ошибок

| Код | Причина |
|-----|---------|
| `422 Unprocessable Entity` | Неверный тип параметра или нарушение ограничений (limit > 100) |
| `500 Internal Server Error` | Feature store parquet не найден |

---

## Формат ошибок `422`

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["query", "limit"],
      "msg": "Input should be less than or equal to 100",
      "input": "200"
    }
  ]
}
```

---

## Планируемые эндпоинты (следующие этапы)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/auth/register` | Регистрация пользователя |
| `POST` | `/api/auth/login` | Вход, получение JWT |
| `GET` | `/api/movies/search?q=` | Поиск по названию |
| `POST` | `/api/movies/{id}/interaction` | Лайк / оценка / watchlist |
| `GET` | `/api/admin/stats` | Статистика платформы |

---

### `GET /api/movies/personal`

Персональные рекомендации для пользователя. Использует двухстадийный пайплайн iALS + CatBoost. При cold-start или отсутствии модели — popularity fallback.

#### Query параметры

| Параметр | Тип | По умолчанию | Ограничения | Описание |
|----------|-----|--------------|-------------|----------|
| `user_id` | int | **обязателен** | — | MovieLens userId |
| `limit` | int | `20` | 1 ≤ x ≤ 100 | Количество рекомендаций |

#### Ответ `200 OK`

```json
{
  "user_id": 123,
  "model": "two_stage",
  "total_returned": 20,
  "movies": [
    {
      "id": 318,
      "score": 0.924,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "year": 1994,
      "avg_rating": 4.43,
      "num_ratings": 97999,
      "popularity_score": 16.1084,
      "tmdb_id": 278
    }
  ]
}
```

Поле `model`: `"two_stage"` — использована ML-модель, `"popularity_fallback"` — колд-старт или модель не загружена.

---

### `GET /api/movies/{id}/similar`

Похожие фильмы на основе косинусного сходства ALS item-vectors. Индекс предвычислен (`make build-similarity`).

#### Query параметры

| Параметр | Тип | По умолчанию | Ограничения | Описание |
|----------|-----|--------------|-------------|----------|
| `limit` | int | `20` | 1 ≤ x ≤ 50 | Число похожих фильмов |

#### Ответ `200 OK`

```json
{
  "movie_id": 1,
  "model": "als_cosine",
  "total_returned": 20,
  "movies": [ { "id": 3114, "title": "Toy Story 2 (1999)", ... } ]
}
```

Поле `model`: `"als_cosine"` — ALS-индекс доступен, `"not_available"` — индекс не найден (запустите `make build-similarity`).

---

### `GET /api/movies/{id}/details`

TMDB-обогащённые детали для одного фильма: постер, описание, трейлер. Требует `TMDB_API_KEY` в `.env`.

#### Ответ `200 OK`

```json
{
  "id": 1,
  "title": "",
  "overview": "Led by Woody, Andy's toys live happily...",
  "poster_url": "https://image.tmdb.org/t/p/w500/...",
  "backdrop_url": "https://image.tmdb.org/t/p/w1280/...",
  "tagline": "The adventure takes off!",
  "runtime": 81,
  "tmdb_rating": 7.9,
  "tmdb_votes": 16844,
  "release_date": "1995-10-30"
}
```

Если `TMDB_API_KEY` не задан или `tmdb_id` у фильма отсутствует — возвращает `404`.

---

### `GET /api/movies/{id}`

Один фильм по MovieLens `movieId`. Возвращает стандартный объект `Movie`.

#### Ответ `404 Not Found`

```json
{ "detail": "Movie 99999 not found" }
```

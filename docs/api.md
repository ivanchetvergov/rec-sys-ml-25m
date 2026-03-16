# REST API Reference

Base URL (local): `http://localhost:8000`

Swagger: `http://localhost:8000/api/docs`

## 1. Health

### GET /api/health

Ответ:

```json
{ "status": "ok" }
```

## 2. Movies

### GET /api/movies/popular

Query:

- `limit` int, default 20, min 1, max 20000
- `offset` int, default 0, min 0

Ответ: `PopularMoviesResponse`

### GET /api/movies/personal

Query:

- `user_id` int (required)
- `limit` int, default 20, max 100

Ответ: `PersonalRecsResponse`

`model` в ответе может быть:

1. `two_stage`
2. `two_stage_live_foldin`
3. `popularity_fallback`

### GET /api/movies/search

Query:

- `q` string, default ""
- `limit` int, default 15, max 50

Ответ: `SearchResponse`

### GET /api/movies/{movie_id}/similar

Query:

- `limit` int, default 20, max 50

Ответ: `SimilarMoviesResponse`

`model`:

1. `als_cosine`
2. `genre_jaccard`
3. `not_available`

### GET /api/movies/{movie_id}/details

Ответ: `MovieDetails`

### GET /api/movies/{movie_id}

Ответ: `Movie`

## 3. Auth

### POST /api/auth/register

Body:

```json
{
  "login": "string",
  "email": "user@example.com",
  "password": "string"
}
```

Ответ: `TokenOut`

### POST /api/auth/login

Content-Type: `application/x-www-form-urlencoded`

Поля:

- `username`
- `password`

Ответ: `TokenOut`

### GET /api/auth/me

Headers:

- `Authorization: Bearer <token>`

Ответ: `UserOut`

## 4. Watchlist (auth required)

### GET /api/watchlist

Ответ: `WatchlistItem[]`

### POST /api/watchlist

Body:

```json
{
  "movie_id": 1,
  "title": "Toy Story (1995)",
  "genres": "Adventure|Animation|Children|Comedy|Fantasy",
  "year": 1995,
  "avg_rating": 3.89,
  "num_ratings": 57309,
  "popularity_score": 42.12,
  "tmdb_id": 862,
  "imdb_id": "0114709"
}
```

Ответ: `WatchlistItem`

### DELETE /api/watchlist/{movie_id}

Ответ: 204

## 5. Watched (auth required)

### GET /api/watched

Ответ: `WatchedItem[]`

### POST /api/watched

Body аналогичен watchlist add.

Ответ: `WatchedItem`

### DELETE /api/watched/{movie_id}

Ответ: 204

### GET /api/watched/export

Возвращает joined watched + reviews для текущего пользователя.

## 6. Reviews

### GET /api/reviews (auth)

Ответ: `Review[]`

### POST /api/reviews (auth)

Body:

```json
{
  "movie_id": 1,
  "title": "Toy Story (1995)",
  "rating": 5,
  "review_text": "Great movie"
}
```

Ответ: `Review`

Побочные эффекты:

1. фильм автоматически добавляется в watched
2. фильм удаляется из watchlist

### GET /api/reviews/movie/{movie_id}

Публичный endpoint, ответ: `MovieReviewOut[]`

### DELETE /api/reviews/{movie_id} (auth)

Ответ: 204

## 7. Users / Profile

### GET /api/users/{user_id}/profile

Публичный профиль.

Если профиль private -> 403.

### GET /api/users/me/privacy (auth)

Ответ:

```json
{ "is_profile_private": false }
```

### PUT /api/users/me/privacy (auth)

Body:

```json
{ "is_profile_private": true }
```

Ответ:

```json
{ "is_profile_private": true }
```

## 8. Admin (admin role required)

### GET /api/admin/stats/overview

Сводные метрики.

### GET /api/admin/stats/daily?days=30

Дневные агрегаты активности.

### GET /api/admin/stats/top-movies?limit=8

Most watched и top rated.

### GET /api/admin/stats/rating-distribution

Распределение оценок 1..5.

### GET /api/admin/stats/users?limit=50&offset=0

Пагинированный список пользователей с activity counters.

## 9. Коды ошибок

1. 401: invalid/expired token
2. 403: admin required или private profile
3. 404: entity not found
4. 422: validation error
5. 500: server/internal errors

## 10. Пример auth flow

```bash
# Register
curl -X POST http://localhost:8000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"login":"demo","email":"demo@example.com","password":"secret123"}'

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=demo&password=secret123'

# Use token
curl http://localhost:8000/api/auth/me \
  -H 'Authorization: Bearer <TOKEN>'
```

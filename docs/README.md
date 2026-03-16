# RecSys Documentation Hub

Актуальная документация по проекту MovieLens 25M recommender platform.

Этот каталог отражает текущее состояние кода: двухстадийная ML-модель, FastAPI backend, Next.js frontend, пользовательские действия (watched, watchlist, reviews), публичные профили и privacy.

## Карта документации

| Файл | Назначение |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Сквозная архитектура: слои, потоки данных, runtime контуры |
| [models.md](models.md) | Подробное описание ML-моделей, артефактов, метрик, online-логики |
| [ml-pipeline.md](ml-pipeline.md) | Data/feature pipeline, train flow, reproducibility, проверка качества |
| [backend.md](backend.md) | Структура backend, сервисы, auth, БД, миграции, operational notes |
| [api.md](api.md) | REST API reference по всем публичным и защищенным endpoint |
| [frontend.md](frontend.md) | Архитектура frontend, страницы, state/auth, интеграция с API |

## Что изменилось относительно старой документации

1. Обновлены endpoint и бизнес-правила:
   - reviews now imply watched
   - rated movie removed from watchlist
   - privacy endpoints для профиля
   - admin statistics endpoints
2. Уточнен runtime recommender flow:
   - базовая two-stage inference
   - quasi-live post-ranking
   - live fold-in user embedding (без полного retrain)
3. Актуализированы команды Makefile и зависимости между шагами.

## Быстрый старт

```bash
# 1) Data pipeline
make preprocess
make extract-movies

# 2) Train production recommender
make train-ranker-sample   # быстрый прогон
# или
make train-ranker          # полный прогон

# 3) Build similarity index
make build-similarity

# 4) Run stack
make web

# 5) Open services
# App:     http://localhost
# Backend: http://localhost:8000/api/docs
# MLflow:  make mlflow-ui
```

## Рекомендуемый порядок чтения

1. [ARCHITECTURE.md](ARCHITECTURE.md)
2. [models.md](models.md)
3. [ml-pipeline.md](ml-pipeline.md)
4. [backend.md](backend.md)
5. [api.md](api.md)
6. [frontend.md](frontend.md)

## Источник истины

При расхождениях между docs и кодом источником истины является код в:

- backend/app/routers
- backend/app/services
- src/models
- src/training
- frontend/src/lib/api.ts

Документация в этом каталоге синхронизирована с текущим состоянием репозитория на 2026-03-16.

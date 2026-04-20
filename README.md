# RecSys

Двухэтапная рекомендательная система фильмов с полным стеком:

- Offline ML-контур: feature store, обучение, оценка, артефакты
- Online backend на FastAPI: персональные рекомендации, похожие фильмы, пользовательские действия
- Frontend на Next.js: каталог, рекомендации, профиль, рейтинги и отзывы

## 1. Назначение

Проект решает три задачи:

1. Строит артефакты рекомендаций из данных MovieLens.
1. Обслуживает рекомендации и пользовательские сценарии через API.
1. Предоставляет веб-интерфейс для взаимодействия и сбора обратной связи.

## 2. Архитектура

Поток данных:

1. Сырые CSV -> препроцессинг -> feature store (`data/processed/feature_store`).
1. Обучение iALS + CatBoost ranker -> артефакты (`data/models/two_stage_ranker`).
1. Построение каталога фильмов и индекса похожести (`data/processed/movies.parquet`, `data/processed/similarity_index.parquet`).
1. Backend загружает артефакты на старте и обрабатывает запросы.
1. Frontend вызывает API и записывает обратную связь (`watched`, `watchlist`, `reviews`) в Postgres.

Ключевая документация:

- `docs/ARCHITECTURE.md`
- `docs/ml-pipeline.md`
- `docs/backend.md`
- `docs/frontend.md`

## 3. Структура репозитория

```text
.
├── backend/                    # FastAPI: роутеры, сервисы, миграции
├── frontend/                   # Next.js приложение
├── src/                        # ML pipeline, модели, скрипты обучения
├── data/
│   ├── raw/                    # Сырые данные
│   ├── processed/              # Feature store, каталог, индекс похожести
│   └── models/                 # Обученные артефакты моделей
├── docs/                       # Техническая документация
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## 4. Требования

- Python 3.11+
- Node.js 18+
- Docker и Docker Compose (для полного запуска стека)
- `make`

Примечания:

- В `Makefile` используются локальные пути к Python. Перед запуском обновите `PYTHON` и `PIP` под ваше окружение.
- Исходные файлы MovieLens должны находиться в `data/raw/ml-25m`.

## 5. Быстрый старт

### Вариант A: полный стек в Docker (рекомендуется)

1. Подготовьте артефакты:

```bash
make preprocess
make train-ranker
make build-similarity
```

1. Запустите стек:

```bash
make web
```

1. Откройте:

- Frontend через nginx: `http://localhost`
- Swagger API: `http://localhost/api/docs`

1. Остановите стек:

```bash
make web-down
```

### Вариант B: локальный запуск backend/frontend

1. Установите Python-зависимости:

```bash
make install
```

1. Запустите backend:

```bash
make backend
```

1. Запустите frontend (в отдельном терминале):

```bash
make frontend
```

## 6. Основные рабочие сценарии

### 6.1 Подготовка данных и артефактов

```bash
make preprocess
make extract-movies
make train-ranker
make build-similarity
```

### 6.2 Базовые модели и эксперименты

```bash
make train-popularity
make train-cf
make train-als
make train-ranker
```

Быстрые прогоны на сэмпле:

```bash
make train-popularity-sample
make train-cf-sample
make train-als-sample
make train-ranker-sample
```

### 6.3 MLflow

```bash
make mlflow-ui
```

UI по умолчанию: `http://localhost:5000`

## 7. API backend

Основные группы роутов под `/api`:

- `movies`: popular, personal, search, similar, details
- `auth`: register, login, me
- `watched`
- `watchlist`
- `reviews`
- `users` (публичный профиль и приватность)
- `admin` (статистика)

Точка входа backend: `backend/app/main.py`

## 8. Runtime-поведение рекомендаций

- Базовый режим: двухэтапная модель (`iALS` кандидаты -> `CatBoost` rerank)
- Online-адаптация: user fold-in и пост-ранжирование на свежих взаимодействиях
- Режим деградации: `popularity_fallback`, если модель/артефакты недоступны

Маркер модели в ответах API:

- `two_stage`
- `two_stage_live_foldin`
- `popularity_fallback`

## 9. Жизненный цикл данных

1. Взаимодействия поступают в препроцессинг.
1. Выполняется временной split и построение фич без leakage.
1. Формируются и версионируются артефакты обучения.
1. Backend обслуживает online-рекомендации из загруженных артефактов.
1. Пользовательский feedback записывается в Postgres и влияет на online-ранжирование.
1. Полное обновление модели выполняется в следующем цикле retrain.

## 10. Типовые проблемы

1. Пустые персональные рекомендации: проверьте артефакты в `data/models/two_stage_ranker` и логи backend на fallback.

1. Не работают похожие фильмы: пересоберите индекс командой `make build-similarity`.

1. Backend не стартует: проверьте подключение к Postgres и применение миграций в `backend/migrations`.

1. Frontend не видит API: проверьте `NEXT_PUBLIC_API_URL` и маршрутизацию nginx в Docker-конфигурации.

## 11. Очистка

```bash
make clean
```

Команда удаляет обработанные данные feature store и артефакты MLflow.

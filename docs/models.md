# ML Models

Справочник по всем ML-моделям проекта: назначение, алгоритм, обучение, хранение артефактов и точка интеграции с бэкендом.

---

## Обзор

| Модель | Класс | Тип | Задача | Backend-сервис |
|--------|-------|-----|--------|----------------|
| PopularityRecommender | `src/models/popularity_based.py` | Non-personalized baseline | Популярные фильмы | `PopularityService` |
| SVDRecommender | `src/models/collaborative_filtering.py` | Matrix factorization (SVD) | Reference baseline | — (не используется в продакшне) |
| ALSRecommender | `src/models/als_recommender.py` | Implicit ALS | Stage 1 — генерация кандидатов | `RecommenderService` (внутри TwoStage) |
| CatBoostRanker | `src/models/catboost_ranker.py` | Gradient boosting (LTR) | Stage 2 — переранжирование | `RecommenderService` (внутри TwoStage) |
| TwoStageRecommender | `src/models/two_stage_recommender.py` | Pipeline ALS + CatBoost | Персональные рекомендации | `RecommenderService` |
| Similarity Index | `src/pipeline/build_similarity_index.py` | ALS косинусное сходство | Похожие фильмы | `SimilarityService` |

---

## 1. PopularityRecommender

**Файл:** `src/models/popularity_based.py`
**Обучение:** `src/training/train_popularity.py`

### Зачем

Непersonalized baseline — один рейтинг популярности на весь каталог. Используется в двух местах:

- Главная страница: блок **Trending Now** (топ по popularity score)
- Cold-start fallback в `TwoStageRecommender` и `RecommenderService`: если пользователь не встречался при обучении — возвращается этот список

### Алгоритм

$$\text{popularity} = w_r \cdot \bar{r} + w_c \cdot \log(1 + n)$$

где $\bar{r}$ — средний рейтинг фильма, $n$ — число оценок, $w_r = w_c = 1.0$.

### Обучение

```bash
make train-popularity-sample  # 10% данных, ~30 сек
make train-popularity          # полный датасет, ~2 мин
```

MLflow эксперимент: `popularity_baseline`

Логирует: `dataset_tag`, `min_ratings`, `seed` → `val/test_precision/recall/ndcg_at_10`

### Артефакты

```
data/models/popularity_baseline/
├── item_stats.parquet   ← movieId + popularity_score
└── config.json          ← гиперпараметры + dataset_tag
```

### Backend-сервис: `PopularityService`

**Файл:** `backend/app/services/popularity_service.py`

```
Запрос GET /api/movies/popular
         ↓
PopularityService.get_popular(limit, offset)
  • При первом вызове читает movies.parquet (~17K строк, ~1 MB)
  • Сортирует по movie_popularity DESC
  • Кеширует в памяти через @lru_cache(maxsize=1)
  • Повторные вызовы < 1 ms
         ↓
list[dict] → PopularMoviesResponse → JSON
```

Дополнительно `PopularityService` реализует:

- `get_movie(movie_id)` — поиск одного фильма по id (используется `SimilarityService`)
- `get_tmdb_id(movie_id)` — для запросов к TMDB
- `total_count()` — для пагинации

---

## 2. SVDRecommender

**Файл:** `src/models/collaborative_filtering.py`
**Обучение:** `src/training/train_collaborative.py`

### Зачем

Reference-baseline для сравнения с iALS. Не используется в продакшне — заменён двухстадийным пайплайном. Оставлен как учебный пример и точка сравнения в MLflow.

### Алгоритм

Truncated SVD на центрированной rating-матрице:

$$R_{centered} \approx U \Sigma V^T, \quad \hat{r}_{ui} = \mu_u + (U_u \cdot \Sigma) \cdot V_i$$

Библиотека: `scipy.sparse.linalg.svds` — без новых зависимостей.

### Обучение

```bash
make train-cf-sample  # 10% данных
make train-cf          # полный датасет
```

MLflow эксперимент: `collaborative_filtering`

### Сравнение с другими моделями

| Модель | nDCG@10 | Coverage | Cold-start |
|--------|---------|----------|------------|
| PopularityRecommender | 0.05 | ~0.1% | ✅ |
| SVDRecommender | 0.13 | ~4% | ❌ |
| ALSRecommender | ~0.16 | ~6% | ❌ |
| **TwoStageRecommender** | **~0.30+** | **~10%** | **fallback** |

---

## 3. ALSRecommender (iALS) — Stage 1

**Файл:** `src/models/als_recommender.py`
**Обучение:** `src/training/train_als.py` (standalone) или внутри `train_ranker.py`

### Зачем

Stage 1 двухстадийного пайплайна — **генератор кандидатов**. За ~1–2 ms достаёт top-300 наиболее релевантных фильмов из 17K для конкретного пользователя. Эти кандидаты затем передаются CatBoostRanker.

Дополнительно: векторы `item_factors` (17,695 × 128) используются для вычисления **item-item косинусного сходства** → блок "Похожие фильмы".

### Алгоритм

Алгоритм Hu, Koren & Volinsky (2008). Каждый рейтинг → confidence:

| `confidence_mode` | Формула |
|-------------------|---------|
| `linear` | $c_{ui} = 1 + \alpha \cdot r_{ui}$ |
| `log` | $c_{ui} = 1 + \alpha \cdot \log(1 + r_{ui})$ |
| `binary` | $c_{ui} = \alpha$ |

Оптимизирует weighted matrix factorization:

$$\min_{U,V} \sum_{u,i} c_{ui}(p_{ui} - U_u V_i^T)^2 + \lambda(\|U\|^2 + \|V\|^2)$$

Решается методом Alternating Least Squares — по очереди фиксирует $U$ и находит $V$, затем наоборот.

### Параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `factors` | `128` | Размерность латентного пространства |
| `iterations` | `20` | Число итераций ALS |
| `regularization` | `0.01` | L2 коэффициент |
| `alpha` | `15.0` | Масштаб confidence |
| `confidence_mode` | `linear` | Функция преобразования рейтинга → confidence |

### Обучение

```bash
# Standalone ALS (для экспериментов с Stage 1)
make train-als-sample   # 10% данных, ~1-3 мин
make train-als          # полный датасет, ~5-20 мин

# В составе двухстадийного пайплайна (рекомендуется)
make train-ranker-sample
make train-ranker
```

MLflow эксперимент: `als_candidate_generator` (или nested run внутри `two_stage_ranker`)

### Артефакты

```
data/models/two_stage_ranker/als/
├── implicit_model.pkl  ← обученная модель implicit.als.AlternatingLeastSquares
│                          .item_factors  shape (17695, 128) float32  ← ключевой артефакт
│                          .user_factors  shape (N_users, 128) float32
└── id_maps.pkl         ← {user_id_map, item_id_map, idx_to_user, idx_to_item}
```

### Интеграция в бэкенд

ALS напрямую **не загружается** в API при каждом запросе. Используется двумя путями:

1. Через `TwoStageRecommender.load()` → `RecommenderService` → `GET /api/movies/personal`
2. `item_factors` → `build_similarity_index.py` (offline) → `similarity_index.parquet` → `SimilarityService` → `GET /api/movies/{id}/similar`

---

## 4. CatBoostRanker — Stage 2

**Файл:** `src/models/catboost_ranker.py`
**Обучение:** `src/training/train_ranker.py`

### Зачем

Stage 2 — **переранжирование** кандидатов от ALS. Принимает 300 кандидатов с их признаками и выдаёт точный top-K. Обучается с loss-функцией YetiRank (pairwise LTR оптимизирует nDCG напрямую). Ключевое преимущество: учитывает контекстные признаки пользователя и фильма, которые ALS не видит.

### Признаки (38 итого)

**User features (7):**
`user_avg_rating`, `user_rating_std`, `user_num_ratings`, `user_min_rating`, `user_max_rating`, `user_activity_days`, `user_rating_velocity`

**Item features (30):**
`movie_avg_rating`, `movie_rating_std`, `movie_num_ratings`, `movie_num_users`, `movie_popularity`, `year`, `movie_age`, `decade`, `title_length`, `num_genres` + 20 бинарных жанровых признаков

**Retrieval feature (1):**
`als_score` — оценка от Stage 1 (самый важный признак по feature importance)

### Как строится обучающий датасет

Для каждого пользователя (выборка `max_ranker_users` из train):

- **positives** = фильмы с rating ≥ `relevance_threshold` (по умолчанию 4.0)
- **hard negatives** = ALS top-`n_candidates` без positives (сложные случаи, которые модель "почти" выбрала)
- **label** = 1 если positive, иначе 0
- **group** = `userId` (обязательно для YetiRank)

### Параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `iterations` | `500` | Число деревьев |
| `learning_rate` | `0.05` | Шаг градиентного бустинга |
| `depth` | `6` | Глубина дерева |
| `loss_function` | `YetiRank` | LTR loss |
| `early_stopping_rounds` | `50` | Остановка при отсутствии улучшения |

### Объяснимость (SHAP)

```python
explanation = ranker.explain(X_row)
# → {'als_score': 0.41, 'movie_popularity': 0.28, 'user_avg_rating': 0.15, ...}
# top-3 признака с их SHAP-вкладами
```

Включается через `explain=True` в `TwoStageRecommender.recommend()`.

### Артефакты

```
data/models/two_stage_ranker/ranker/
├── catboost_model.cbm         ← бинарная модель CatBoost
├── model_config.json          ← гиперпараметры, feature_names, best_iteration
└── feature_importances.csv    ← важность признаков (SHAP-based)
```

---

## 5. TwoStageRecommender — production pipeline

**Файл:** `src/models/two_stage_recommender.py`
**Обучение:** `src/training/train_ranker.py`

### Зачем

Связывает ALS и CatBoost в единый inference-пайплайн. Это то, что реально обслуживает запросы на персональные рекомендации в продакшне.

### Inference pipeline

```
GET /api/movies/personal?user_id=123
           ↓
RecommenderService._ensure_loaded()
           ↓
TwoStageRecommender.recommend(user_id=123, n=20)
           │
           ├─ [Stage 1] ALSRecommender.recommend_with_scores()
           │     → top-300 candidate_ids + als_scores   ~1-2 ms
           │
           ├─ [Stage 2] _build_feature_matrix()
           │     user_features[123] ⊕ item_features[candidates] ⊕ als_scores
           │
           ├─ CatBoostRanker.predict(X)
           │     → scores[300]   ~5-10 ms
           │     → argsort → top-20
           │
           └─ [Fallback] если user_id не в ALS → PopularityService.get_popular()
           ↓
list[dict] → PersonalRecsResponse → JSON
```

### Команды

```bash
make train-ranker-sample   # 10% данных, ~5-10 мин
make train-ranker          # полный датасет, ~30-60 мин

# Кастомный запуск
python -m src.training.train_ranker \
    --dataset-tag ml_v_20260215_184134 \
    --als-factors 128 --als-confidence-mode log \
    --ranker-iterations 600 \
    --n-candidates 300 --max-ranker-users 10000
```

MLflow эксперимент: `two_stage_ranker`

Логирует:

- **params:** все ALS + Ranker гиперпараметры, `dataset_tag`, `seed`
- **metrics:** `val/test_precision/recall/ndcg/map/coverage` @5/10/20
- **artifacts:** `als/`, `ranker/`, `user_features.parquet`, `item_features.parquet`, `feature_importances.csv`, `training_summary.json`

### Артефакты (полная структура)

```
data/models/two_stage_ranker/
├── als/
│   ├── implicit_model.pkl    ← ALSRecommender (item/user factors)
│   └── id_maps.pkl           ← idx↔id маппинги
├── ranker/
│   ├── catboost_model.cbm    ← CatBoostRanker
│   ├── model_config.json
│   └── feature_importances.csv
├── user_features.parquet     ← 7 user-level признаков (indexed by userId)
├── item_features.parquet     ← 30 item-level признаков (indexed by movieId)
└── metadata.json             ← dataset_tag, metrics, timestamp
```

### Backend-сервис: `RecommenderService`

**Файл:** `backend/app/services/recommender_service.py`

- Загружает `TwoStageRecommender.load(model_dir)` при первом запросе
- `model_available: bool` — флаг для мониторинга
- При ошибке загрузки или cold-start пользователе → fallback на `PopularityService`
- Синглтон через `@lru_cache(maxsize=1)`

```
GET /api/movies/personal?user_id=123&limit=20
         ↓
RecommenderService.get_personal_recs(user_id, n, pop_fallback)
  → (movies: list[dict], model_name: "two_stage" | "popularity_fallback")
         ↓
PersonalRecsResponse → JSON
```

---

## 6. Similarity Index (item-item ALS)

**Файл:** `src/pipeline/build_similarity_index.py`

### Зачем

Блок **"Похожие фильмы"** на странице фильма. Показывает 20–24 наиболее похожих фильма на основе косинусного сходства ALS-векторов.

Не вычисляется онлайн (17K×17K матрица — ~1.16 GB RAM, загрузка `implicit` занимает секунды) — вместо этого **предварительно вычисляется** один раз и сохраняется в компактный parquet (~3 MB). Бэкенд просто читает его при старте и отвечает за O(1).

### Algo

1. Загрузить `item_factors` из `implicit_model.pkl` — shape `(17695, 128)`
2. L2-нормализовать: $\hat{v}_i = v_i / \|v_i\|$
3. Батчевое матричное умножение: `batch @ normed.T` → косинусное сходство
4. `np.argpartition` для top-20 на батч без полной сортировки
5. `idx_to_item` для перевода индексов → `movieId`
6. Fallback: если `implicit_model.pkl` не найден → Jaccard по жанрам

### Сборка индекса

```bash
make build-similarity   # ~1-2 мин, пересобирать только при переобучении ALS
```

### Артефакт

```
data/processed/similarity_index.parquet
  movieId           int64
  similar_ids       list[int]    ← top-20 movieId по убыванию сходства
  similarity_scores list[float]  ← соответствующие косинусные оценки
```

### Backend-сервис: `SimilarityService`

**Файл:** `backend/app/services/similarity_service.py`

- Читает `similarity_index.parquet` при первом запросе → `dict[movieId → list[movieId]]`
- `available: bool` — True если ALS-индекс загружен, False если parquet не найден
- `get_similar_ids(movie_id, n)` — O(1) lookup
- Синглтон через `@lru_cache(maxsize=1)`
- Graceful degradation: если parquet не найден → логирует warning, возвращает пустой список

```
GET /api/movies/{id}/similar?limit=20
         ↓
SimilarityService.get_similar_ids(movie_id, n=20)
  → [movieId_1, movieId_2, ...]
         ↓
PopularityService.get_movie(mid) для каждого id → metadata
         ↓
SimilarMoviesResponse(model="als_cosine" | "not_available") → JSON
```

---

## Зависимости между моделями

```
                    train_ranker.py
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ALSRecommender       CatBoostRanker
        (Stage 1)            (Stage 2)
              │                     │
              └──────────┬──────────┘
                         ▼
               TwoStageRecommender
                         │
                    artifacts/
              ┌──────────┴──────────┐
              ▼                     ▼
    RecommenderService      build_similarity_index.py
    (personal recs API)              │
                                     ▼
                           SimilarityService
                           (similar movies API)
```

---

## Быстрый старт (полный цикл от нуля)

```bash
# 1. Препроцессинг и feature store
make preprocess

# 2. Извлечь каталог фильмов (лёгкий parquet для бэкенда)
make extract-movies

# 3. Обучить production-модель
make train-ranker           # ~30-60 мин на полном датасете
# или для быстрого теста:
make train-ranker-sample    # ~5-10 мин

# 4. Собрать индекс похожих фильмов (из обученных ALS-векторов)
make build-similarity       # ~1-2 мин

# 5. Запустить сервис
make web                    # Docker: backend + frontend + nginx

# 6. Посмотреть эксперименты
make mlflow-ui              # http://localhost:5000
```

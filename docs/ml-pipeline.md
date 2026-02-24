# ML Pipeline

Полный ETL/ELT пайплайн от сырых CSV до готового feature store и обученной модели.

## Запуск

```bash
# Полный пайплайн (препроцессинг + feature store)
make preprocess

# Обучить PopularityRecommender на 10% данных (для быстрой проверки)
make train-popularity-sample

# Обучить PopularityRecommender на полных данных
make train-popularity

# Обучить SVD Collaborative Filtering на 10% данных
make train-cf-sample

# Обучить SVD Collaborative Filtering на полных данных
make train-cf

# Посмотреть эксперименты в MLflow UI
make mlflow-ui        # http://localhost:5000
make mlflow-ui-alt    # http://localhost:5001 (если 5000 занят AirPlay)
```

---

## Структура `src/`

```
src/
├── config.py                        ← константы и пути
├── data_loader.py                   ← загрузка CSV в DataFrame
├── preprocessor.py                  ← фильтрация, очистка, merge
├── feature_engineer.py              ← создание 56 признаков
├── data_splitter.py                 ← temporal split 70/15/15
├── feature_store.py                 ← сохранение parquet + metadata.json
├── pipeline/
│   └── preprocess_pipeline.py      ← оркестратор (5 шагов)
├── models/
│   ├── popularity_based.py         ← PopularityRecommender
│   └── collaborative_filtering.py  ← SVDRecommender
├── training/
│   ├── train_popularity.py         ← baseline, MLflow experiment: popularity_baseline
│   └── train_collaborative.py      ← SVD CF, MLflow experiment: collaborative_filtering
└── evaluation/
    └── metrics.py                  ← Precision@K, Recall@K, nDCG@K, MAP@K
```

---

## Шаг 1 — Загрузка данных (`data_loader.py`)

Класс `DataLoader` читает CSV файлы MovieLens 25M:

| Файл | Строк | Колонки |
|------|-------|---------|
| `ratings.csv` | 25M | userId, movieId, rating, timestamp |
| `movies.csv` | 62K | movieId, title, genres |
| `links.csv` | 62K | movieId, imdbId, tmdbId |

```python
loader = DataLoader()
datasets = loader.load_all(load_tags=False, load_links=False)
```

---

## Шаг 2 — Препроцессинг (`preprocessor.py`)

Класс `Preprocessor` с итеративной фильтрацией:

1. Удалить рейтинги вне диапазона [0.5, 5.0]
2. Оставить только пользователей с ≥ 20 рейтингами (`MIN_USER_RATINGS`)
3. Оставить только фильмы с ≥ 10 рейтингами (`MIN_MOVIE_RATINGS`)
4. Повторять пп. 2–3 до стабилизации (обычно 3 итерации)
5. Join movies → добавить title, genres

**Результат:** ~24.9M строк, потери < 1%.

---

## Шаг 3 — Feature Engineering (`feature_engineer.py`)

Класс `FeatureEngineer` создаёт **56 признаков**:

### Признаки фильма

| Признак | Описание |
|---------|----------|
| `genre_drama`, `genre_comedy`, … | 20 бинарных признаков жанра |
| `year` | Год выхода (из названия) |
| `movie_age` | Возраст фильма на момент рейтинга |
| `decade` | Десятилетие (1990, 2000, ...) |
| `movie_avg_rating` | Средний рейтинг по всем пользователям |
| `movie_num_ratings` | Количество оценок |
| `movie_popularity` | `avg_rating × log(1 + num_ratings)` |

### Признаки пользователя

| Признак | Описание |
|---------|----------|
| `user_avg_rating` | Средний рейтинг пользователя |
| `user_num_ratings` | Количество оценок пользователя |
| `user_rating_std` | Стандартное отклонение рейтингов |
| `user_activity_days` | Дней между первой и последней оценкой |
| `user_rating_velocity` | Оценок в день |

### Признаки взаимодействия

| Признак | Описание |
|---------|----------|
| `rating_deviation_user` | Рейтинг − средний рейтинг пользователя |
| `rating_deviation_movie` | Рейтинг − средний рейтинг фильма |
| `interaction_month` | Месяц оценки (сезонность) |
| `interaction_day_of_week` | День недели |

---

## Шаг 4 — Сплит (`data_splitter.py`)

**Временной сплит** — критически важен для рекомендательных систем:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  старые взаимодействия            │  вал  │  тест
  TRAIN (70%)                       │  15%  │  15% новые
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   17.4M строк           3.7M строк   3.7M строк
```

Разделение по **timestamp**, а не случайно — модель обучается на прошлом и предсказывает будущее.

---

## Шаг 5 — Feature Store (`feature_store.py`)

Класс `FeatureStore` сохраняет в `data/processed/feature_store/<dataset_tag>/`:

```
ml_v_20260215_184134/
├── train.parquet        ← 17.4M строк, 56 колонок, ~500MB
├── val.parquet          ← 3.7M строк
├── test.parquet         ← 3.7M строк
├── train_statistics.csv ← средние/std по каждому признаку
└── metadata.json        ← конфигурация, размеры, dtypes
```

`dataset_tag` формат: `ml_v_YYYYMMDD_HHMMSS`.

---

## Модель — PopularityRecommender (`models/popularity_based.py`)

Непersonalized baseline. Ранжирует фильмы по:

$$\text{popularity} = w_r \cdot \bar{r} + w_c \cdot \log(1 + n)$$

где $\bar{r}$ — средний рейтинг, $n$ — количество оценок, $w_r = w_c = 1.0$.

### Методы

| Метод | Описание |
|-------|----------|
| `fit(train_df)` | Вычислить popularity score по всем фильмам |
| `recommend(n, exclude)` | Вернуть top-N movieId |
| `recommend_batch(user_ids, seen_items, n)` | Батч-рекомендации |
| `save(path)` | Сохранить `item_stats.parquet` + `config.json` |
| `load(path)` | Загрузить сохранённую модель |

### Обучение с MLflow

```bash
make train-popularity-sample
```

Логирует в MLflow:

- **params:** `dataset_tag`, `min_ratings`, `rating_weight`, `count_weight`, `seed`
- **metrics:** `val_precision_at_10`, `val_recall_at_10`, `val_ndcg_at_10`, `test_*`
- **artifacts:** `item_stats.parquet`, `config.json`

---

## Метрики оценки (`evaluation/metrics.py`)

Все метрики используют нотацию `_at_` (MLflow не поддерживает `@`).

| Функция | Формула |
|---------|---------|
| `precision_at_k(recs, relevant, k)` | $\frac{\|recs_k \cap relevant\|}{k}$ |
| `recall_at_k(recs, relevant, k)` | $\frac{\|recs_k \cap relevant\|}{\|relevant\|}$ |
| `ndcg_at_k(recs, relevant, k)` | $\frac{DCG_k}{IDCG_k}$ |
| `map_at_k(recs, relevant, k)` | среднее AP по пользователям |
| `coverage(all_recs, catalog)` | доля каталога в рекомендациях |

### Результаты текущей модели (10% выборка)

| Метрика | Val | Test |
|---------|-----|------|
| Precision@10 | 0.0418 | 0.0421 |
| Recall@10 | 0.0410 | 0.0325 |
| nDCG@10 | 0.0528 | 0.0512 |
| Coverage@10 | 0.09% | — |

Низкое покрытие (0.09%) ожидаемо — непersonalized модель всегда рекомендует одни и те же топ-фильмы.

---

## Модель — SVDRecommender (`models/collaborative_filtering.py`)

Персонализированная модель матричной факторизации через **Truncated SVD**.

### Как работает

Построим центрированную матрицу $R_{centered}$ (рейтинги минус среднее пользователя), затем разложим:

$$R_{centered} \approx U \Sigma V^T$$

Score для неизвестной пары $(u, i)$:

$$\hat{r}_{ui} = \mu_u + (U_u \cdot \Sigma) \cdot V_i$$

где $\mu_u$ — средний рейтинг пользователя (bias).

**Используемые библиотеки:** `scipy.sparse.linalg.svds` — входит в `scipy`, новых зависимостей не требуется.

### Параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--factors` | `100` | Число латентных факторов $k$ |
| `--sample-frac` | `None` | Доля train-данных (None = полный датасет) |
| `--relevance-threshold` | `4.0` | Минимальный рейтинг для положительного примера |
| `--max-eval-users` | `1000` | Число пользователей для оценки на val/test |
| `--seed` | `42` | Фиксация случайности |

### Методы

| Метод | Описание |
|-------|----------|
| `fit(train_df)` | Построить sparse-матрицу, запустить svds, сохранить факторы |
| `recommend(user_id, n, exclude_items)` | Top-N через dot product + argsort |
| `recommend_batch(user_ids, seen_items, n)` | Батч-рекомендации |
| `score(user_id, item_id)` | Предсказанный рейтинг с восстановленным bias |
| `save(path)` | `user_factors.npy`, `item_factors.npy`, `id_maps.pkl`, `model_config.json` |
| `load(path)` | Загрузить сохранённую модель |

**Cold-start:** пользователь не встречался в train → возвращается `[]`; рекомендуется фолбэк на `PopularityRecommender`.

### Обучение с MLflow

```bash
# Быстрый прогон — 10% данных, 50 факторов (~2-5 мин)
make train-cf-sample

# Полный прогон — весь датасет, 100 факторов (~20-40 мин)
make train-cf

# Или напрямую с кастомными параметрами
python -m src.training.train_collaborative \
    --dataset-tag ml_v_20260215_184134 \
    --factors 150 \
    --sample-frac 0.2 \
    --max-eval-users 2000 \
    --run-name svd_k150_exp1
```

Логирует в MLflow (эксперимент `collaborative_filtering`):

- **params:** `model_type`, `dataset_tag`, `factors`, `sample_frac`, `seed`, `relevance_threshold`
- **metrics:** `val_precision_at_5/10/20`, `val_recall_at_5/10/20`, `val_ndcg_at_5/10/20`, `val_map_at_5/10/20`, `val_coverage_at_5/10/20`, и аналогичные `test_*`
- **artifacts:** `user_factors.npy`, `item_factors.npy`, `id_maps.pkl`, `model_config.json`, `training_summary.json`

### Загрузка сохранённой модели

```python
from src.models.collaborative_filtering import SVDRecommender

model = SVDRecommender.load("path/to/artifacts")

# Рекомендации для одного пользователя
recs = model.recommend(user_id=123, n=10, exclude_items={100, 200})

# Батч для нескольких пользователей
seen = {123: {100, 200}, 456: {300}}
recs_batch = model.recommend_batch([123, 456], n=10, exclude_items_per_user=seen)
```

### Сравнение моделей

| Модель | Тип | Coverage | Скорость обучения | Холодный старт |
|--------|-----|----------|-------------------|---------------|
| `PopularityRecommender` | Non-personalized | ~0.1% | < 1 сек | ✅ Все пользователи |
| `SVDRecommender` | Matrix Factorization | > 5% | 2–40 мин | ❌ Только in-matrix |

---

## Конфигурация (`config.py`)

```python
MIN_USER_RATINGS = 20   # минимум оценок для включения пользователя
MIN_MOVIE_RATINGS = 10  # минимум оценок для включения фильма
TRAIN_SPLIT = 0.70      # доля тренировочных данных
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
TOP_GENRES  = 20        # сколько жанров брать как бинарные признаки
```

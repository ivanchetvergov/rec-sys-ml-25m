# ML Pipeline

Подробное описание data pipeline от сырых MovieLens CSV до production-ready артефактов.

## 1. Основные цели

1. Реплицируемая подготовка данных без leakage.
2. Версионирование feature store.
3. Воспроизводимое обучение моделей.
4. Сохранение артефактов для online inference.

## 2. Команды

```bash
# end-to-end preprocessing
make preprocess

# catalogue extraction for backend
make extract-movies

# model training
make train-popularity
make train-cf
make train-als
make train-ranker

# fast dev runs
make train-popularity-sample
make train-cf-sample
make train-als-sample
make train-ranker-sample

# similarity index
make build-similarity
```

## 3. Preprocessing pipeline

Файл: src/pipeline/preprocess_pipeline.py

Шаги:

1. Load datasets
2. Preprocess and merge
3. Temporal split
4. No-leakage feature engineering
5. Save feature store

### 3.1 Почему split до feature engineering

В pipeline сначала делается temporal split, и только потом строятся features. Это предотвращает data leakage: статистики train не должны знать про val/test.

## 4. Feature store

Путь:

`data/processed/feature_store/<dataset_tag>`

Содержимое:

- train.parquet
- val.parquet
- test.parquet
- metadata/statistics файлы

`dataset_tag` задает версию набора данных и используется в train скриптах.

## 5. Training pipeline (two-stage)

Файл: src/training/train_ranker.py

Этапы:

1. Load train/val/test
2. Build user/item lookup feature tables
3. Train ALS
4. Build ranker dataset (hard + random negatives)
5. Train CatBoost ranker
6. Evaluate end-to-end
7. Save artifacts + MLflow log

## 6. Артефакты и их назначение

### 6.1 Для backend каталогов

src/pipeline/extract_movies.py -> data/processed/movies.parquet

Это легкий справочник фильмов с метаданными и идентификаторами.

### 6.2 Для similar movies

src/pipeline/build_similarity_index.py -> data/processed/similarity_index.parquet

Файл нужен SimilarityService для быстрых ответов в runtime.

### 6.3 Для персональных рекомендаций

data/models/two_stage_ranker:

- ALS модель и маппинги
- CatBoost модель и конфиг
- user/item feature tables

## 7. MLflow

Запуск UI:

```bash
make mlflow-ui
```

Логируются:

1. params
2. metrics
3. artifacts

Ключевые эксперименты:

- als_candidate_generator
- two_stage_ranker

## 8. Оценка качества

Метрики из src/evaluation/metrics.py:

1. precision@k
2. recall@k
3. ndcg@k
4. map@k
5. coverage@k

Рекомендуемый протокол:

1. смотреть val/test одновременно
2. сравнивать не только ndcg, но и coverage
3. фиксировать k-values одинаково между экспериментами

## 9. Reproducibility checklist

1. Зафиксирован dataset_tag.
2. Зафиксирован seed.
3. Зафиксированы hyperparams.
4. Run сохранен в MLflow с артефактами.
5. Зафиксирована версия кода (git commit).

## 10. Частые ошибки

1. Обучили модель, но не пересобрали similarity index.
2. Запустили backend без movies.parquet.
3. Смешали старый dataset_tag и новые model artifacts.
4. Изменили feature schema без полного retrain.

## 11. Production последовательность

```bash
make preprocess
make extract-movies
make train-ranker
make build-similarity
make web
```

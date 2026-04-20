# ML Models: подробный справочник

## 1. Модельный стек

Проект использует гибридный production-стек:

1. Retrieval: ImplicitALSRecommender (iALS)
2. Re-ranking: CatBoostRanker (LTR, YetiRank)
3. Pipeline orchestration: TwoStageRecommender
4. Fallback: popularity ranking

Дополнительно:

- Similar items индекс из ALS item factors (cosine)
- Quasi-live post-ranking в backend
- Live fold-in  user embedding без полного retrain

## 2. ImplicitALSRecommender

Файл: src/models/ials_recommender.py

### 2.1 Назначение

Быстрый генератор кандидатов (stage 1), оптимизированный под implicit feedback.

### 2.2 Математика

Используется схема Hu, Koren, Volinsky:

$$
\min_{U,V} \sum_{u,i} c_{ui}(p_{ui} - U_u V_i^T)^2 + \lambda(\|U\|^2 + \|V\|^2)
$$

где confidence строится из ratings в одном из режимов:

1. linear: $c_{ui} = 1 + \alpha r_{ui}$
2. log: $c_{ui} = 1 + \alpha \log(1+r_{ui})$
3. binary: константная уверенность для наблюдений

### 2.3 Важные методы

1. fit(...)
2. recommend_with_scores(...)
3. recommend_with_recalculated_user(...)

Последний метод поддерживает online fold-in: пересчет user representation при фиксированных item factors на основании свежих взаимодействий.

### 2.4 Артефакты

`data/models/two_stage_ranker/als`:

- implicit_model.pkl
- id_maps.pkl

## 3. CatBoostRanker

Файл: src/models/catboost_ranker.py

### 3.1 Назначение

Переранжирование candidate pool с использованием rich feature space.

### 3.2 Признаки

1. User features
2. Item features
3. Retrieval feature: als_score
4. Cross-features user x item

Кросс-фичи задаются централизованно через CROSS_FEATURE_DEFINITIONS и добавляются как в train, так и в inference.

### 3.3 Explainability

Поддерживается SHAP объяснение через explain(top_n=3).

### 3.4 Артефакты

`data/models/two_stage_ranker/ranker`:

- catboost_model.cbm
- ranker_config.json
- feature_importances.csv

## 4. TwoStageRecommender

Файл: src/models/two_stage_recommender.py

### 4.1 Назначение

Объединение retrieval + ranking в едином inference API.

### 4.2 Inference pipeline

1. ALS top-N candidates
2. feature matrix assembly
3. CatBoost score
4. sort desc and return top-K
5. cold-start fallback при отсутствии candidates

### 4.3 Артефакты pipeline

`data/models/two_stage_ranker`:

- als/
- ranker/
- user_features.parquet
- item_features.parquet
- pipeline_config.json

## 5. Similarity index

Файл: src/pipeline/build_similarity_index.py

### 5.1 Основной режим

Cosine similarity по item_factors ALS.

### 5.2 Fallback

Если ALS артефактов нет, строится жанровый Jaccard index.

### 5.3 Артефакт

`data/processed/similarity_index.parquet`:

- movieId
- similar_ids
- similarity_scores

## 6. Training orchestration

Файл: src/training/train_ranker.py

### 6.1 Этапы

1. load feature store
2. train ALS
3. build ranker dataset
4. train CatBoost
5. evaluate val/test
6. log to MLflow
7. save artifacts

### 6.2 Ranker dataset логика

Для каждого пользователя:

1. positives: rating >= threshold
2. ALS candidates: hard negatives + часть positives
3. uniform negatives: случайные easy negatives
4. group id: userId (для YetiRank)

## 7. Online personalization in backend

Файл: backend/app/services/recommender_service.py

### 7.1 Quasi-live post-ranking

1. Exclude seen (watched + reviews)
2. Similar boost from recent seed movies
3. Genre preference boost

### 7.2 Live fold-in path

При наличии свежих сигналов и доступных ALS factors возможно пересчитать online user vector и ранжировать кандидаты с обновленными признаками.

## 8. Метрики и качество

Проект использует стандартные ranking метрики из src/evaluation/metrics.py:

1. precision@k
2. recall@k
3. ndcg@k
4. map@k
5. coverage@k

MLflow логирует val/test метрики по k-values из train запуска.

## 9. Команды

```bash
# train full
make train-ranker

# train sample
make train-ranker-sample

# inspect experiments
make mlflow-ui
```

## 10. Практические рекомендации

1. После каждого retrain выполнять make build-similarity.
2. Для smoke test использовать sample run.
3. Для production фиксировать dataset tag и seed.
4. При изменении feature schema retrain обязателен.

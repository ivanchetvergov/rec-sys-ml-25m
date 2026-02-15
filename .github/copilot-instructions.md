# Copilot Instructions for Digital Analytics Recommender System

## Project Overview
Full-stack recommender system with separation: ML logic, infrastructure/deployment, web frontend. AI assists across all layers.

## Tech Stack
- **ML**: Python, NumPy, Pandas, Scikit-learn, CatBoost; MLflow for logging; Pytest for testing; Git for versioning.
- **Infrastructure**: Docker, Kubernetes/Docker Compose; Terraform; AWS/GCP; ELK/Prometheus for monitoring.
- **Frontend**: React/TypeScript, Next.js; Tailwind CSS; Redux/Zustand; Axios/React Query; Jest/Cypress; Vite.

## Code Conventions
- **Python**: PEP8, black, isort; type hints; mypy. Docstrings: Google/NumPy.
- **TypeScript**: Strict mode, ESLint, Prettier. Functional components/hooks.
- **Naming**: snake_case (Python), camelCase (TS/JS), PascalCase (classes).
- **Architecture**: SOLID, SRP. Comments explain "why". DRY, KISS principles.

## ML Development Guidelines
- **Training**: Fix seeds (numpy, torch, random, sklearn); require `dataset_tag`; time-based splits for recsys.
- **Experiments**: Log to MLflow (params, metrics, artifacts, signatures); tag runs with `dataset_tag`, `seed`.
- **Artifacts**: Generate model files, metadata JSON ({model_id, schemas, metrics, resources}), endpoint specs (POST /predict).
- **Explainability**: SHAP for rankers; output {item_id, score, explanation: top3 features}.
- **Testing**: Unit (logic/metrics), integration (pipeline), data (schema checks with Great Expectations).

## Infrastructure & Deployment Guidelines
- **Containerization**: Docker multi-stage; load ML artifacts; health checks.
- **Workflow**: Receive artifacts → build/test locally → deploy to cloud (load balancers, auto-scaling) → monitor/alert.
- **Security**: No hardcoded secrets (env vars/AWS Secrets); least privilege; HTTPS; sanitize inputs.

## Frontend Development Guidelines
- **UI/UX**: Responsive, mobile-first, accessible (WCAG); integrate APIs for recommendations/explanations.
- **Visualization**: D3.js/Recharts for explanations; real-time updates (WebSocket/polling).
- **Workflow**: Receive API specs → build components (lists, cards) → fetch/render → test (unit/E2E/accessibility).

## Integration & Communication
- **Handoffs**: ML → Infra (artifacts/metadata); Infra → Frontend (endpoints/schemas).
- **API Example**: POST /predict {user_id, item_ids, features} → {predictions: [{item_id, score, explanation}]}; <100ms latency.

## Best Practices
- Clarify uncertainties: Ask for data paths, URIs, limits, contracts.
- Propose MVPs on sample data.
- PRs: Summary, code/tests, metrics, QA checklist; require 1 review.
- No assumptions on formats/constraints.

## Common Patterns
### Model Training
```python
def train_model(dataset_tag: str, seed: int = 42):
    np.random.seed(seed)
    # ... logic
    mlflow.log_param("dataset_tag", dataset_tag)
```

### Batch Inference
```python
def predict_batch(data: pd.DataFrame) -> List[float]:
    return model.predict(data)
```

### Frontend API Call
```typescript
const fetchRecs = async (userId: number, itemIds: number[]) =>
  (await axios.post('/predict', { user_id: userId, item_ids: itemIds })).data.predictions;
```

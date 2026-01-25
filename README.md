# Model Release Platform (Self-hosted)

This repo is an end-to-end, production-shaped MLflow project:
- MLflow Tracking Server (HTTP)
- Postgres backend store
- MinIO (S3-compatible) artifact store
- Pipeline: ingest -> featurize -> train -> evaluate -> register (to **Staging**)
- Promotion step: **Staging -> Production**
- Serving: FastAPI always serves `models:/<MODEL_NAME>/Production`
- Smoke test hits the service

## Prereqs
- Docker + Docker Compose

## Run
```bash
make up
make run-pipeline
make promote
make serve
make smoke-test
```

## URLs
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001 (user/pass: minioadmin/minioadmin)
- Serving API: http://localhost:8000 (GET /health, POST /predict)

## Notes
- Default model name: `breast_cancer_clf`
- Dataset: `sklearn.datasets.load_breast_cancer` (no external download)

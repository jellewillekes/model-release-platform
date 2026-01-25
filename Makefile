SHELL := /bin/bash
PROJECT_NAME := mlflow-e2e-platform

.PHONY: up down logs run-pipeline promote serve smoke-test clean

up:
	docker compose up -d postgres minio mlflow-server minio-init
	@echo "MLflow UI: http://localhost:5050"
	@echo "MinIO Console: http://localhost:9001 (user: minioadmin / pass: minioadmin)"

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

run-pipeline:
	docker compose run --rm pipeline

promote:
	docker compose run --rm promote

serve:
	docker compose up -d serving
	@echo "Serving API: http://localhost:8000 (GET /health, POST /predict)"

smoke-test:
	docker compose run --rm smoke

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ || true

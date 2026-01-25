SHELL := /bin/bash
PROJECT_NAME := model-release-platform

.PHONY: up down logs build run-pipeline promote serve smoke-test clean reset

# Start core infra services
up:
	docker compose up -d postgres minio mlflow-server minio-init
	@echo "MLflow UI: http://localhost:5050"
	@echo "MinIO Console: http://localhost:9001 (user: minioadmin / pass: minioadmin)"

# Stop everything and wipe volumes
down:
	docker compose down -v

# Full nuke: containers, volumes, and images
reset: down
	docker compose build --no-cache pipeline promote serving smoke

# Tail logs
logs:
	docker compose logs -f --tail=200

# Build all runtime images
build:
	docker compose build pipeline promote serving smoke

# Run training + registration pipeline (rebuild first)
run-pipeline: build
	docker compose run --rm pipeline

# Promote candidate -> prod (rebuild first)
promote: build
	docker compose run --rm promote

# Start serving API
serve: build
	docker compose up -d --build serving
	@echo "Serving API: http://localhost:8000 (GET /health, POST /predict)"

# Run smoke tests against serving API
smoke-test: build
	docker compose run --rm --build smoke

# Clean local Python junk
clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ || true

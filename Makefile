# Makefile
# - Fast-by-default local gates
# - Self-documenting `make help`
# - Deterministic tool execution via uv
# - Safe shell flags 

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

PROJECT_NAME := model-release-platform

# Paths
UV_PROJECT_DIR := project

# Tooling
UV ?= uv
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose

# Always run tools against the uv project in ./project
UV_RUN := $(UV) run --project $(UV_PROJECT_DIR)

PY ?= $(UV_RUN) python

RUFF ?= $(PY) -m ruff
MYPY ?= $(PY) -m mypy
PYTEST ?= $(PY) -m pytest
PRECOMMIT ?= $(PY) -m pre_commit

MYPY_CONFIG ?= mypy.ini
MYPY_PATHS ?= project/src serving
PYTEST_ARGS ?= -q

# Docker compose service names (centralize to avoid drift)
SVC_INFRA := postgres minio mlflow-server minio-init
SVC_IMAGES := pipeline promote serving smoke

# Helpers
.PHONY: help
help:
	@echo ""
	@echo "$(PROJECT_NAME)"
	@echo ""
	@echo "Local quality gates:"
	@echo "  make check          Run format+lint+type+test (fast gate)"
	@echo "  make fix            Auto-fix formatting + safe lint fixes"
	@echo "  make precommit      Run all pre-commit hooks on all files"
	@echo "  make install-hooks  Install git pre-commit hooks"
	@echo ""
	@echo "Docker workflow:"
	@echo "  make up             Start infra services ($(SVC_INFRA))"
	@echo "  make down           Stop everything and wipe volumes"
	@echo "  make logs           Tail docker compose logs"
	@echo "  make build          Build runtime images ($(SVC_IMAGES))"
	@echo "  make reset          Full nuke: down + no-cache rebuild"
	@echo "  make run-pipeline   Run training + registration pipeline"
	@echo "  make promote        Promote candidate -> prod"
	@echo "  make serve          Start serving API"
	@echo "  make smoke-test     Run smoke tests against serving API"
	@echo ""
	@echo "Housekeeping:"
	@echo "  make clean          Remove local caches"
	@echo ""

# Local quality gates
.PHONY: check format lint type test fix

check: format lint type test
	@echo "✅ All checks passed"

format:
	@$(RUFF) format --check .

lint:
	@$(RUFF) check .

type:
	@$(MYPY) --config-file $(MYPY_CONFIG) $(MYPY_PATHS)

test:
	@$(PYTEST) $(PYTEST_ARGS)

fix:
	@$(RUFF) format .
	@$(RUFF) check --fix .

# Pre-commit
.PHONY: precommit install-hooks

precommit:
	@$(PRECOMMIT) run --all-files

install-hooks:
	@$(PRECOMMIT) install

# Docker / local infra
.PHONY: up down logs build reset run-pipeline promote serve smoke-test

up:
	@$(COMPOSE) up -d $(SVC_INFRA)
	@echo "MLflow UI: http://localhost:5050"
	@echo "MinIO Console: http://localhost:9001 (user: minioadmin / pass: minioadmin)"

down:
	@$(COMPOSE) down -v

reset: down
	@$(COMPOSE) build --no-cache $(SVC_IMAGES)

logs:
	@$(COMPOSE) logs -f --tail=200

build:
	@$(COMPOSE) build $(SVC_IMAGES)

run-pipeline: build
	@$(COMPOSE) run --rm pipeline

promote: build
	@$(COMPOSE) run --rm promote

serve: build
	@$(COMPOSE) up -d --build serving
	@echo "Serving API: http://localhost:8000 (GET /health, POST /predict)"

smoke-test: build
	@$(COMPOSE) run --rm --build smoke

# Housekeeping
.PHONY: clean
clean:
	@rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ || true

SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

PROJECT_NAME := ml-lifecycle-platform

UV_PROJECT_DIR := project

UV ?= uv
DOCKER ?= docker
COMPOSE ?= $(DOCKER) compose

UV_RUN := $(UV) run --project $(UV_PROJECT_DIR)
PY := $(UV_RUN) python

RUFF := $(PY) -m ruff
MYPY := $(PY) -m mypy
PYTEST := $(PY) -m pytest
PRECOMMIT := $(PY) -m pre_commit

MYPY_CONFIG ?= mypy.ini
MYPY_PATHS ?= project/src serving
PYTEST_ARGS ?= -q

SVC_INFRA := postgres minio mlflow-server minio-init
SVC_IMAGES := pipeline promote rollback serving smoke

# ---- helpers ----
define assert_allowed_true
	out="$$($(COMPOSE) run --rm promote python -m src.promote --dry-run --format json)"; \
	echo "$$out"; \
	if command -v jq >/dev/null 2>&1; then \
		echo "$$out" | jq -e '.allowed == true' >/dev/null; \
	else \
		echo "$$out" | grep -Eq '"allowed"[[:space:]]*:[[:space:]]*true'; \
	fi
endef

.PHONY: help
help:
	@echo ""
	@echo "$(PROJECT_NAME)"
	@echo ""
	@echo "Local:"
	@echo "  make check             format+lint+type+test"
	@echo "  make fix               format + safe autofix"
	@echo "  make precommit         run all hooks"
	@echo "  make install-hooks     install git hooks"
	@echo ""
	@echo "Docker:"
	@echo "  make up                start infra ($(SVC_INFRA))"
	@echo "  make down              stop + wipe volumes"
	@echo "  make logs              tail logs"
	@echo "  make build             build runtime images ($(SVC_IMAGES))"
	@echo "  make reset             down + no-cache rebuild"
	@echo "  make run-pipeline       train+eval+register (candidate)"
	@echo "  make policy-check       dry-run promotion gate check (fails if blocked)"
	@echo "  make promote            candidate -> prod"
	@echo "  make promote-dry-run    show dry-run JSON (no side effects)"
	@echo "  make rollback-prod      prod -> previous prod"
	@echo "  make serve              start serving API"
	@echo "  make smoke-test         smoke tests against serving"
	@echo "  make e2e                pipeline -> gate -> promote -> serve -> smoke"
	@echo "  make e2e-keep           like e2e, but keep stack up"
	@echo ""
	@echo "Housekeeping:"
	@echo "  make clean             remove local caches"
	@echo ""

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

.PHONY: precommit install-hooks
precommit:
	@$(PRECOMMIT) run --all-files

install-hooks:
	@$(PRECOMMIT) install

.PHONY: up down logs build reset run-pipeline policy-check promote promote-dry-run rollback-prod serve smoke-test e2e e2e-keep
up:
	@$(COMPOSE) up -d $(SVC_INFRA)
	@echo "MLflow UI: http://localhost:5050"
	@echo "MinIO Console: http://localhost:9001 (user: minioadmin / pass: minioadmin)"

down:
	@$(COMPOSE) down -v

logs:
	@$(COMPOSE) logs -f --tail=200

build:
	@$(COMPOSE) build $(SVC_IMAGES)

reset: down
	@$(COMPOSE) build --no-cache $(SVC_IMAGES)

run-pipeline: build
	@$(COMPOSE) run --rm pipeline

# CI friendly: fails if promotion not allowed
policy-check: build
	@$(assert_allowed_true)
	@echo "✅ Policy gate passed (allowed=true)"

# promotion 
promote: build
	@$(COMPOSE) run --rm promote python -m src.promote

# explicit dry run (prints JSON)
promote-dry-run: build
	@$(COMPOSE) run --rm promote python -m src.promote --dry-run --format json

rollback-prod: build
	@$(COMPOSE) run --rm rollback

serve: build
	@$(COMPOSE) up -d --build serving
	@echo "Serving API: http://localhost:8000 (GET /health, POST /predict)"

smoke-test: build
	@$(COMPOSE) run --rm --build smoke

# E2E: prove the system end-to-end
e2e: build
	@set -euo pipefail; \
	cleanup() { $(COMPOSE) down -v; }; \
	trap cleanup EXIT; \
	$(COMPOSE) up -d $(SVC_INFRA); \
	$(COMPOSE) run --rm pipeline; \
	$(assert_allowed_true); \
	$(COMPOSE) run --rm promote; \
	$(COMPOSE) up -d --build serving; \
	$(COMPOSE) run --rm --build smoke; \
	echo "✅ E2E passed"

# Same as e2e but keeps infra+serving up for manual investigtion
e2e-keep: build
	@set -euo pipefail; \
	$(COMPOSE) up -d $(SVC_INFRA); \
	$(COMPOSE) run --rm pipeline; \
	$(assert_allowed_true); \
	$(COMPOSE) run --rm promote; \
	$(COMPOSE) up -d --build serving; \
	$(COMPOSE) run --rm --build smoke; \
	echo "✅ E2E passed (stack kept up). Use 'make logs' or 'make down' when done."

.PHONY: clean
clean:
	@rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ || true

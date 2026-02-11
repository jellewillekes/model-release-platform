# Model Release Platform

[![CI](https://github.com/jellewillekes/model-release-platform/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jellewillekes/model-release-platform/actions/workflows/ci.yml)
[![E2E](https://github.com/jellewillekes/model-release-platform/actions/workflows/e2e.yml/badge.svg?branch=master)](https://github.com/jellewillekes/model-release-platform/actions/workflows/e2e.yml)

A production-style model release platform that manages the full lifecycle of machine learning models with an emphasis on safety, reproducibility, and operational discipline.

The platform supports:

- Training and evaluation
- Quality gating
- Registry-based releases
- Alias-based promotion
- Progressive delivery (canary / shadow)
- Deterministic rollback
- Online serving
- End-to-end verification

This repository serves as a reference implementation for ML platform engineering patterns.

---

## System Guarantees

The platform enforces the following guarantees by construction:

- Reproducible training
  Every training run is tracked, versioned, and immutable.

- Quality-gated releases
  Models are only registered and promoted when evaluation criteria are met.

- Alias-based release workflow
  Models move through aliases rather than deprecated MLflow stages.

- Deterministic rollback
  Each promotion records the previous production version.

- Artifact lineage
  Every model version links back to its source run and metadata.

- Separation of concerns
  Training and serving are decoupled.

- End-to-end automation
  The full lifecycle is executable via Make targets.

- Verifiable production
  Smoke and E2E tests validate deployments.

---

## Release Model (Alias-Based)

MLflow stages are intentionally not used.

### Aliases

| Alias     | Description                    |
|-----------|--------------------------------|
| candidate | Most recent gated model         |
| prod      | Current production model        |
| champion  | Synonym for prod                |

### Promotion Guardrails

Required metadata:

- dataset_fingerprint
- git_sha
- config_hash
- training_run_id

Promotion is blocked if any tag is missing.

### Rollback Metadata

previous_prod_version=<version>

---

## Architecture Overview

### Control Plane

- Make targets
- CI/CD workflows
- Promotion gates
- Metadata validation

### Data Plane

- Training artifacts (MinIO)
- Model registry (MLflow)
- Evaluation reports
- Prediction logs

### Serving Plane

- FastAPI inference service
- Alias resolver
- Canary router
- Shadow traffic duplicator

### Lifecycle

Ingest → Featurize → Train → Evaluate → Register → Promote → Serve

Serving Path:

models:/<name>@prod → FastAPI → Clients

---

## Technology Stack

- MLflow
- PostgreSQL
- MinIO
- FastAPI
- Docker Compose
- Makefile

---

## Repository Structure
```bash
.
├── project/      # Training, evaluation, promotion
├── serving/      # Inference service
├── docs/         # Architecture and runbooks
├── scripts/      # Tooling and automation
└── .github/      # CI governance
```

---

## Quickstart (Local)

### Prerequisites

- Docker
- Docker Compose
- GNU Make
- Python 3.11+

### Start Infrastructure

```bash
make up
```

Service Endpoints:

- MLflow UI: http://localhost:5050
- MinIO Console: http://localhost:9001

### Run Training Pipeline

```bash
make run-pipeline
```

### Promote Model

```bash
make promote
```

### Start Serving

```bash
make serve
```

### Verify Deployment

```bash
make smoke-test
```

---

## End-to-End Validation

```bash
make e2e
make e2e-keep
```

---

## Rollback

```bash
make rollback-prod
```

---

## Serving Modes

### Endpoint

```bash
POST /predict?mode=prod|candidate|canary|shadow
```

### Mode Semantics

| Mode      | Behavior                               |
|-----------|----------------------------------------|
| prod      | Default production model               |
| candidate | Staging model                          |
| canary    | Partial traffic to new model           |
| shadow    | Mirrored traffic (no user impact)      |

### Canary Configuration

```bash
export CANARY_PCT=10
```

---

## Failure Handling

### Promotion Failures

- Missing metadata → promotion blocked
- Metric regression → candidate rejected

### Serving Failures

- Registry unavailable → fallback to last prod
- Canary instability → automatic rollback

### Recovery

```bash
make rollback-prod
```
---
## Local Development

```bash
make check
make fix
```
---

## Governance & Contribution

- Code ownership via CODEOWNERS
- Mandatory pull requests
- Standard PR templates
- CI enforcement

See:
- CONTRIBUTING.md
- CODEOWNERS

---

## Reproducibility

Models are reproducible from:

- Dataset fingerprints
- Config hashes
- Git SHA
- Source run ID

---

## Releases & Versioning

This project follows Conventional Commits.

- Automated changelogs
- Semantic versioning
- Release Please automation

---

## Operational Workflow

```bash
make up
make run-pipeline
make promote
make serve
make smoke-test
make e2e
```

---

## Status

Reference-grade ML platform implementation.

---

## Security & Licensing

- Security issues: see SECURITY.md
- License: MIT (see LICENSE)

# ML Lifecycle Platform

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

The platform enforces the following invariants:

- Reproducible runs  
  Every training run logs dataset fingerprint, config hash, git SHA, and is immutable.

- Quality-gated promotion  
  `candidate → prod` promotion only happens when evaluation gates pass and all required metadata is present.

- Alias-first registry model  
  Deployment is driven by MLflow aliases (`candidate`, `prod`, `champion`). Stages are not used.

- Deterministic rollback  
  Each promotion records `previous_prod_version`. Rollback is a based on alias mutation.

- Artifact lineage  
  Every model version links to its source training run and metadata.

- Control-plane / data-plane separation  
  Training, registry policy, and serving are independent.

- End-to-end verifiability  
  CI and E2E validate training, policy checks, promotion, serving, and rollback.

---

## Release Model (Alias-Based)

MLflow stages are not used.

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

### Policy Check (Non-Mutating)

Promotion can be executed in dry-run mode.

- Evaluates metadata requirements
- Verifies evaluation gate status
- Returns a structured JSON decision report
- Performs no registry writes or state changes

Example:

```bash
make policy-check
```

The output contract:

```json
{
  "allowed": true|false,
  "context": {},
  "errors": [],
  "warnings": []
}
```

CI and E2E rely on this contract.

### Rollback Metadata

```
previous_prod_version=<version>
```
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
```
Ingest → Featurize → Train → Evaluate → Register → Promote → Serve
```
Serving Path:
```
models:/<name>@prod → FastAPI → Clients
```

### Execution Flow

```
Train/Evaluate (project/)
        |
        v
MLflow Registry (PostgreSQL backend)
        |
        |-- aliases: candidate / prod / champion
        |
        v
Serving (FastAPI)
        |
        |-- prod (default)
        |-- candidate (optional)
        |-- canary (bucketed routing)
        |-- shadow (mirrored inference)
        |
        v
Clients
```

---

## Technology Stack

- MLflow
- PostgreSQL
- MinIO
- FastAPI
- Docker Compose
- Makefile

---

## Interface Contracts

### Registry Contract
- Deployment is alias-driven.
- Serving resolves `models:/<name>@prod` by default.

### Promotion Contract
- Required tags: `dataset_fingerprint`, `git_sha`, `config_hash`, `training_run_id`
- Dry-run mode outputs `{allowed, context, errors, warnings}`

### Rollback Contract
- Rollback mutates aliases only.
- No rebuild or retraining required.

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

## Local Execution 

```bash
make down && make clean && make up && make run-pipeline && make policy-check && make promote && make serve && make smoke-test && make e2e
```

Service Endpoints:

- MLflow UI: http://localhost:5050
- MinIO Console: http://localhost:9001
- Serving API: http://localhost:8000

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

### Routing Modes

| Mode      | Behavior                                                                 |
|-----------|--------------------------------------------------------------------------|
| prod      | Resolves `@prod` (default production alias)                              |
| candidate | Resolves `@candidate` if available; otherwise returns 503                |
| canary    | Deterministic traffic split between prod and candidate                   |
| shadow    | Executes candidate in parallel; response derived from prod               |

Canary routing is deterministic per request (bucketed by payload hash).

---

## Failure Handling

### Promotion Failures

- Missing required metadata → promotion blocked
- Evaluation gate failed → candidate rejected

### Serving Failures

- Alias resolution failure → HTTP 503
- Registry connectivity issues → HTTP 503

### Recovery

Rollback is explicit and deterministic:

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

## Security & Licensing

- Security issues: see SECURITY.md
- License: MIT (see LICENSE)

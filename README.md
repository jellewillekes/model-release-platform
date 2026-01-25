Model Release Platform

This repository implements a production-style model release platform
that manages the full lifecycle of machine learning models:

-   Training
-   Evaluation
-   Quality gating
-   Registration
-   Promotion
-   Serving
-   Verification

The system is designed around a release discipline,
reproducibility, and artifact lineage.

------------------------------------------------------------------------

System Guarantees

The platform enforces the following guarantees:

-   Reproducible training — every run is tracked, versioned, and
    immutable.
-   Quality-gated releases — models are only registered if evaluation
    criteria are met.
-   Explicit promotion workflow — models move through Staging →
    Production.
-   Rollback safety — previous Production versions are preserved.
-   Artifact lineage — every model version links back to data, code, and
    metrics.
-   Serving seperate from training — online inference loads only from
    the registry.
-   End-to-end automation — the entire lifecycle is executable via CLI
-   Verifiable production — smoke tests validate live deployments

------------------------------------------------------------------------

Architecture

    ┌────────────┐
    │  Ingest    │
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │ Featurize  │
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │  Train     │───► logs model + metrics
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │ Evaluate   │───► applies quality gates
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │ Register   │───► registers to Staging
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │ Promote    │───► Staging → Production
    └────────────┘

    Serving:
    Model Registry → Inference API → Clients

------------------------------------------------------------------------

Technology Stack

-   MLflow (experiment tracking + model registry)
-   PostgreSQL (metadata store)
-   MinIO (artifact store)
-   FastAPI (online inference)
-   Docker Compose (infrastructure orchestration)
-   Makefile (workflow automation)

------------------------------------------------------------------------

Repository Structure

    .
    ├── docker-compose.yml
    ├── Makefile
    ├── src/
    │   ├── orchestrate.py
    │   ├── ingest.py
    │   ├── featurize.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── register.py
    │   └── promote.py
    ├── serving/
    │   └── app.py
    └── smoke_test.py

------------------------------------------------------------------------

Operating the System

Start infrastructure

    make up

-   MLflow UI: http://localhost:5050
-   MinIO Console: http://localhost:9001

------------------------------------------------------------------------

Run full pipeline (train → evaluate → register)

    make run-pipeline

------------------------------------------------------------------------

Promote model to Production

    make promote

------------------------------------------------------------------------

Start inference service

    make serve

------------------------------------------------------------------------

Verify deployment

    make smoke-test

------------------------------------------------------------------------

Release Policy

-   A model is only registered if it passes evaluation thresholds.
-   A model is only promoted via an explicit promotion command.
-   The registry is the single source of truth for serving.
-   Serving never loads models from training outputs directly.

------------------------------------------------------------------------

Design Principles

-   Immutable artifacts
-   Registry-driven deployment
-   Promotion instead of overwrite
-   Explicit release steps
-   Full traceability
-   Automation-first operation

------------------------------------------------------------------------

Planned Extensions

-   Canary deployments
-   Shadow deployments
-   CI-based quality gates
-   Dataset fingerprinting
-   Drift detection
-   Alias-based releases
-   Automated rollback policies

------------------------------------------------------------------------


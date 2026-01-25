Model Release Platform

This repository implements a production-style model release platform that manages the full lifecycle of machine learning models:

- Training
- Evaluation
- Quality gating
- Registration
- Promotion (alias-based)
- Serving
- Verification

The system is designed around release discipline, reproducibility, and artifact lineage.

------------------------------------------------------------------------

System Guarantees

- Reproducible training — every run is tracked, versioned, and immutable.
- Quality-gated releases — models are only registered if evaluation criteria are met.
- Alias-based promotion workflow — models move through aliases rather than deprecated “stages”.
- Rollback metadata — promotions record the previous production version via model version tags.
- Artifact lineage — every model version links back to the source run id and metadata.
- Serving separate from training — online inference loads only from the registry.
- End-to-end automation — the entire lifecycle is executable via Make targets.
- Verifiable production — smoke tests validate live deployments.

------------------------------------------------------------------------

Release Model (No Stages)

We do NOT use MLflow stages (“Staging”/“Production”). Instead we use:

Aliases (pointers):
- candidate: most recently registered + gated model version
- prod: version currently served by the inference API
- champion: synonym for prod (current best / current production)

Model version tags (metadata):
- release_status: candidate | prod | champion | previous_prod
- source_run_id: originating MLflow run id
- gate: passed | failed
- previous_prod_version: (optional) version that prod used to point to
- promoted_from_alias: candidate

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
    │ Register   │───► registers model version + sets alias "candidate"
    └─────┬──────┘
          │
    ┌─────▼──────┐
    │ Promote    │───► moves alias "prod" (and "champion") to chosen version
    └────────────┘

    Serving:
    Model Registry (models:/name@prod) → Inference API → Clients

------------------------------------------------------------------------

Technology Stack

- MLflow (experiment tracking + model registry)
- PostgreSQL (metadata store)
- MinIO (artifact store)
- FastAPI (online inference)
- Docker Compose (infrastructure)
- Makefile (workflow automation)

------------------------------------------------------------------------

Repository Structure

    .
    ├── docker-compose.yml
    ├── Makefile
    ├── project/
    │   └── src/              # pipeline steps + orchestration
    ├── serving/
    │   └── app.py            # FastAPI inference
    └── smoke_test.py         # deployment verification

------------------------------------------------------------------------

Operating the System

Start infrastructure:

    make up

- MLflow UI: http://localhost:5050
- MinIO Console: http://localhost:9001

Run full pipeline (train → evaluate → register):

    make run-pipeline

Promote candidate to production (alias-based):

    make promote

Start inference service:

    make serve

Verify deployment:

    make smoke-test

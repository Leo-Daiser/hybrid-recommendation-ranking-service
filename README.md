# Hybrid Recommendation and Ranking Service

## Overview

Hybrid Recommendation and Ranking Service is a production-like recommendation platform built around a two-stage architecture:

1. Candidate generation / retrieval
2. Supervised ranking

The project uses MovieLens as the main dataset and includes a migration adapter for the MIND news recommendation dataset.

This is not a notebook-only recommender system. The repository includes data validation, temporal splitting, feature generation, retrieval baselines, ranking dataset construction, model training, offline evaluation, FastAPI serving, PostgreSQL logging, feedback logging, and lightweight pipeline orchestration.

---

## Key Features

- MovieLens raw data ingestion and schema validation
- Temporal train/validation/test split
- Implicit feedback conversion from ratings
- User and item feature generation
- Popularity-based retrieval baseline
- Item-item cosine similarity retrieval
- Offline retrieval evaluation with Precision@K, Recall@K, MAP@K, NDCG@K, Coverage@K
- Ranking dataset construction with negative sampling
- Logistic Regression and CatBoost ranker training
- FastAPI `/recommend` endpoint
- Cold-start fallback using popularity candidates
- PostgreSQL logging for recommendation requests and feedback events
- Lightweight batch pipeline orchestration
- MIND dataset adapter layer
- Unit and integration tests

---

## Architecture

```text
Raw MovieLens data
    ↓
Raw schema validation
    ↓
Temporal split
    ↓
User / item feature generation
    ↓
Candidate generation
    ├── Popularity retrieval
    └── Item-item cosine retrieval
    ↓
Offline retrieval evaluation
    ↓
Ranking dataset + negative sampling
    ↓
Ranker training
    ├── Logistic Regression
    └── CatBoost
    ↓
FastAPI recommendation service
    ↓
PostgreSQL request / feedback logging
```

## Repository Structure
```text
.
├── configs/                  # YAML configuration files
├── src/
│   ├── api/                  # FastAPI app, routes, schemas, recommender service
│   ├── core/                 # App configuration and shared utilities
│   ├── data/                 # Data loading, validation, MovieLens and MIND preparation
│   ├── db/                   # SQLAlchemy models, sessions, repositories
│   ├── evaluation/           # Retrieval and ranking evaluation
│   ├── features/             # User/item feature generation
│   ├── jobs/                 # Pipeline orchestration
│   ├── ranking/              # Ranking dataset and ranker training
│   └── retrieval/            # Popularity and item-KNN retrieval
├── tests/                    # Pytest test suite
├── sql/                      # SQL initialization scripts
├── data/                     # Local data directory, ignored by Git
├── artifacts/                # Local generated artifacts, ignored by Git
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

## Dataset

The main dataset is MovieLens ml-latest-small.

Expected files after downloading:

data/raw/movielens/ml-latest-small/
├── ratings.csv
├── movies.csv
├── tags.csv
└── links.csv

Raw datasets and generated artifacts are not committed to Git.

## Setup
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Start PostgreSQL
```bash
docker compose up -d db
```
3. Initialize database
```bash
python -m src.cli init-db
```
4. Check database connection
```bash
python -m src.cli db-check
```

## Full Pipeline

The project supports a lightweight local batch pipeline.

```bash
python -m src.cli run-full-pipeline
```

Check pipeline output status:

```bash
python -m src.cli pipeline-status
```

Manual step-by-step execution:

```bash
python -m src.cli download-data
python -m src.cli validate-raw
python -m src.cli prepare-interactions
python -m src.cli build-features
python -m src.cli build-popularity-candidates
python -m src.cli build-itemknn-candidates
python -m src.cli evaluate-retrieval
python -m src.cli build-ranking-dataset
python -m src.cli train-ranker
```

## API

Start API locally:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

Get recommendations:

```bash
curl "http://127.0.0.1:8000/recommend?user_id=1&k=10"
```

Example response:

```json
{
  "request_id": "uuid",
  "user_id": 1,
  "items": [
    {
      "item_id": 1200,
      "score": 0.0026,
      "rank": 1,
      "retrieval_score": 20.45,
      "explanation": {
        "source": "itemknn"
      }
    }
  ],
  "model_version": "ranker_logreg_v1",
  "fallback_used": false,
  "scoring_mode": "ranker",
  "warnings": []
}
```

Send feedback:

```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -H "Content-Type: application/json" \
  -d "{\"request_id\":\"REQUEST_ID\",\"user_id\":1,\"item_id\":1200,\"event_type\":\"click\"}"
```

## Retrieval Evaluation Results

Validation split results:

| Model | K | Precision | Recall | MAP | NDCG | Coverage |
|-------|---|-----------|--------|-----|------|----------|
| popularity_v1 | 10 | 0.0048 | 0.0003 | 0.0016 | 0.0052 | 0.0023 |
| itemknn_cosine_v1 | 10 | 0.0476 | 0.0478 | 0.0170 | 0.0508 | 0.0641 |
| popularity_v1 | 50 | 0.0010 | 0.0003 | 0.0003 | 0.0018 | 0.0103 |
| itemknn_cosine_v1 | 50 | 0.0343 | 0.1431 | 0.0216 | 0.0828 | 0.2202 |

Item-KNN substantially improves both ranking quality and item coverage compared to the popularity baseline.

## Ranking Evaluation Results

Validation ranking metrics:

| Model | K | Precision | Recall | MAP | NDCG |
|-------|---|-----------|--------|-----|------|
| logistic_regression | 10 | 0.3727 | 0.9545 | 0.9848 | 0.9927 |
| catboost | 10 | 0.3727 | 0.9545 | 0.9167 | 0.9386 |

Important limitation: ranking metrics should be interpreted as pipeline validation rather than final production-quality ranking performance. The ranking dataset is relatively small because candidate positives are limited by retrieval recall on the temporal validation window.

## MIND Migration Layer

The repository includes an adapter layer for MIND news recommendation data.

Expected raw layout:

data/raw/mind/MINDsmall/
├── train/
│   ├── behaviors.tsv
│   └── news.tsv
└── valid/
    ├── behaviors.tsv
    └── news.tsv

Run:

```bash
python -m src.cli prepare-mind
```

The MIND adapter converts news impressions into a common internal schema:

- user_id
- item_id
- timestamp
- label
- source
- impression_id

MIND is not yet connected to the full MovieLens retrieval/ranking pipeline. This is a migration adapter for future extension.

## Testing

Run all tests:

```bash
pytest -q
```

Current status: **155 passed**

## Generated Files

The following are generated locally and ignored by Git:

- data/raw/
- data/processed/
- artifacts/models/
- artifacts/metrics/
- artifacts/reports/
- artifacts/pipeline_runs/

## Limitations
- The ranking dataset is small, so ranker metrics should be interpreted carefully.
- MIND is implemented as an adapter layer but not yet connected to the full retrieval/ranking pipeline.
- The local orchestrator is lightweight and does not replace Airflow, Prefect, or production scheduling.
- Feedback is logged but not yet used for online learning or retraining.
- No A/B testing layer is implemented.

## Tech Stack
- Python 3.11
- pandas
- scikit-learn
- CatBoost
- scipy sparse matrices
- FastAPI
- SQLAlchemy
- PostgreSQL
- Docker Compose
- pytest
- Typer

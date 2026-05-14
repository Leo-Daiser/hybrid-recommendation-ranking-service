# Hybrid Recommendation and Ranking Service

A production-like two-stage ML system for recommendations (candidate generation + ranking).

## Current Phase: Phase 0 (Foundation Layer)

### Project Structure
- `src/`: Source code
- `configs/`: YAML configurations
- `tests/`: Tests
- `sql/`: Database initialization scripts
- `docker-compose.yml`: Infrastructure definitions

### How to Run

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Build and start services:
   ```bash
   make up
   # or docker-compose up -d --build
   ```
3. Initialize the database schema:
   ```bash
   make init-db
   # or docker-compose exec api python -m src.cli init-db
   ```
4. Run tests:
   ```bash
   make test
   # or docker-compose exec api pytest -q
   ```
5. Check health status:
   ```bash
   curl http://localhost:8000/health
   ```

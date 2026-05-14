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
   **Note**: `sql/init.sql` is used only for basic DB extensions if needed. Tables and schema are created via SQLAlchemy ORM.
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

### Implemented in Phase 4A (Current Phase)
- **Popularity Retrieval Baseline**: Implemented the Most Popular recommender system logic based on training item ratings positive ratios and counts.
- **Candidate Cache**: Generated `candidate_cache_popularity.parquet` cache storing fallback offline recommendations for users.
- **Leakage Prevention**: Cold-start logic returns top overall items, known users have all previously seen training items excluded from their candidate list.

### Available Commands

1. **Download Data**:
   ```bash
   python -m src.cli download-data
   ```
2. **Validate Raw Data**:
   ```bash
   python -m src.cli validate-raw
   ```
3. **Prepare Interactions**:
   ```bash
   python -m src.cli prepare-interactions
   ```
4. **Build Features**:
   ```bash
   python -m src.cli build-features
   ```
5. **Build Popularity Candidates (Phase 4A)**:
   ```bash
   python -m src.cli build-popularity-candidates
   ```
6. **Run Tests**:
   ```bash
   pytest -q
   ```
7. **Initialize DB**:
   ```bash
   python -m src.cli init-db
   ```

### Known Limitations (Phase 4A)
- No item-item (KNN) or ALS retrieval implemented yet.
- No ranking models implemented.
- No offline top-K evaluation.
- No real-time recommend API.


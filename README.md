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

### Implemented in Phase 3 (Current Phase)
- **User Features**: Extracted statistics (mean rating, positive ratio, activity span) strictly from the training interactions.
- **Item Features**: Extracted item statistics and metadata (genres) strictly from the training interactions and raw movies data.
- **Leakage Prevention**: Valid and Test splits are excluded from all feature building steps to ensure models do not peek into the future.
- **Artifacts Generated**: `user_features.parquet`, `item_features.parquet`, `genre_features.parquet` in `data/processed/`.

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
4. **Build Features (Phase 3)**:
   ```bash
   python -m src.cli build-features
   ```
5. **Run Tests**:
   ```bash
   pytest -q
   ```
6. **Initialize DB**:
   ```bash
   python -m src.cli init-db
   ```

### Known Limitations (Phase 3)
- No candidate generation (retrieval) implemented.
- No ranking models implemented.
- No offline top-K evaluation.
- No real-time recommend API.


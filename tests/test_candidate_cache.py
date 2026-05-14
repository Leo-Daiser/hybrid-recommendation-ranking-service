import pytest
import pandas as pd
from pathlib import Path
from src.retrieval.candidate_cache import (
    validate_candidate_cache,
    save_candidate_cache,
    load_candidate_cache
)

def test_validate_candidate_cache_success():
    cache = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [10, 20],
        "retrieval_score": [0.9, 0.8],
        "rank": [1, 2],
        "retrieval_model": ["pop", "pop"],
        "generated_at": ["today", "today"]
    })
    validate_candidate_cache(cache)

def test_validate_candidate_cache_missing_required_column():
    cache = pd.DataFrame({"user_id": [1]})
    with pytest.raises(ValueError):
        validate_candidate_cache(cache)

def test_validate_candidate_cache_duplicate_pairs():
    cache = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [10, 10],
        "retrieval_score": [0.9, 0.8],
        "rank": [1, 2],
        "retrieval_model": ["pop", "pop"],
        "generated_at": ["today", "today"]
    })
    with pytest.raises(ValueError):
        validate_candidate_cache(cache)

def test_validate_candidate_cache_duplicate_rank_per_user():
    cache = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [10, 20],
        "retrieval_score": [0.9, 0.8],
        "rank": [1, 1],
        "retrieval_model": ["pop", "pop"],
        "generated_at": ["today", "today"]
    })
    with pytest.raises(ValueError):
        validate_candidate_cache(cache)

def test_save_and_load_candidate_cache(tmp_path):
    cache = pd.DataFrame({
        "user_id": [1],
        "item_id": [10],
        "retrieval_score": [0.9],
        "rank": [1],
        "retrieval_model": ["pop"],
        "generated_at": ["today"]
    })
    p = tmp_path / "cache.parquet"
    save_candidate_cache(cache, p)
    loaded = load_candidate_cache(p)
    pd.testing.assert_frame_equal(cache, loaded)

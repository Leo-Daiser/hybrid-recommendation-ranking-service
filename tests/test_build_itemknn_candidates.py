import pytest
import pandas as pd
from pathlib import Path
import yaml
from src.retrieval.build_itemknn_candidates import run_build_itemknn_candidates

def test_run_build_itemknn_candidates_creates_outputs(tmp_path):
    cfg_path = tmp_path / "retrieval.yaml"
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    train_path = proc_dir / "train.parquet"
    pop_cache_path = proc_dir / "pop_cache.parquet"
    sim_out_path = proc_dir / "sim.parquet"
    cache_out_path = proc_dir / "knn_cache.parquet"
    
    pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]}).to_parquet(train_path)
    pd.DataFrame({
        "user_id": [1],
        "item_id": [20],
        "retrieval_score": [1.0],
        "rank": [1],
        "retrieval_model": ["pop"],
        "generated_at": ["now"]
    }).to_parquet(pop_cache_path)
    
    with open(cfg_path, "w") as f:
        yaml.dump({
            "retrieval": {
                "user_id_column": "user_id",
                "item_id_column": "item_id",
                "rating_column": "rating",
                "label_column": "label",
                "timestamp_column": "timestamp",
                "train_interactions_path": str(train_path),
                "item_features_path": "fake",
                "output_candidate_cache_path": str(pop_cache_path),
                "popularity": {
                    "top_k": 5,
                    "candidate_pool_size": 10,
                    "score_column": "item_positive_ratio",
                    "tie_breaker_column": "item_rating_count",
                    "model_name": "pop"
                },
                "cold_start": {"fallback_top_k": 5}
            },
            "item_knn": {
                "top_k": 5,
                "max_neighbors_per_item": 10,
                "min_similarity": 0.0,
                "interaction_weight": "binary_positive",
                "aggregation": "sum",
                "model_name": "itemknn",
                "output_similarity_path": str(sim_out_path),
                "output_candidate_cache_path": str(cache_out_path)
            }
        }, f)
        
    res = run_build_itemknn_candidates(cfg_path)
    assert sim_out_path.exists()
    assert cache_out_path.exists()

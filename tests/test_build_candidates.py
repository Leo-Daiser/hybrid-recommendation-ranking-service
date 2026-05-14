import pytest
import pandas as pd
from pathlib import Path
import yaml
from src.retrieval.build_candidates import run_build_popularity_candidates

def test_run_build_popularity_candidates_creates_output(tmp_path):
    cfg_path = tmp_path / "retrieval.yaml"
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    train_path = proc_dir / "train.parquet"
    item_feat_path = proc_dir / "item_features.parquet"
    out_cache_path = proc_dir / "cache.parquet"
    
    pd.DataFrame({"user_id": [1], "item_id": [10]}).to_parquet(train_path)
    pd.DataFrame({
        "item_id": [10, 20],
        "item_positive_ratio": [1.0, 0.9],
        "item_rating_count": [10, 5]
    }).to_parquet(item_feat_path)
    
    with open(cfg_path, "w") as f:
        yaml.dump({
            "retrieval": {
                "user_id_column": "user_id",
                "item_id_column": "item_id",
                "rating_column": "rating",
                "label_column": "label",
                "timestamp_column": "timestamp",
                "train_interactions_path": str(train_path),
                "item_features_path": str(item_feat_path),
                "output_candidate_cache_path": str(out_cache_path),
                "popularity": {
                    "top_k": 5,
                    "candidate_pool_size": 10,
                    "score_column": "item_positive_ratio",
                    "tie_breaker_column": "item_rating_count",
                    "model_name": "pop"
                },
                "cold_start": {"fallback_top_k": 5}
            }
        }, f)
        
    res = run_build_popularity_candidates(cfg_path)
    assert len(res) > 0
    assert out_cache_path.exists()

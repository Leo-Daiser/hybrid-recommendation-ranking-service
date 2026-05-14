import pytest
import pandas as pd
import yaml
from pathlib import Path
from src.ranking.build_ranking_dataset import run_build_ranking_datasets

def test_run_build_ranking_datasets_creates_outputs(tmp_path):
    cfg_path = tmp_path / "ranking.yaml"
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    valid_path = proc_dir / "valid.parquet"
    test_path = proc_dir / "test.parquet"
    uf_path = proc_dir / "uf.parquet"
    if_path = proc_dir / "if.parquet"
    cache1_path = proc_dir / "c1.parquet"
    cache2_path = proc_dir / "c2.parquet"
    train_out = proc_dir / "ranking_train.parquet"
    valid_out = proc_dir / "ranking_valid.parquet"
    
    pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]}).to_parquet(valid_path)
    pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]}).to_parquet(test_path)
    pd.DataFrame({"user_id": [1]}).to_parquet(uf_path)
    pd.DataFrame({"item_id": [10]}).to_parquet(if_path)
    pd.DataFrame({"user_id": [1], "item_id": [10], "rank": [1], "retrieval_score": [1.0]}).to_parquet(cache1_path)
    pd.DataFrame({"user_id": [1], "item_id": [20], "rank": [1], "retrieval_score": [1.0]}).to_parquet(cache2_path)
    
    with open(cfg_path, "w") as f:
        yaml.dump({
            "ranking": {
                "user_id_column": "user_id",
                "item_id_column": "item_id",
                "label_column": "label",
                "input": {
                    "interactions_train_path": "fake",
                    "interactions_valid_path": str(valid_path),
                    "interactions_test_path": str(test_path),
                    "user_features_path": str(uf_path),
                    "item_features_path": str(if_path),
                    "candidate_caches": {
                        "pop": str(cache1_path),
                        "knn": str(cache2_path)
                    }
                },
                "output": {
                    "ranking_train_path": str(train_out),
                    "ranking_valid_path": str(valid_out)
                },
                "labels": {
                    "positive_label": 1,
                    "target_column": "target"
                },
                "candidates": {
                    "use_models": ["pop", "knn"],
                    "deduplicate_user_item_pairs": True
                },
                "negative_sampling": {
                    "enabled": True,
                    "negative_to_positive_ratio": 5,
                    "random_seed": 42,
                    "keep_all_positives": True
                }
            }
        }, f)
        
    res = run_build_ranking_datasets(cfg_path)
    assert train_out.exists()
    assert valid_out.exists()

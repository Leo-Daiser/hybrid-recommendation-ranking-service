import pytest
import pandas as pd
from pathlib import Path
import yaml
from src.features.build_features import run_build_features

def test_run_build_features_creates_outputs(tmp_path):
    data_cfg_path = tmp_path / "data.yaml"
    feat_cfg_path = tmp_path / "features.yaml"
    
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    movies_path = raw_dir / "movies.csv"
    pd.DataFrame({
        "movieId": [1, 2],
        "title": ["A", "B"],
        "genres": ["Action", "Comedy"]
    }).to_csv(movies_path, index=False)
    
    train_path = proc_dir / "train.parquet"
    pd.DataFrame({
        "user_id": [100, 100, 200],
        "item_id": [1, 2, 1],
        "rating": [5.0, 3.0, 4.0],
        "label": [1, 0, 1],
        "timestamp": [10, 20, 30]
    }).to_parquet(train_path)
    
    with open(data_cfg_path, "w") as f:
        yaml.dump({
            "dataset": {"raw_data_dir": str(raw_dir)},
            "tables": {"movies": {"filename": "movies.csv"}}
        }, f)
        
    with open(feat_cfg_path, "w") as f:
        yaml.dump({
            "features": {
                "user_id_column": "user_id",
                "item_id_column": "item_id",
                "original_user_id_column": "userId",
                "original_item_id_column": "movieId",
                "rating_column": "rating",
                "label_column": "label",
                "timestamp_column": "timestamp"
            },
            "input": {
                "train_interactions_path": str(train_path),
                "movies_table": "movies"
            },
            "output": {
                "user_features_path": str(proc_dir / "user_features.parquet"),
                "item_features_path": str(proc_dir / "item_features.parquet"),
                "genre_features_path": str(proc_dir / "genre_features.parquet")
            },
            "item_features": {
                "unknown_genre_token": "(no genres listed)"
            }
        }, f)
        
    res = run_build_features(data_cfg_path, feat_cfg_path)
    
    assert "user_features" in res
    assert "item_features" in res
    assert "genre_features" in res
    
    assert (proc_dir / "user_features.parquet").exists()
    assert (proc_dir / "item_features.parquet").exists()
    assert (proc_dir / "genre_features.parquet").exists()

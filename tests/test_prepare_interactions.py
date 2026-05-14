import pytest
import pandas as pd
from pathlib import Path
import yaml
from src.data.prepare_interactions import (
    create_implicit_labels,
    temporal_global_split,
    prepare_interactions,
    save_interaction_splits
)

def test_create_implicit_labels_success():
    df = pd.DataFrame({"rating": [3.0, 4.0, 5.0, 2.0]})
    res = create_implicit_labels(df, "rating", "label", 4.0)
    assert "label" in res.columns
    assert res["label"].tolist() == [0, 1, 1, 0]

def test_create_implicit_labels_missing_rating_column():
    df = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(ValueError):
        create_implicit_labels(df, "rating", "label", 4.0)

def test_create_implicit_labels_existing_label_column_raises():
    df = pd.DataFrame({"rating": [3.0], "label": [1]})
    with pytest.raises(ValueError):
        create_implicit_labels(df, "rating", "label", 4.0)

def test_temporal_global_split_success():
    df = pd.DataFrame({"timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    train, valid, test = temporal_global_split(df, "timestamp", 0.8, 0.1, 0.1)
    assert len(train) == 8
    assert len(valid) == 1
    assert len(test) == 1
    assert train["timestamp"].max() < valid["timestamp"].min()
    assert valid["timestamp"].max() < test["timestamp"].min()

def test_temporal_global_split_no_random_shuffle():
    df = pd.DataFrame({"timestamp": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "val": range(10)})
    train, valid, test = temporal_global_split(df, "timestamp", 0.8, 0.1, 0.1)
    # Output must be sorted by timestamp ascending
    assert train["timestamp"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert valid["timestamp"].tolist() == [9]
    assert test["timestamp"].tolist() == [10]

def test_temporal_global_split_invalid_ratios():
    df = pd.DataFrame({"timestamp": [1, 2, 3]})
    with pytest.raises(ValueError):
        temporal_global_split(df, "timestamp", 0.8, 0.1, 0.2)

def test_temporal_global_split_too_small_dataset():
    df = pd.DataFrame({"timestamp": [1, 2]})
    with pytest.raises(ValueError):
        temporal_global_split(df, "timestamp", 0.8, 0.1, 0.1)

def test_save_interaction_splits_creates_parquet_files(tmp_path):
    train = pd.DataFrame({"a": [1]})
    valid = pd.DataFrame({"a": [2]})
    test = pd.DataFrame({"a": [3]})
    full = pd.DataFrame({"a": [1, 2, 3]})
    
    cfg = {
        "train_path": str(tmp_path / "train.parquet"),
        "valid_path": str(tmp_path / "valid.parquet"),
        "test_path": str(tmp_path / "test.parquet"),
        "full_path": str(tmp_path / "full.parquet"),
    }
    
    save_interaction_splits(train, valid, test, full, cfg)
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "valid.parquet").exists()
    assert (tmp_path / "test.parquet").exists()
    assert (tmp_path / "full.parquet").exists()

def test_prepare_interactions_end_to_end_with_tmp_files(tmp_path):
    data_cfg_path = tmp_path / "data.yaml"
    int_cfg_path = tmp_path / "interactions.yaml"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    # Create mock ratings.csv
    ratings_path = raw_dir / "ratings.csv"
    pd.DataFrame({
        "userId": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "movieId": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "rating": [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 3.0, 5.0, 4.0, 1.0],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }).to_csv(ratings_path, index=False)
    
    with open(data_cfg_path, "w") as f:
        yaml.dump({
            "dataset": {"raw_data_dir": str(raw_dir)},
            "tables": {"ratings": {"filename": "ratings.csv"}}
        }, f)
        
    with open(int_cfg_path, "w") as f:
        yaml.dump({
            "dataset": {"name": "test"},
            "input": {"ratings_table": "ratings"},
            "columns": {"user_id": "userId", "item_id": "movieId", "rating": "rating", "timestamp": "timestamp"},
            "implicit_feedback": {"positive_threshold": 4.0, "label_column": "label", "positive_label": 1, "negative_label": 0},
            "split": {"strategy": "temporal_global", "train_size": 0.8, "valid_size": 0.1, "test_size": 0.1},
            "output": {
                "train_path": str(proc_dir / "train.parquet"),
                "valid_path": str(proc_dir / "valid.parquet"),
                "test_path": str(proc_dir / "test.parquet"),
                "full_path": str(proc_dir / "full.parquet"),
            }
        }, f)
        
    splits = prepare_interactions(data_config_path=data_cfg_path, interactions_config_path=int_cfg_path)
    assert len(splits["train"]) == 8
    assert len(splits["valid"]) == 1
    assert len(splits["test"]) == 1
    assert "label" in splits["full"].columns
    assert (proc_dir / "train.parquet").exists()

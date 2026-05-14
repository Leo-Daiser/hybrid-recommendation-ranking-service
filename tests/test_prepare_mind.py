import pandas as pd
import pytest
import yaml
from pathlib import Path
from src.data.prepare_mind import (
    parse_impression_tokens,
    convert_behaviors_to_interactions,
    convert_behaviors_to_impressions,
    prepare_mind_items,
    run_prepare_mind
)

def test_parse_impression_tokens_success():
    res = parse_impression_tokens("N1-1 N2-0")
    assert len(res) == 2
    assert res[0] == {"item_id": "N1", "label": 1}
    assert res[1] == {"item_id": "N2", "label": 0}

def test_parse_impression_tokens_empty_string():
    assert parse_impression_tokens("") == []
    assert parse_impression_tokens("   ") == []
    assert parse_impression_tokens(None) == []

def test_parse_impression_tokens_skips_malformed_token():
    res = parse_impression_tokens("N1-1 N2 N3-X N4-0")
    assert len(res) == 2
    assert res[0] == {"item_id": "N1", "label": 1}
    assert res[1] == {"item_id": "N4", "label": 0}

def test_convert_behaviors_to_interactions_success():
    df = pd.DataFrame({
        "impression_id": [1],
        "user_id": ["U1"],
        "time": ["11/11/2019 9:05:58 AM"],
        "history": ["N1"],
        "impressions": ["N2-1 N3-0"]
    })
    
    res = convert_behaviors_to_interactions(df)
    assert len(res) == 2
    assert list(res["item_id"]) == ["N2", "N3"]
    assert list(res["label"]) == [1, 0]
    assert list(res["user_id"]) == ["U1", "U1"]
    assert list(res["source"]) == ["mind", "mind"]
    assert pd.api.types.is_datetime64_any_dtype(res["timestamp"])

def test_convert_behaviors_to_interactions_labels_are_int():
    df = pd.DataFrame({
        "impression_id": [1],
        "user_id": ["U1"],
        "time": ["11/11/2019"],
        "history": [""],
        "impressions": ["N1-1"]
    })
    res = convert_behaviors_to_interactions(df)
    assert isinstance(res.iloc[0]["label"], (int, float, pd.Int64Dtype))
    # Standard python int or numpy int64
    assert int(res.iloc[0]["label"]) == 1

def test_convert_behaviors_to_impressions_success():
    df = pd.DataFrame({
        "impression_id": [1],
        "user_id": ["U1"],
        "time": ["11/11/2019"],
        "history": ["N1"],
        "impressions": ["N2-1 N3-0"]
    })
    
    res = convert_behaviors_to_impressions(df)
    assert len(res) == 1
    assert res.iloc[0]["positive_items"] == ["N2"]
    assert res.iloc[0]["negative_items"] == ["N3"]
    assert res.iloc[0]["positive_count"] == 1
    assert res.iloc[0]["negative_count"] == 1
    assert res.iloc[0]["total_impression_items"] == 2

def test_prepare_mind_items_deduplicates_items():
    train = pd.DataFrame({
        "item_id": ["N1", "N2"],
        "title": ["T1", "T2"]
    })
    valid = pd.DataFrame({
        "item_id": ["N2", "N3"],
        "title": ["T2_new", "T3"]
    })
    
    res = prepare_mind_items(train, valid)
    assert len(res) == 3
    assert set(res["item_id"]) == {"N1", "N2", "N3"}
    assert res[res["item_id"] == "N2"]["title"].iloc[0] == "T2" # First kept
    assert all(res["source"] == "mind")

def test_prepare_mind_items_handles_missing_valid_news():
    train = pd.DataFrame({"item_id": ["N1"], "title": ["T1"]})
    res = prepare_mind_items(train, None)
    assert len(res) == 1
    assert res.iloc[0]["item_id"] == "N1"

@pytest.fixture
def mock_mind_config(tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "proc"
    raw_dir.mkdir()
    proc_dir.mkdir()
    
    (raw_dir / "train_b.tsv").write_text("1\tU1\t11/11/2019\t\tN1-1 N2-0\n")
    (raw_dir / "train_n.tsv").write_text("N1\tcat\tsub\tT1\tA1\tU1\t\t\n")
    (raw_dir / "valid_b.tsv").write_text("2\tU2\t11/12/2019\t\tN2-1\n")
    (raw_dir / "valid_n.tsv").write_text("N2\tcat\tsub\tT2\tA2\tU2\t\t\n")
    
    config = {
        "mind": {
            "dataset_name": "MIND",
            "version": "MINDsmall",
            "raw_data_dir": str(raw_dir),
            "processed_data_dir": str(proc_dir),
            "expected_files": {
                "train_behaviors": "train_b.tsv",
                "train_news": "train_n.tsv",
                "valid_behaviors": "valid_b.tsv",
                "valid_news": "valid_n.tsv"
            },
            "columns": {"behaviors": {}, "news": {}},
            "output": {
                "train_interactions_path": str(proc_dir / "train_int.parquet"),
                "valid_interactions_path": str(proc_dir / "valid_int.parquet"),
                "items_path": str(proc_dir / "items.parquet"),
                "impressions_path": str(proc_dir / "impress.parquet"),
            },
            "parsing": {
                "positive_label": 1,
                "negative_label": 0,
                "source": "mind"
            },
            "validation": {"strict_foreign_keys": False}
        }
    }
    
    cfg_path = tmp_path / "mind.yaml"
    with open(cfg_path, "w") as f:
        import yaml
        yaml.dump(config, f)
        
    return cfg_path, proc_dir

def test_run_prepare_mind_creates_outputs(mock_mind_config):
    cfg_path, proc_dir = mock_mind_config
    
    res = run_prepare_mind(cfg_path)
    
    assert len(res["train_interactions"]) == 2
    assert len(res["valid_interactions"]) == 1
    assert len(res["items"]) == 2
    assert len(res["impressions"]) == 2
    
    assert (proc_dir / "train_int.parquet").exists()
    assert (proc_dir / "valid_int.parquet").exists()
    assert (proc_dir / "items.parquet").exists()
    assert (proc_dir / "impress.parquet").exists()

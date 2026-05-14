import pytest
import pandas as pd
import yaml
from pathlib import Path
from src.evaluation.run_retrieval_evaluation import run_retrieval_evaluation

def test_run_retrieval_evaluation_creates_outputs(tmp_path):
    cfg_path = tmp_path / "evaluation.yaml"
    proc_dir = tmp_path / "proc"
    out_dir = tmp_path / "out"
    proc_dir.mkdir()
    
    valid_path = proc_dir / "valid.parquet"
    item_feat_path = proc_dir / "item_feat.parquet"
    cache1_path = proc_dir / "cache1.parquet"
    cache2_path = proc_dir / "cache2.parquet"
    
    json_out = out_dir / "valid.json"
    csv_out = out_dir / "valid.csv"
    md_out = out_dir / "valid.md"
    
    pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]}).to_parquet(valid_path)
    pd.DataFrame({"item_id": [10, 20]}).to_parquet(item_feat_path)
    pd.DataFrame({"user_id": [1], "item_id": [10], "rank": [1], "retrieval_score": [1.0]}).to_parquet(cache1_path)
    pd.DataFrame({"user_id": [1], "item_id": [20], "rank": [1], "retrieval_score": [1.0]}).to_parquet(cache2_path)
    
    with open(cfg_path, "w") as f:
        yaml.dump({
            "evaluation": {
                "user_id_column": "user_id",
                "item_id_column": "item_id",
                "label_column": "label",
                "rank_column": "rank",
                "score_column": "retrieval_score",
                "positive_label": 1,
                "interactions": {
                    "train_path": "fake",
                    "valid_path": str(valid_path),
                    "test_path": "fake"
                },
                "item_features_path": str(item_feat_path),
                "candidate_caches": {
                    "pop": str(cache1_path),
                    "knn": str(cache2_path)
                },
                "k_values": [1],
                "output": {
                    "valid_json_path": str(json_out),
                    "valid_csv_path": str(csv_out),
                    "valid_report_path": str(md_out)
                },
                "behavior": {
                    "exclude_users_without_ground_truth": True
                }
            }
        }, f)
        
    res = run_retrieval_evaluation(cfg_path)
    assert len(res) == 2
    assert json_out.exists()
    assert csv_out.exists()
    assert md_out.exists()

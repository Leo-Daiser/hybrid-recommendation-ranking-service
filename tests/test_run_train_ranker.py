import pytest
import pandas as pd
import yaml
from pathlib import Path
from src.ranking.run_train_ranker import run_train_ranker

def test_run_train_ranker_creates_outputs(tmp_path):
    cfg_path = tmp_path / "ranker.yaml"
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    
    train_path = proc_dir / "train.parquet"
    valid_path = proc_dir / "valid.parquet"
    
    df = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 10, 30],
        "target": [1, 0, 0, 1],
        "feat1": [0.1, 0.2, 0.3, 0.4]
    })
    df.to_parquet(train_path)
    df.to_parquet(valid_path)
    
    logreg_out = tmp_path / "logreg.pkl"
    challenger_out = tmp_path / "challenger.pkl"
    json_out = tmp_path / "metrics.json"
    csv_out = tmp_path / "metrics.csv"
    report_out = tmp_path / "report.md"
    
    with open(cfg_path, "w") as f:
        yaml.dump({
            "ranker": {
                "input": {
                    "train_path": str(train_path),
                    "valid_path": str(valid_path),
                },
                "output": {
                    "logreg_model_path": str(logreg_out),
                    "challenger_model_path": str(challenger_out),
                    "metrics_json_path": str(json_out),
                    "metrics_csv_path": str(csv_out),
                    "report_path": str(report_out)
                },
                "columns": {
                    "user_id_column": "user_id",
                    "item_id_column": "item_id",
                    "target_column": "target"
                },
                "training": {
                    "random_seed": 42,
                    "baseline_model": "logreg",
                    "challenger_model": "challenger"
                },
                "features": {
                    "exclude_columns": ["user_id", "item_id", "target"]
                },
                "evaluation": {
                    "k_values": [1, 2]
                }
            }
        }, f)
        
    res = run_train_ranker(cfg_path)
    assert not res.empty
    assert logreg_out.exists()
    assert challenger_out.exists()
    assert json_out.exists()
    assert csv_out.exists()
    assert report_out.exists()

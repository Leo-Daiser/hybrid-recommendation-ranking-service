import pytest
from pathlib import Path
import pandas as pd
import yaml
from src.data.load_raw import load_data_config, resolve_table_paths, load_raw_tables

def test_resolve_table_paths_success():
    config = {
        "dataset": {"raw_data_dir": "data"},
        "tables": {"t1": {"filename": "t1.csv"}}
    }
    paths = resolve_table_paths(config)
    assert paths == {"t1": Path("data/t1.csv")}

def test_load_raw_tables_success(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    f1 = raw_dir / "t1.csv"
    f1.write_text("a,b\n1,2")
    
    config_path = tmp_path / "config.yaml"
    config_data = {
        "dataset": {"raw_data_dir": str(raw_dir)},
        "tables": {"t1": {"filename": "t1.csv"}}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    tables = load_raw_tables(config_path)
    assert "t1" in tables
    assert tables["t1"].shape == (1, 2)

def test_load_raw_tables_missing_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_data = {
        "dataset": {"raw_data_dir": str(tmp_path)},
        "tables": {"t1": {"filename": "missing.csv"}}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    with pytest.raises(FileNotFoundError):
        load_raw_tables(config_path)

def test_load_raw_tables_unknown_table(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    f1 = raw_dir / "t1.csv"
    f1.write_text("a,b\n1,2")
    
    config_path = tmp_path / "config.yaml"
    config_data = {
        "dataset": {"raw_data_dir": str(raw_dir)},
        "tables": {"t1": {"filename": "t1.csv"}}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    with pytest.raises(ValueError):
        load_raw_tables(config_path, table_names=["unknown"])

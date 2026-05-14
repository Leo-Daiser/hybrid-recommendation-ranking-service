import pytest
from pathlib import Path
from src.data.download import load_data_config, expected_raw_files, download_movielens
import yaml

def test_load_data_config_success(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_data = {"dataset": {"name": "test"}}
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    loaded = load_data_config(config_path)
    assert loaded == config_data

def test_load_data_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data_config("nonexistent.yaml")

def test_expected_raw_files():
    config = {
        "dataset": {"raw_data_dir": "data/raw"},
        "tables": {
            "t1": {"filename": "f1.csv"}
        }
    }
    expected = expected_raw_files(config)
    assert expected == [Path("data/raw/f1.csv")]

def test_download_skips_if_files_exist(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    f1 = raw_dir / "f1.csv"
    f1.write_text("a,b\n1,2")
    
    config_path = tmp_path / "config.yaml"
    config_data = {
        "dataset": {
            "source_url": "http://fake",
            "raw_data_dir": str(raw_dir)
        },
        "tables": {"t1": {"filename": "f1.csv"}}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    def mock_retrieve(*args, **kwargs):
        raise RuntimeError("Should not be called")
    monkeypatch.setattr("urllib.request.urlretrieve", mock_retrieve)
    
    result_dir = download_movielens(config_path, force=False)
    assert result_dir == raw_dir

def test_download_extracts_zip_from_local_mock_or_tmp_fixture(tmp_path, monkeypatch):
    import zipfile
    raw_dir = tmp_path / "raw"
    
    config_path = tmp_path / "config.yaml"
    config_data = {
        "dataset": {
            "source_url": "http://fake.zip",
            "raw_data_dir": str(raw_dir)
        },
        "tables": {"t1": {"filename": "f1.csv"}}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    def mock_retrieve(url, filename):
        with zipfile.ZipFile(filename, 'w') as zf:
            zf.writestr(f"{raw_dir.name}/f1.csv", "a,b\n1,2")
            
    monkeypatch.setattr("urllib.request.urlretrieve", mock_retrieve)
    
    result_dir = download_movielens(config_path, force=True)
    assert result_dir == raw_dir
    assert (raw_dir / "f1.csv").exists()

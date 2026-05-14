import os
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
import yaml

def load_data_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def expected_raw_files(config: dict) -> list[Path]:
    raw_data_dir = Path(config["dataset"]["raw_data_dir"])
    tables = config.get("tables", {})
    return [raw_data_dir / table_info["filename"] for table_info in tables.values()]

def download_movielens(config_path: str | Path = "configs/data.yaml", force: bool = False) -> Path:
    config = load_data_config(config_path)
    dataset_cfg = config["dataset"]
    url = dataset_cfg["source_url"]
    raw_data_dir = Path(dataset_cfg["raw_data_dir"])
    
    expected_files = expected_raw_files(config)
    
    if not force:
        if all(f.exists() for f in expected_files):
            print("All expected files already exist. Skipping download.")
            return raw_data_dir
    
    raw_data_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = raw_data_dir.parent / "temp.zip"
    
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    
    print("Extracting files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir.parent)
    except zipfile.BadZipFile:
        raise RuntimeError("Downloaded file is not a valid zip archive.")
    finally:
        if zip_path.exists():
            zip_path.unlink()
            
    print(f"Dataset downloaded and extracted to {raw_data_dir}")
    return raw_data_dir

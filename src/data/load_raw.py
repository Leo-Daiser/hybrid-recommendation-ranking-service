from pathlib import Path
import yaml
import pandas as pd

def load_data_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_table_paths(config: dict) -> dict[str, Path]:
    if "dataset" not in config or "raw_data_dir" not in config["dataset"]:
        raise ValueError("Missing 'raw_data_dir' in config['dataset']")
    if "tables" not in config:
        raise ValueError("Missing 'tables' in config")
        
    raw_data_dir = Path(config["dataset"]["raw_data_dir"])
    return {
        table_name: raw_data_dir / table_info["filename"]
        for table_name, table_info in config["tables"].items()
    }

def load_raw_tables(config_path: str | Path, table_names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    config = load_data_config(config_path)
    table_paths = resolve_table_paths(config)
    
    if table_names is None:
        table_names = list(table_paths.keys())
        
    tables = {}
    for t_name in table_names:
        if t_name not in table_paths:
            raise ValueError(f"Unknown table: {t_name}")
            
        path = table_paths[t_name]
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        tables[t_name] = pd.read_csv(path)
        
    return tables

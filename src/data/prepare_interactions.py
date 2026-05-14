from pathlib import Path
import yaml
import pandas as pd
import math
from src.data.load_raw import load_raw_tables

def load_interactions_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Interactions config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_implicit_labels(
    ratings: pd.DataFrame,
    rating_column: str,
    label_column: str,
    positive_threshold: float,
) -> pd.DataFrame:
    if rating_column not in ratings.columns:
        raise ValueError(f"Rating column '{rating_column}' not found in dataframe.")
    if label_column in ratings.columns:
        raise ValueError(f"Label column '{label_column}' already exists in dataframe.")
    
    df = ratings.copy()
    df[label_column] = (df[rating_column] >= positive_threshold).astype(int)
    return df

def temporal_global_split(
    interactions: pd.DataFrame,
    timestamp_column: str,
    train_size: float,
    valid_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    if not math.isclose(train_size + valid_size + test_size, 1.0, rel_tol=1e-5):
        raise ValueError(f"Split sizes must sum to 1.0, got: {train_size + valid_size + test_size}")
        
    if len(interactions) < 3:
        raise ValueError("Dataset too small to split into 3 non-empty sets.")

    df = interactions.sort_values(by=timestamp_column, ascending=True).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_size)
    valid_end = train_end + int(n * valid_size)
    
    if train_end == 0 or valid_end == train_end or valid_end == n:
        raise ValueError("Dataset is too small to yield 3 non-empty splits with given ratios.")

    train = df.iloc[:train_end].reset_index(drop=True)
    valid = df.iloc[train_end:valid_end].reset_index(drop=True)
    test = df.iloc[valid_end:].reset_index(drop=True)
    
    return train, valid, test

def save_interaction_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    full: pd.DataFrame,
    output_config: dict,
) -> None:
    for path_key, df in [
        ("train_path", train),
        ("valid_path", valid),
        ("test_path", test),
        ("full_path", full)
    ]:
        path = Path(output_config[path_key])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

def prepare_interactions(
    data_config_path: str | Path = "configs/data.yaml",
    interactions_config_path: str | Path = "configs/interactions.yaml",
) -> dict[str, pd.DataFrame]:
    
    int_config = load_interactions_config(interactions_config_path)
    ratings_table_name = int_config["input"]["ratings_table"]
    
    tables = load_raw_tables(data_config_path, table_names=[ratings_table_name])
    ratings = tables[ratings_table_name]
    
    col_cfg = int_config["columns"]
    fb_cfg = int_config["implicit_feedback"]
    split_cfg = int_config["split"]
    
    full = create_implicit_labels(
        ratings,
        rating_column=col_cfg["rating"],
        label_column=fb_cfg["label_column"],
        positive_threshold=fb_cfg["positive_threshold"],
    )
    
    if col_cfg["user_id"] != "user_id":
        full["user_id"] = full[col_cfg["user_id"]]
    if col_cfg["item_id"] != "item_id":
        full["item_id"] = full[col_cfg["item_id"]]
        
    train, valid, test = temporal_global_split(
        interactions=full,
        timestamp_column=col_cfg["timestamp"],
        train_size=split_cfg["train_size"],
        valid_size=split_cfg["valid_size"],
        test_size=split_cfg["test_size"],
    )
    
    save_interaction_splits(train, valid, test, full, int_config["output"])
    
    return {
        "full": full,
        "train": train,
        "valid": valid,
        "test": test
    }

def run_prepare_interactions(
    data_config_path: str | Path = "configs/data.yaml",
    interactions_config_path: str | Path = "configs/interactions.yaml",
) -> dict[str, pd.DataFrame]:
    return prepare_interactions(data_config_path, interactions_config_path)

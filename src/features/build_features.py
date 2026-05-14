from pathlib import Path
import yaml
import pandas as pd
from src.data.load_raw import load_raw_tables
from src.features.user_features import build_user_features
from src.features.item_features import build_item_features, parse_genres

def load_feature_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feature config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_build_features(
    data_config_path: str | Path = "configs/data.yaml",
    feature_config_path: str | Path = "configs/features.yaml",
) -> dict[str, pd.DataFrame]:
    
    feat_cfg = load_feature_config(feature_config_path)
    col_cfg = feat_cfg["features"]
    
    train_path = Path(feat_cfg["input"]["train_interactions_path"])
    if not train_path.exists():
        raise FileNotFoundError(f"Train interactions not found: {train_path}")
        
    train_interactions = pd.read_parquet(train_path)
    
    movies_table_name = feat_cfg["input"]["movies_table"]
    raw_tables = load_raw_tables(data_config_path, table_names=[movies_table_name])
    movies = raw_tables[movies_table_name]
    
    user_features = build_user_features(
        train_interactions,
        user_col=col_cfg["user_id_column"],
        rating_col=col_cfg["rating_column"],
        label_col=col_cfg["label_column"],
        timestamp_col=col_cfg["timestamp_column"]
    )
    
    item_features = build_item_features(
        train_interactions,
        movies,
        item_col=col_cfg["item_id_column"],
        original_item_col=col_cfg["original_item_id_column"],
        rating_col=col_cfg["rating_column"],
        label_col=col_cfg["label_column"],
        timestamp_col=col_cfg["timestamp_column"],
        unknown_genre_token=feat_cfg["item_features"]["unknown_genre_token"]
    )
    
    genre_features = parse_genres(
        movies,
        item_col=col_cfg["original_item_id_column"],
        genres_col="genres"
    )
    
    # Save parquet
    out_cfg = feat_cfg["output"]
    u_path = Path(out_cfg["user_features_path"])
    i_path = Path(out_cfg["item_features_path"])
    g_path = Path(out_cfg["genre_features_path"])
    
    u_path.parent.mkdir(parents=True, exist_ok=True)
    
    user_features.to_parquet(u_path, index=False)
    item_features.to_parquet(i_path, index=False)
    genre_features.to_parquet(g_path, index=False)
    
    return {
        "user_features": user_features,
        "item_features": item_features,
        "genre_features": genre_features
    }

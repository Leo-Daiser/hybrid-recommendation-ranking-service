import yaml
import pandas as pd
from pathlib import Path
from src.retrieval.popularity import build_popularity_candidate_cache
from src.retrieval.candidate_cache import save_candidate_cache, validate_candidate_cache

def load_retrieval_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_build_popularity_candidates(
    retrieval_config_path: str | Path = "configs/retrieval.yaml",
) -> pd.DataFrame:
    
    cfg = load_retrieval_config(retrieval_config_path)
    ret_cfg = cfg["retrieval"]
    
    train_path = Path(ret_cfg["train_interactions_path"])
    item_feat_path = Path(ret_cfg["item_features_path"])
    
    train_interactions = pd.read_parquet(train_path)
    item_features = pd.read_parquet(item_feat_path)
    
    user_col = ret_cfg["user_id_column"]
    item_col = ret_cfg["item_id_column"]
    
    pop_cfg = ret_cfg["popularity"]
    
    cache = build_popularity_candidate_cache(
        train_interactions=train_interactions,
        item_features=item_features,
        user_col=user_col,
        item_col=item_col,
        score_col=pop_cfg["score_column"],
        tie_breaker_col=pop_cfg["tie_breaker_column"],
        top_k=pop_cfg["top_k"],
        candidate_pool_size=pop_cfg["candidate_pool_size"],
        model_name=pop_cfg["model_name"]
    )
    
    validate_candidate_cache(
        cache,
        user_col=user_col,
        item_col=item_col,
        rank_col="rank",
        score_col="retrieval_score"
    )
    
    out_path = ret_cfg["output_candidate_cache_path"]
    save_candidate_cache(cache, out_path)
    
    return cache

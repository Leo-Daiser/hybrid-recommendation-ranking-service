import yaml
import pandas as pd
from pathlib import Path
from src.retrieval.item_knn import build_item_similarity_topk, build_itemknn_candidate_cache
from src.retrieval.candidate_cache import save_candidate_cache, validate_candidate_cache

def load_retrieval_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_build_itemknn_candidates(
    retrieval_config_path: str | Path = "configs/retrieval.yaml",
) -> dict[str, pd.DataFrame]:
    
    cfg = load_retrieval_config(retrieval_config_path)
    ret_cfg = cfg["retrieval"]
    knn_cfg = cfg["item_knn"]
    
    train_path = Path(ret_cfg["train_interactions_path"])
    pop_cache_path = Path(ret_cfg["output_candidate_cache_path"])
    
    train_interactions = pd.read_parquet(train_path)
    popularity_fallback = pd.read_parquet(pop_cache_path)
    
    user_col = ret_cfg["user_id_column"]
    item_col = ret_cfg["item_id_column"]
    label_col = ret_cfg["label_column"]
    
    sim_df = build_item_similarity_topk(
        train_interactions=train_interactions,
        user_col=user_col,
        item_col=item_col,
        label_col=label_col,
        max_neighbors_per_item=knn_cfg["max_neighbors_per_item"],
        min_similarity=knn_cfg["min_similarity"]
    )
    
    sim_out_path = Path(knn_cfg["output_similarity_path"])
    sim_out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_parquet(sim_out_path, index=False)
    
    cache = build_itemknn_candidate_cache(
        train_interactions=train_interactions,
        item_similarity=sim_df,
        popularity_fallback=popularity_fallback,
        user_col=user_col,
        item_col=item_col,
        top_k=knn_cfg["top_k"],
        model_name=knn_cfg["model_name"],
        aggregation=knn_cfg["aggregation"]
    )
    
    validate_candidate_cache(
        cache,
        user_col=user_col,
        item_col=item_col,
        rank_col="rank",
        score_col="retrieval_score"
    )
    
    cache_out_path = Path(knn_cfg["output_candidate_cache_path"])
    save_candidate_cache(cache, cache_out_path)
    
    return {
        "item_similarity": sim_df,
        "candidate_cache": cache
    }

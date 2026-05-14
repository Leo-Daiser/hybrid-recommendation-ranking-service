import pandas as pd
from pathlib import Path
from src.ranking.dataset import (
    load_ranking_config,
    merge_candidate_caches,
    build_ranking_dataset
)

def run_build_ranking_datasets(
    ranking_config_path: str | Path = "configs/ranking.yaml",
) -> dict[str, pd.DataFrame]:
    cfg = load_ranking_config(ranking_config_path)["ranking"]
    
    valid_interactions = pd.read_parquet(cfg["input"]["interactions_valid_path"])
    test_interactions = pd.read_parquet(cfg["input"]["interactions_test_path"])
    
    user_features = pd.read_parquet(cfg["input"]["user_features_path"])
    item_features = pd.read_parquet(cfg["input"]["item_features_path"])
    
    caches = {}
    for model_name in cfg["candidates"]["use_models"]:
        cache_path = cfg["input"]["candidate_caches"][model_name]
        caches[model_name] = pd.read_parquet(cache_path)
        
    merged_candidates = merge_candidate_caches(
        caches,
        user_col=cfg["user_id_column"],
        item_col=cfg["item_id_column"]
    )
    
    ranking_train = build_ranking_dataset(
        candidates=merged_candidates,
        future_interactions=valid_interactions,
        user_features=user_features,
        item_features=item_features,
        user_col=cfg["user_id_column"],
        item_col=cfg["item_id_column"],
        interaction_label_col=cfg["label_column"],
        target_col=cfg["labels"]["target_column"],
        positive_label=cfg["labels"]["positive_label"],
        negative_to_positive_ratio=cfg["negative_sampling"]["negative_to_positive_ratio"],
        random_seed=cfg["negative_sampling"]["random_seed"],
        apply_negative_sampling=cfg["negative_sampling"]["enabled"],
    )
    
    ranking_valid = build_ranking_dataset(
        candidates=merged_candidates,
        future_interactions=test_interactions,
        user_features=user_features,
        item_features=item_features,
        user_col=cfg["user_id_column"],
        item_col=cfg["item_id_column"],
        interaction_label_col=cfg["label_column"],
        target_col=cfg["labels"]["target_column"],
        positive_label=cfg["labels"]["positive_label"],
        negative_to_positive_ratio=cfg["negative_sampling"]["negative_to_positive_ratio"],
        random_seed=cfg["negative_sampling"]["random_seed"],
        apply_negative_sampling=cfg["negative_sampling"]["enabled"],
    )
    
    train_out = Path(cfg["output"]["ranking_train_path"])
    valid_out = Path(cfg["output"]["ranking_valid_path"])
    train_out.parent.mkdir(parents=True, exist_ok=True)
    valid_out.parent.mkdir(parents=True, exist_ok=True)
    
    ranking_train.to_parquet(train_out, index=False)
    ranking_valid.to_parquet(valid_out, index=False)
    
    return {
        "ranking_train": ranking_train,
        "ranking_valid": ranking_valid
    }

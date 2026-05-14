import yaml
import pandas as pd
import numpy as np
from pathlib import Path

def load_ranking_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def merge_candidate_caches(
    candidate_caches: dict[str, pd.DataFrame],
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    dfs = []
    for model_name, cache in candidate_caches.items():
        if cache.empty:
            continue
        df = cache.copy()
        df["retrieval_model_name"] = model_name
        dfs.append(df)
        
    if not dfs:
        return pd.DataFrame(columns=[
            user_col, item_col, "max_retrieval_score", "min_rank",
            "retrieval_model_count", "came_from_popularity", "came_from_itemknn"
        ])
        
    combined = pd.concat(dfs, ignore_index=True)
    
    grouped = combined.groupby([user_col, item_col]).agg({
        "retrieval_score": "max",
        "rank": "min",
    }).reset_index()
    
    models_info = combined.groupby([user_col, item_col])["retrieval_model_name"].apply(list).reset_index()
    
    merged = pd.merge(grouped, models_info, on=[user_col, item_col])
    
    merged["retrieval_model_count"] = merged["retrieval_model_name"].apply(lambda x: len(set(x)))
    merged["came_from_popularity"] = merged["retrieval_model_name"].apply(lambda x: 1 if "popularity_v1" in x else 0)
    merged["came_from_itemknn"] = merged["retrieval_model_name"].apply(lambda x: 1 if "itemknn_cosine_v1" in x else 0)
    
    merged = merged.rename(columns={
        "retrieval_score": "max_retrieval_score",
        "rank": "min_rank"
    }).drop(columns=["retrieval_model_name"])
    
    return merged

def build_future_positive_pairs(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    label_col: str = "label",
    positive_label: int = 1,
) -> set[tuple[int, int]]:
    if interactions.empty:
        return set()
    pos = interactions[interactions[label_col] == positive_label]
    return set(zip(pos[user_col], pos[item_col]))

def label_candidate_pairs(
    candidates: pd.DataFrame,
    future_positive_pairs: set[tuple[int, int]],
    user_col: str = "user_id",
    item_col: str = "item_id",
    target_col: str = "target",
) -> pd.DataFrame:
    df = candidates.copy()
    if df.empty:
        df[target_col] = pd.Series(dtype=int)
        return df
    
    pairs = pd.Series(zip(df[user_col], df[item_col]))
    df[target_col] = pairs.isin(future_positive_pairs).astype(int)
    return df

def add_user_item_features(
    labeled_candidates: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    if labeled_candidates.empty:
        return labeled_candidates
        
    df = labeled_candidates.copy()
    
    if not user_features.empty:
        df = pd.merge(df, user_features, on=user_col, how="left")
        
    if not item_features.empty:
        df = pd.merge(df, item_features, on=item_col, how="left")
        
    return df

def sample_negatives(
    ranking_data: pd.DataFrame,
    target_col: str = "target",
    negative_to_positive_ratio: int = 5,
    random_seed: int = 42,
    keep_all_positives: bool = True,
) -> pd.DataFrame:
    if ranking_data.empty:
        return ranking_data
        
    positives = ranking_data[ranking_data[target_col] == 1]
    negatives = ranking_data[ranking_data[target_col] == 0]
    
    n_pos = len(positives)
    n_neg = len(negatives)
    
    if n_pos == 0:
        if n_neg > 0:
            sample_size = min(100, n_neg)
            neg_sampled = negatives.sample(n=sample_size, random_state=random_seed)
            return neg_sampled
        return ranking_data

    target_neg_count = n_pos * negative_to_positive_ratio
    
    if n_neg > target_neg_count:
        neg_sampled = negatives.sample(n=target_neg_count, random_state=random_seed)
    else:
        neg_sampled = negatives
        
    res_dfs = []
    if keep_all_positives:
        res_dfs.append(positives)
    
    res_dfs.append(neg_sampled)
    
    return pd.concat(res_dfs, ignore_index=True)

def build_ranking_dataset(
    candidates: pd.DataFrame,
    future_interactions: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    interaction_label_col: str = "label",
    target_col: str = "target",
    positive_label: int = 1,
    negative_to_positive_ratio: int = 5,
    random_seed: int = 42,
    apply_negative_sampling: bool = True,
) -> pd.DataFrame:
    
    future_positives = build_future_positive_pairs(
        future_interactions, user_col, item_col, interaction_label_col, positive_label
    )
    
    labeled = label_candidate_pairs(
        candidates, future_positives, user_col, item_col, target_col
    )
    
    with_features = add_user_item_features(
        labeled, user_features, item_features, user_col, item_col
    )
    
    if apply_negative_sampling:
        with_features = sample_negatives(
            with_features, target_col, negative_to_positive_ratio, random_seed
        )
        
    return with_features

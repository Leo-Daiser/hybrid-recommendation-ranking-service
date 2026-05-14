import pandas as pd
from datetime import datetime, timezone

def build_popularity_ranking(
    item_features: pd.DataFrame,
    item_col: str = "item_id",
    score_col: str = "item_positive_ratio",
    tie_breaker_col: str = "item_rating_count",
    candidate_pool_size: int = 500,
) -> pd.DataFrame:
    if score_col not in item_features.columns:
        raise ValueError(f"Score column {score_col} missing in item_features.")
    if tie_breaker_col not in item_features.columns:
        raise ValueError(f"Tie breaker column {tie_breaker_col} missing in item_features.")
        
    df = item_features[[item_col, score_col, tie_breaker_col]].copy()
    
    df = df.sort_values(
        by=[score_col, tie_breaker_col],
        ascending=[False, False]
    ).head(candidate_pool_size).reset_index(drop=True)
    
    df["popularity_rank"] = df.index + 1
    df = df.rename(columns={score_col: "retrieval_score"})
    df["retrieval_model"] = "popularity_v1"
    
    return df[[item_col, "retrieval_score", "popularity_rank", "retrieval_model"]]

def get_seen_items_by_user(
    train_interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> dict[int, set[int]]:
    seen = train_interactions.groupby(user_col)[item_col].apply(set).to_dict()
    return seen

def recommend_popular_for_user(
    user_id: int,
    popularity_ranking: pd.DataFrame,
    seen_items_by_user: dict[int, set[int]],
    item_col: str = "item_id",
    score_col: str = "retrieval_score",
    top_k: int = 50,
    model_name: str = "popularity_v1",
) -> pd.DataFrame:
    seen = seen_items_by_user.get(user_id, set())
    
    cands = popularity_ranking[~popularity_ranking[item_col].isin(seen)]
    cands = cands.head(top_k).copy()
    
    cands["user_id"] = user_id
    cands["rank"] = range(1, len(cands) + 1)
    cands["retrieval_model"] = model_name
    
    return cands[["user_id", item_col, score_col, "rank", "retrieval_model"]]

def build_popularity_candidate_cache(
    train_interactions: pd.DataFrame,
    item_features: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    score_col: str = "item_positive_ratio",
    tie_breaker_col: str = "item_rating_count",
    top_k: int = 50,
    candidate_pool_size: int = 500,
    model_name: str = "popularity_v1",
) -> pd.DataFrame:
    
    pop_ranking = build_popularity_ranking(
        item_features, item_col, score_col, tie_breaker_col, candidate_pool_size
    )
    pop_ranking["retrieval_model"] = model_name
    
    seen = get_seen_items_by_user(train_interactions, user_col, item_col)
    all_users = train_interactions[user_col].unique()
    
    cache_dfs = []
    for uid in all_users:
        user_recs = recommend_popular_for_user(
            user_id=uid,
            popularity_ranking=pop_ranking,
            seen_items_by_user=seen,
            item_col=item_col,
            score_col="retrieval_score",
            top_k=top_k,
            model_name=model_name
        )
        cache_dfs.append(user_recs)
        
    if not cache_dfs:
        cache = pd.DataFrame(columns=["user_id", item_col, "retrieval_score", "rank", "retrieval_model"])
    else:
        cache = pd.concat(cache_dfs, ignore_index=True)
        
    cache["generated_at"] = datetime.now(timezone.utc).isoformat()
    
    return cache

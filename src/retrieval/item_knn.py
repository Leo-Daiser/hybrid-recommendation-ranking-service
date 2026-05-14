import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone

def build_user_item_matrix(
    train_interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    label_col: str = "label",
) -> tuple:
    df = train_interactions[train_interactions[label_col] == 1].copy()
    
    unique_users = df[user_col].unique()
    unique_items = df[item_col].unique()
    
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {it: i for i, it in enumerate(unique_items)}
    
    row = df[user_col].map(user_map).values
    col = df[item_col].map(item_map).values
    data = np.ones(len(df), dtype=np.float32)
    
    matrix = csr_matrix((data, (row, col)), shape=(len(unique_users), len(unique_items)))
    
    return matrix, user_map, item_map

def build_item_similarity_topk(
    train_interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    label_col: str = "label",
    max_neighbors_per_item: int = 100,
    min_similarity: float = 0.0,
) -> pd.DataFrame:
    
    matrix, user_map, item_map = build_user_item_matrix(
        train_interactions, user_col, item_col, label_col
    )
    
    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=["item_id", "similar_item_id", "similarity", "rank", "model_name"])
    
    item_matrix = matrix.T
    sim_matrix = cosine_similarity(item_matrix, dense_output=False)
    
    inverse_item_map = {i: it for it, i in item_map.items()}
    
    records = []
    
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix.getrow(i)
        indices = row.indices
        data = row.data
        
        mask = (indices != i) & (data > min_similarity)
        indices = indices[mask]
        data = data[mask]
        
        if len(indices) == 0:
            continue
            
        sort_idx = np.argsort(data)[::-1][:max_neighbors_per_item]
        top_indices = indices[sort_idx]
        top_data = data[sort_idx]
        
        item_id = inverse_item_map[i]
        
        for rank_idx, (sim_item_idx, sim_score) in enumerate(zip(top_indices, top_data)):
            records.append({
                "item_id": item_id,
                "similar_item_id": inverse_item_map[sim_item_idx],
                "similarity": float(sim_score),
                "rank": rank_idx + 1,
                "model_name": "itemknn_cosine_v1"
            })
            
    if not records:
        return pd.DataFrame(columns=["item_id", "similar_item_id", "similarity", "rank", "model_name"])
        
    return pd.DataFrame(records)

def recommend_itemknn_for_user(
    user_id: int,
    train_interactions: pd.DataFrame,
    item_similarity: pd.DataFrame,
    popularity_fallback: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    top_k: int = 50,
    model_name: str = "itemknn_cosine_v1",
    aggregation: str = "sum",
) -> pd.DataFrame:
    
    user_history = train_interactions[train_interactions[user_col] == user_id]
    
    def get_fallback():
        fallback = popularity_fallback[popularity_fallback[user_col] == user_id].copy()
        if fallback.empty:
            fallback = popularity_fallback.head(top_k).copy()
            fallback["user_id"] = user_id
        fallback["retrieval_model"] = f"fallback_popularity"
        return fallback.head(top_k)[[user_col, item_col, "retrieval_score", "rank", "retrieval_model"]]

    if user_history.empty or not (user_history["label"] == 1).any():
        return get_fallback()
        
    user_positive_history = user_history[user_history["label"] == 1][item_col].unique()
    
    sim_items = item_similarity[item_similarity["item_id"].isin(user_positive_history)]
    
    if sim_items.empty:
        return get_fallback()
        
    if aggregation == "sum":
        cands = sim_items.groupby("similar_item_id")["similarity"].sum().reset_index()
    elif aggregation == "max":
        cands = sim_items.groupby("similar_item_id")["similarity"].max().reset_index()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
        
    cands = cands.rename(columns={"similar_item_id": item_col, "similarity": "retrieval_score"})
    
    seen_items = user_history[item_col].unique()
    cands = cands[~cands[item_col].isin(seen_items)]
    
    cands = cands.sort_values("retrieval_score", ascending=False).head(top_k).reset_index(drop=True)
    
    if cands.empty:
        return get_fallback()
    
    cands["user_id"] = user_id
    cands["rank"] = cands.index + 1
    cands["retrieval_model"] = model_name
    
    return cands[["user_id", item_col, "retrieval_score", "rank", "retrieval_model"]]

def build_itemknn_candidate_cache(
    train_interactions: pd.DataFrame,
    item_similarity: pd.DataFrame,
    popularity_fallback: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    top_k: int = 50,
    model_name: str = "itemknn_cosine_v1",
    aggregation: str = "sum",
) -> pd.DataFrame:
    
    all_users = train_interactions[user_col].unique()
    cache_dfs = []
    
    for uid in all_users:
        user_recs = recommend_itemknn_for_user(
            user_id=uid,
            train_interactions=train_interactions,
            item_similarity=item_similarity,
            popularity_fallback=popularity_fallback,
            user_col=user_col,
            item_col=item_col,
            top_k=top_k,
            model_name=model_name,
            aggregation=aggregation
        )
        cache_dfs.append(user_recs)
        
    if not cache_dfs:
        cache = pd.DataFrame(columns=["user_id", item_col, "retrieval_score", "rank", "retrieval_model"])
    else:
        cache = pd.concat(cache_dfs, ignore_index=True)
        
    cache["generated_at"] = datetime.now(timezone.utc).isoformat()
    return cache

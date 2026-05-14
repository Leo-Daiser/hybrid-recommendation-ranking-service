import pandas as pd
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
    coverage_at_k
)

def build_ground_truth(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    label_col: str = "label",
    positive_label: int = 1,
) -> dict[int, set[int]]:
    pos_interactions = interactions[interactions[label_col] == positive_label]
    return pos_interactions.groupby(user_col)[item_col].apply(set).to_dict()

def evaluate_recommendations(
    candidate_cache: pd.DataFrame,
    ground_truth: dict[int, set[int]],
    total_items: int,
    k_values: list[int],
    user_col: str = "user_id",
    item_col: str = "item_id",
    rank_col: str = "rank",
    score_col: str = "retrieval_score",
    exclude_users_without_ground_truth: bool = True,
) -> pd.DataFrame:
    
    df = candidate_cache.sort_values(by=[user_col, rank_col])
    recs_by_user = df.groupby(user_col)[item_col].apply(list).to_dict()
    
    users = list(recs_by_user.keys())
    if exclude_users_without_ground_truth:
        users = [u for u in users if u in ground_truth and ground_truth[u]]
        
    results = []
    
    for k in k_values:
        precisions = []
        recalls = []
        maps = []
        ndcgs = []
        
        for u in users:
            recs = recs_by_user.get(u, [])
            relevant = ground_truth.get(u, set())
            
            if not relevant and exclude_users_without_ground_truth:
                continue
                
            precisions.append(precision_at_k(recs, relevant, k))
            recalls.append(recall_at_k(recs, relevant, k))
            maps.append(average_precision_at_k(recs, relevant, k))
            ndcgs.append(ndcg_at_k(recs, relevant, k))
            
        cov = coverage_at_k(candidate_cache, total_items, item_col=item_col, rank_col=rank_col, k=k)
        
        avg_prec = sum(precisions) / len(precisions) if precisions else 0.0
        avg_rec = sum(recalls) / len(recalls) if recalls else 0.0
        avg_map = sum(maps) / len(maps) if maps else 0.0
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        
        users_eval = len(users)
        rel_counts = [len(ground_truth.get(u, set())) for u in users]
        avg_rel = sum(rel_counts) / users_eval if users_eval > 0 else 0.0
        
        results.append({
            "k": k,
            "precision": avg_prec,
            "recall": avg_rec,
            "map": avg_map,
            "ndcg": avg_ndcg,
            "coverage": cov,
            "users_evaluated": users_eval,
            "users_with_recommendations": len(recs_by_user),
            "users_with_ground_truth": len([u for u, items in ground_truth.items() if items]),
            "avg_relevant_items_per_user": avg_rel
        })
        
    return pd.DataFrame(results)

def compare_retrieval_models(
    candidate_caches: dict[str, pd.DataFrame],
    ground_truth: dict[int, set[int]],
    total_items: int,
    k_values: list[int],
    exclude_users_without_ground_truth: bool = True,
) -> pd.DataFrame:
    
    all_res = []
    for model_name, cache in candidate_caches.items():
        res = evaluate_recommendations(
            candidate_cache=cache,
            ground_truth=ground_truth,
            total_items=total_items,
            k_values=k_values,
            exclude_users_without_ground_truth=exclude_users_without_ground_truth
        )
        res.insert(0, "model_name", model_name)
        all_res.append(res)
        
    return pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame()

import pandas as pd
import numpy as np
from src.evaluation.metrics import precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k

def attach_scores(
    ranking_df: pd.DataFrame,
    scores: np.ndarray,
    score_col: str = "ranking_score",
) -> pd.DataFrame:
    df = ranking_df.copy()
    df[score_col] = scores
    return df

def rank_by_model_score(
    scored_df: pd.DataFrame,
    user_col: str = "user_id",
    score_col: str = "ranking_score",
) -> pd.DataFrame:
    if scored_df.empty:
        df = scored_df.copy()
        df["model_rank"] = pd.Series(dtype=int)
        return df
        
    df = scored_df.sort_values(
        by=[user_col, score_col], ascending=[True, False]
    )
    df["model_rank"] = df.groupby(user_col).cumcount() + 1
    return df

def evaluate_ranked_candidates(
    scored_df: pd.DataFrame,
    k_values: list[int],
    user_col: str = "user_id",
    item_col: str = "item_id",
    target_col: str = "target",
    rank_col: str = "model_rank",
) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()
        
    positives = scored_df[scored_df[target_col] == 1]
    ground_truth = positives.groupby(user_col)[item_col].apply(set).to_dict()
    
    users_with_gt = set(ground_truth.keys())
    if not users_with_gt:
        return pd.DataFrame()
        
    df = scored_df[scored_df[user_col].isin(users_with_gt)]
    
    recs_grouped = df.sort_values(rank_col).groupby(user_col)[item_col].apply(list).to_dict()
    
    results = []
    for k in k_values:
        prec_list = []
        rec_list = []
        map_list = []
        ndcg_list = []
        
        for u in users_with_gt:
            recs = recs_grouped.get(u, [])
            gt = ground_truth[u]
            
            prec_list.append(precision_at_k(recs, gt, k))
            rec_list.append(recall_at_k(recs, gt, k))
            map_list.append(average_precision_at_k(recs, gt, k))
            ndcg_list.append(ndcg_at_k(recs, gt, k))
            
        results.append({
            "k": k,
            "precision": np.mean(prec_list),
            "recall": np.mean(rec_list),
            "map": np.mean(map_list),
            "ndcg": np.mean(ndcg_list),
        })
        
    return pd.DataFrame(results)

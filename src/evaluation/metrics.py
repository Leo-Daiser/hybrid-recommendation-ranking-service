import pandas as pd
import numpy as np
import math

def precision_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    rec_k = recommended_items[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevant_items)
    return hits / k

def recall_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    rec_k = recommended_items[:k]
    hits = sum(1 for item in rec_k if item in relevant_items)
    return hits / len(relevant_items)

def average_precision_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    rec_k = recommended_items[:k]
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant_items:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / min(len(relevant_items), k)

def dcg_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    rec_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant_items:
            dcg += 1.0 / math.log2(i + 2)
    return dcg

def ndcg_at_k(recommended_items: list[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    dcg = dcg_at_k(recommended_items, relevant_items, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_items), k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg

def coverage_at_k(
    recommendations: pd.DataFrame,
    total_items: int,
    item_col: str = "item_id",
    rank_col: str = "rank",
    k: int = 10,
) -> float:
    if total_items == 0:
        return 0.0
    rec_k = recommendations[recommendations[rank_col] <= k]
    unique_items = rec_k[item_col].nunique()
    return unique_items / total_items

import pytest
import pandas as pd
from src.ranking.dataset import (
    merge_candidate_caches,
    build_future_positive_pairs,
    label_candidate_pairs,
    add_user_item_features,
    sample_negatives,
    build_ranking_dataset
)

def test_merge_candidate_caches_success():
    pop = pd.DataFrame({"user_id": [1], "item_id": [10], "retrieval_score": [0.5], "rank": [1]})
    knn = pd.DataFrame({"user_id": [2], "item_id": [20], "retrieval_score": [0.8], "rank": [1]})
    res = merge_candidate_caches({"pop": pop, "knn": knn})
    assert len(res) == 2
    assert "max_retrieval_score" in res.columns
    assert "came_from_popularity" in res.columns
    assert "came_from_itemknn" in res.columns

def test_merge_candidate_caches_deduplicates_user_item_pairs():
    pop = pd.DataFrame({"user_id": [1], "item_id": [10], "retrieval_score": [0.5], "rank": [2]})
    knn = pd.DataFrame({"user_id": [1], "item_id": [10], "retrieval_score": [0.8], "rank": [1]})
    res = merge_candidate_caches({"popularity_v1": pop, "itemknn_cosine_v1": knn})
    assert len(res) == 1
    assert res.iloc[0]["max_retrieval_score"] == 0.8
    assert res.iloc[0]["min_rank"] == 1
    assert res.iloc[0]["retrieval_model_count"] == 2
    assert res.iloc[0]["came_from_popularity"] == 1
    assert res.iloc[0]["came_from_itemknn"] == 1

def test_build_future_positive_pairs_uses_only_positive_labels():
    df = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20], "label": [1, 0]})
    res = build_future_positive_pairs(df)
    assert res == {(1, 10)}

def test_label_candidate_pairs_success():
    cands = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
    pos = {(1, 10)}
    res = label_candidate_pairs(cands, pos)
    assert res.iloc[0]["target"] == 1
    assert res.iloc[1]["target"] == 0

def test_add_user_item_features_preserves_candidate_rows():
    cands = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20], "target": [1, 0]})
    uf = pd.DataFrame({"user_id": [1], "user_feat": [99]})
    itf = pd.DataFrame({"item_id": [20], "item_feat": [88]})
    res = add_user_item_features(cands, uf, itf)
    assert len(res) == 2
    assert res.loc[0, "user_feat"] == 99
    assert pd.isna(res.loc[1, "user_feat"])
    assert res.loc[1, "item_feat"] == 88

def test_sample_negatives_keeps_all_positives():
    df = pd.DataFrame({"user_id": [1, 2, 3], "target": [1, 1, 0]})
    res = sample_negatives(df, keep_all_positives=True)
    assert len(res[res["target"] == 1]) == 2

def test_sample_negatives_respects_ratio():
    df = pd.DataFrame({"user_id": range(11), "target": [1] + [0]*10})
    res = sample_negatives(df, negative_to_positive_ratio=2)
    assert len(res[res["target"] == 1]) == 1
    assert len(res[res["target"] == 0]) == 2
    assert len(res) == 3

def test_sample_negatives_reproducible():
    df = pd.DataFrame({"user_id": range(101), "target": [1] + [0]*100})
    res1 = sample_negatives(df, negative_to_positive_ratio=2, random_seed=42)
    res2 = sample_negatives(df, negative_to_positive_ratio=2, random_seed=42)
    assert res1.equals(res2)

def test_build_ranking_dataset_success():
    cands = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [10, 20, 30]})
    future = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]})
    res = build_ranking_dataset(
        cands, future, pd.DataFrame(), pd.DataFrame(), negative_to_positive_ratio=1
    )
    assert len(res) == 2
    assert "target" in res.columns

def test_build_ranking_dataset_no_positive_labels_does_not_crash():
    cands = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
    future = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [0]})
    res = build_ranking_dataset(cands, future, pd.DataFrame(), pd.DataFrame())
    assert "target" in res.columns
    assert (res["target"] == 0).all()

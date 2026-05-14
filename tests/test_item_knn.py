import pytest
import pandas as pd
import numpy as np
from src.retrieval.item_knn import (
    build_user_item_matrix,
    build_item_similarity_topk,
    recommend_itemknn_for_user,
    build_itemknn_candidate_cache
)

def test_build_item_similarity_topk_success():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 10, 20],
        "label": [1, 1, 1, 1]
    })
    sim = build_item_similarity_topk(interactions)
    assert not sim.empty
    assert "item_id" in sim.columns
    assert "similar_item_id" in sim.columns
    val = sim[(sim["item_id"] == 10) & (sim["similar_item_id"] == 20)]["similarity"].iloc[0]
    assert np.isclose(val, 1.0)

def test_build_item_similarity_topk_uses_only_positive_labels():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 10, 20],
        "label": [0, 0, 0, 0]
    })
    sim = build_item_similarity_topk(interactions)
    assert sim.empty

def test_build_item_similarity_topk_excludes_self_similarity():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "item_id": [10, 20, 10, 20],
        "label": [1, 1, 1, 1]
    })
    sim = build_item_similarity_topk(interactions)
    assert not (sim["item_id"] == sim["similar_item_id"]).any()

def test_build_item_similarity_topk_respects_max_neighbors():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "item_id": [10, 20, 30, 40],
        "label": [1, 1, 1, 1]
    })
    sim = build_item_similarity_topk(interactions, max_neighbors_per_item=2)
    counts = sim.groupby("item_id").size()
    assert (counts <= 2).all()

def test_recommend_itemknn_for_known_user_success():
    train = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]})
    sim = pd.DataFrame({
        "item_id": [10],
        "similar_item_id": [20],
        "similarity": [0.9]
    })
    fallback = pd.DataFrame()
    res = recommend_itemknn_for_user(1, train, sim, fallback)
    assert res.iloc[0]["item_id"] == 20
    assert res.iloc[0]["retrieval_score"] == 0.9

def test_recommend_itemknn_excludes_seen_items():
    train = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [10, 20],
        "label": [1, 0]
    })
    sim = pd.DataFrame({
        "item_id": [10, 10],
        "similar_item_id": [20, 30],
        "similarity": [0.9, 0.8]
    })
    res = recommend_itemknn_for_user(1, train, sim, pd.DataFrame())
    assert 20 not in res["item_id"].tolist()
    assert 30 in res["item_id"].tolist()

def test_recommend_itemknn_unknown_user_uses_popularity_fallback():
    train = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]})
    sim = pd.DataFrame({"item_id": [10], "similar_item_id": [20], "similarity": [0.9]})
    fallback = pd.DataFrame({
        "user_id": [999],
        "item_id": [50],
        "retrieval_score": [1.0],
        "rank": [1],
        "retrieval_model": ["pop"]
    })
    res = recommend_itemknn_for_user(999, train, sim, fallback)
    assert res.iloc[0]["item_id"] == 50
    assert res.iloc[0]["retrieval_model"] == "fallback_popularity"

def test_recommend_itemknn_user_without_positive_history_uses_popularity_fallback():
    train = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [0]})
    sim = pd.DataFrame()
    fallback = pd.DataFrame({"user_id": [1], "item_id": [50], "retrieval_score": [1.0], "rank": [1], "retrieval_model": ["pop"]})
    res = recommend_itemknn_for_user(1, train, sim, fallback)
    assert res.iloc[0]["item_id"] == 50

def test_recommend_itemknn_rank_starts_at_one():
    train = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]})
    sim = pd.DataFrame({"item_id": [10, 10], "similar_item_id": [20, 30], "similarity": [0.9, 0.8]})
    res = recommend_itemknn_for_user(1, train, sim, pd.DataFrame())
    assert res.iloc[0]["rank"] == 1

def test_build_itemknn_candidate_cache_success():
    train = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20], "label": [1, 1]})
    sim = pd.DataFrame({
        "item_id": [10, 20],
        "similar_item_id": [30, 40],
        "similarity": [0.9, 0.8]
    })
    fallback = pd.DataFrame()
    cache = build_itemknn_candidate_cache(train, sim, fallback)
    assert len(cache) == 2

def test_build_itemknn_candidate_cache_no_duplicate_pairs():
    train = pd.DataFrame({"user_id": [1], "item_id": [10], "label": [1]})
    sim = pd.DataFrame({
        "item_id": [10, 10],
        "similar_item_id": [30, 30],
        "similarity": [0.5, 0.4]
    })
    cache = build_itemknn_candidate_cache(train, sim, pd.DataFrame())
    assert not cache.duplicated(subset=["user_id", "item_id"]).any()
    assert cache.iloc[0]["retrieval_score"] == 0.9

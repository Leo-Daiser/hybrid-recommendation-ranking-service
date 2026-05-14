import pytest
import pandas as pd
from src.retrieval.popularity import (
    build_popularity_ranking,
    get_seen_items_by_user,
    recommend_popular_for_user,
    build_popularity_candidate_cache
)

def test_build_popularity_ranking_success():
    items = pd.DataFrame({
        "item_id": [1, 2, 3],
        "item_positive_ratio": [0.5, 0.9, 0.9],
        "item_rating_count": [100, 10, 50]
    })
    res = build_popularity_ranking(items)
    assert res.iloc[0]["item_id"] == 3
    assert res.iloc[1]["item_id"] == 2
    assert res.iloc[2]["item_id"] == 1
    assert "retrieval_score" in res.columns
    assert "popularity_rank" in res.columns

def test_build_popularity_ranking_missing_score_column():
    items = pd.DataFrame({"item_id": [1]})
    with pytest.raises(ValueError):
        build_popularity_ranking(items)

def test_get_seen_items_by_user_success():
    interactions = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": [10, 20, 30]
    })
    seen = get_seen_items_by_user(interactions)
    assert seen[1] == {10, 20}
    assert seen[2] == {30}

def test_recommend_popular_for_known_user_excludes_seen_items():
    pop_ranking = pd.DataFrame({
        "item_id": [10, 20, 30, 40],
        "retrieval_score": [0.9, 0.8, 0.7, 0.6]
    })
    seen = {1: {10, 20}}
    res = recommend_popular_for_user(1, pop_ranking, seen, top_k=2)
    assert res["item_id"].tolist() == [30, 40]
    assert res["rank"].tolist() == [1, 2]

def test_recommend_popular_for_unknown_user_uses_cold_start_fallback():
    pop_ranking = pd.DataFrame({
        "item_id": [10, 20, 30, 40],
        "retrieval_score": [0.9, 0.8, 0.7, 0.6]
    })
    seen = {1: {10, 20}}
    res = recommend_popular_for_user(999, pop_ranking, seen, top_k=2)
    assert res["item_id"].tolist() == [10, 20]
    assert res["rank"].tolist() == [1, 2]

def test_recommend_popular_for_user_rank_starts_at_one():
    pop_ranking = pd.DataFrame({
        "item_id": [10, 20],
        "retrieval_score": [0.9, 0.8]
    })
    res = recommend_popular_for_user(1, pop_ranking, {})
    assert res.iloc[0]["rank"] == 1

def test_build_popularity_candidate_cache_success():
    interactions = pd.DataFrame({"user_id": [1, 2], "item_id": [10, 20]})
    items = pd.DataFrame({
        "item_id": [10, 20, 30],
        "item_positive_ratio": [1.0, 1.0, 1.0],
        "item_rating_count": [10, 5, 1]
    })
    cache = build_popularity_candidate_cache(interactions, items, top_k=2)
    # user 1 has seen 10. Top overall is 10, 20, 30. User 1 gets 20, 30.
    # user 2 has seen 20. Top overall is 10, 20, 30. User 2 gets 10, 30.
    u1 = cache[cache["user_id"] == 1]
    u2 = cache[cache["user_id"] == 2]
    assert u1["item_id"].tolist() == [20, 30]
    assert u2["item_id"].tolist() == [10, 30]

def test_build_popularity_candidate_cache_no_seen_items():
    interactions = pd.DataFrame({"user_id": [1], "item_id": [10]})
    items = pd.DataFrame({
        "item_id": [10, 20],
        "item_positive_ratio": [1.0, 1.0],
        "item_rating_count": [10, 5]
    })
    cache = build_popularity_candidate_cache(interactions, items, top_k=10)
    assert 10 not in cache[cache["user_id"] == 1]["item_id"].tolist()

def test_build_popularity_candidate_cache_no_duplicate_pairs():
    interactions = pd.DataFrame({"user_id": [1], "item_id": [10]})
    items = pd.DataFrame({
        "item_id": [20],
        "item_positive_ratio": [1.0],
        "item_rating_count": [10]
    })
    cache = build_popularity_candidate_cache(interactions, items)
    assert not cache.duplicated(subset=["user_id", "item_id"]).any()

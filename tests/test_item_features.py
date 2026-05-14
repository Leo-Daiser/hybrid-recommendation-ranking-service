import pytest
import pandas as pd
from src.features.item_features import build_item_features, parse_genres

def test_parse_genres_success():
    movies = pd.DataFrame({"movieId": [1], "genres": ["Action|Comedy"]})
    res = parse_genres(movies, "movieId", "genres")
    assert len(res) == 2
    assert "genre" in res.columns
    assert "has_genre" in res.columns
    assert set(res["genre"]) == {"Action", "Comedy"}

def test_parse_genres_handles_no_genres():
    movies = pd.DataFrame({"movieId": [1], "genres": ["(no genres listed)"]})
    res = parse_genres(movies)
    assert res.iloc[0]["genre"] == "(no genres listed)"

def test_build_item_features_success():
    interactions = pd.DataFrame({
        "item_id": [10, 10],
        "rating": [4.0, 5.0],
        "label": [1, 1],
        "timestamp": [10, 20]
    })
    movies = pd.DataFrame({
        "movieId": [10],
        "genres": ["Action"]
    })
    res = build_item_features(interactions, movies)
    assert "item_id" in res.columns
    assert "item_genres" in res.columns
    assert res.iloc[0]["item_genre_count"] == 1

def test_build_item_features_popularity_rank():
    interactions = pd.DataFrame({
        "item_id": [10, 10, 20],
        "rating": [4.0, 5.0, 1.0],
        "label": [1, 1, 0],
        "timestamp": [1, 2, 3]
    })
    movies = pd.DataFrame({
        "movieId": [10, 20],
        "genres": ["Action", "Comedy"]
    })
    res = build_item_features(interactions, movies)
    res = res.sort_values("item_id")
    rank_10 = res[res["item_id"] == 10]["item_popularity_rank"].iloc[0]
    rank_20 = res[res["item_id"] == 20]["item_popularity_rank"].iloc[0]
    assert rank_10 < rank_20

def test_build_item_features_positive_ratio():
    interactions = pd.DataFrame({
        "item_id": [10, 10, 10, 10],
        "rating": [5.0, 5.0, 5.0, 1.0],
        "label": [1, 1, 1, 0],
        "timestamp": [1, 2, 3, 4]
    })
    movies = pd.DataFrame({"movieId": [10], "genres": ["Action"]})
    res = build_item_features(interactions, movies)
    assert res.iloc[0]["item_positive_ratio"] == 0.75

def test_build_item_features_does_not_mutate_input():
    interactions = pd.DataFrame({
        "item_id": [10],
        "rating": [5.0],
        "label": [1],
        "timestamp": [1]
    })
    movies = pd.DataFrame({"movieId": [10], "genres": ["Action"]})
    int_copy = interactions.copy()
    mov_copy = movies.copy()
    build_item_features(interactions, movies)
    pd.testing.assert_frame_equal(interactions, int_copy)
    pd.testing.assert_frame_equal(movies, mov_copy)

import pytest
import pandas as pd
from src.features.user_features import build_user_features

def test_build_user_features_success():
    df = pd.DataFrame({
        "user_id": [1, 1],
        "rating": [4.0, 5.0],
        "label": [1, 1],
        "timestamp": [10, 20]
    })
    res = build_user_features(df)
    assert "user_id" in res.columns
    assert "user_rating_count" in res.columns
    assert res.iloc[0]["user_rating_count"] == 2
    assert res.iloc[0]["user_std_rating"] > 0

def test_build_user_features_positive_ratio():
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 1],
        "rating": [5.0, 5.0, 5.0, 1.0],
        "label": [1, 1, 1, 0],
        "timestamp": [1, 2, 3, 4]
    })
    res = build_user_features(df)
    assert res.iloc[0]["user_positive_ratio"] == 0.75

def test_build_user_features_activity_span():
    df = pd.DataFrame({
        "user_id": [1, 1],
        "rating": [5.0, 1.0],
        "label": [1, 0],
        "timestamp": [100, 200]
    })
    res = build_user_features(df)
    assert res.iloc[0]["user_activity_span"] == 100

def test_build_user_features_does_not_mutate_input():
    df = pd.DataFrame({
        "user_id": [1],
        "rating": [5.0],
        "label": [1],
        "timestamp": [1]
    })
    df_copy = df.copy()
    build_user_features(df)
    pd.testing.assert_frame_equal(df, df_copy)

import pytest
import pandas as pd
import numpy as np
from src.ranking.train_ranker import (
    prepare_ranker_features,
    train_logistic_regression_ranker,
    predict_ranker_scores
)

def test_prepare_ranker_features_excludes_columns():
    train = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [1, 0], "user_id": [1, 2]})
    valid = pd.DataFrame({"a": [2], "b": [5], "target": [1], "user_id": [1]})
    
    Xt, yt, Xv, yv, feats, fill_vals = prepare_ranker_features(train, valid, "target", ["user_id"])
    assert "user_id" not in Xt.columns
    assert "target" not in Xt.columns
    assert list(Xt.columns) == ["a", "b"]
    assert len(feats) == 2

def test_prepare_ranker_features_uses_only_numeric_columns():
    train = pd.DataFrame({"a": [1], "str_col": ["hello"], "target": [1]})
    Xt, yt, Xv, yv, feats, fill_vals = prepare_ranker_features(train, train, "target", [])
    assert "str_col" not in Xt.columns

def test_prepare_ranker_features_fills_nan():
    train = pd.DataFrame({"a": [1, np.nan, 3], "target": [1, 0, 1]})
    Xt, yt, Xv, yv, feats, fill_vals = prepare_ranker_features(train, train, "target", [])
    assert Xt.isna().sum().sum() == 0
    assert Xt.iloc[1]["a"] == 2.0
    assert "a" in fill_vals

def test_train_logistic_regression_ranker_success():
    Xt = pd.DataFrame({"a": [1, 2, 3], "b": [0, 0, 1]})
    yt = pd.Series([0, 1, 1])
    model = train_logistic_regression_ranker(Xt, yt)
    assert model is not None

def test_predict_ranker_scores_returns_probabilities():
    Xt = pd.DataFrame({"a": [1, 2, 3]})
    yt = pd.Series([0, 1, 1])
    model = train_logistic_regression_ranker(Xt, yt)
    
    scores = predict_ranker_scores(model, Xt)
    assert len(scores) == 3
    assert all((scores >= 0) & (scores <= 1))

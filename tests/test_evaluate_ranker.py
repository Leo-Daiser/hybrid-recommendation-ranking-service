import pytest
import pandas as pd
import numpy as np
from src.ranking.evaluate_ranker import (
    attach_scores,
    rank_by_model_score,
    evaluate_ranked_candidates
)

def test_attach_scores_success():
    df = pd.DataFrame({"user_id": [1, 2]})
    scores = np.array([0.9, 0.1])
    res = attach_scores(df, scores)
    assert "ranking_score" in res.columns
    assert res.iloc[0]["ranking_score"] == 0.9

def test_rank_by_model_score_orders_within_user():
    df = pd.DataFrame({"user_id": [1, 1, 2], "ranking_score": [0.5, 0.9, 0.2]})
    res = rank_by_model_score(df)
    assert list(res["ranking_score"]) == [0.9, 0.5, 0.2]
    assert list(res["model_rank"]) == [1, 2, 1]

def test_evaluate_ranked_candidates_success():
    df = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [10, 20],
        "target": [1, 0],
        "model_rank": [1, 2]
    })
    res = evaluate_ranked_candidates(df, [1])
    assert not res.empty
    assert res.iloc[0]["precision"] == 1.0

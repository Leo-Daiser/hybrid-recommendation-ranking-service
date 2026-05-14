import pytest
import pandas as pd
from src.evaluation.offline_evaluator import (
    build_ground_truth,
    evaluate_recommendations,
    compare_retrieval_models
)

def test_build_ground_truth_uses_only_positive_labels():
    df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": [10, 20, 30],
        "label": [1, 0, 1]
    })
    gt = build_ground_truth(df)
    assert gt[1] == {10}
    assert gt[2] == {30}

def test_evaluate_recommendations_success():
    cache = pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [10, 20],
        "rank": [1, 2],
        "retrieval_score": [1.0, 0.9]
    })
    gt = {1: {10}}
    res = evaluate_recommendations(cache, gt, total_items=100, k_values=[2])
    assert len(res) == 1
    assert res.iloc[0]["precision"] == 0.5
    assert res.iloc[0]["users_evaluated"] == 1

def test_evaluate_recommendations_excludes_users_without_ground_truth():
    cache = pd.DataFrame({
        "user_id": [1, 2],
        "item_id": [10, 20],
        "rank": [1, 1],
        "retrieval_score": [1.0, 0.9]
    })
    gt = {1: {10}, 2: set()}
    res = evaluate_recommendations(cache, gt, total_items=100, k_values=[1], exclude_users_without_ground_truth=True)
    assert res.iloc[0]["users_evaluated"] == 1

def test_compare_retrieval_models_success():
    cache1 = pd.DataFrame({"user_id": [1], "item_id": [10], "rank": [1], "retrieval_score": [1.0]})
    cache2 = pd.DataFrame({"user_id": [1], "item_id": [20], "rank": [1], "retrieval_score": [1.0]})
    gt = {1: {10}}
    res = compare_retrieval_models({"m1": cache1, "m2": cache2}, gt, total_items=10, k_values=[1])
    assert len(res) == 2
    assert "model_name" in res.columns
    assert res[res["model_name"] == "m1"]["precision"].iloc[0] == 1.0
    assert res[res["model_name"] == "m2"]["precision"].iloc[0] == 0.0

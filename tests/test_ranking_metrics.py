import pytest
import pandas as pd
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
    coverage_at_k
)

def test_precision_at_k():
    assert precision_at_k([1, 2, 3], {2, 3, 4}, 2) == 0.5
    assert precision_at_k([1, 2, 3], set(), 2) == 0.0

def test_recall_at_k():
    assert recall_at_k([1, 2, 3], {2, 3, 4, 5}, 3) == 0.5
    assert recall_at_k([1, 2, 3], set(), 2) == 0.0

def test_average_precision_at_k():
    ap = average_precision_at_k([1, 2, 3], {1, 3}, 3)
    assert pytest.approx(ap) == (1.0 + 2.0/3.0) / 2.0

def test_ndcg_at_k():
    import math
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert pytest.approx(ndcg_at_k([1, 2], {2}, 2)) == expected

def test_metrics_empty_relevant_items():
    assert precision_at_k([1, 2], set(), 2) == 0.0
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert average_precision_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0

def test_coverage_at_k():
    recs = pd.DataFrame({"item_id": [1, 2, 3], "rank": [1, 2, 3]})
    assert coverage_at_k(recs, total_items=10, k=2) == 0.2
    assert coverage_at_k(recs, total_items=0, k=2) == 0.0

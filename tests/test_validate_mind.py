import pandas as pd
from src.data.validate_mind import (
    validate_mind_behaviors,
    validate_mind_news,
    validate_mind_outputs
)

def test_validate_mind_behaviors_missing_columns():
    df = pd.DataFrame({"impression_id": [1], "user_id": ["U1"]})
    report = validate_mind_behaviors(df)
    assert report["status"] == "failed"
    assert any("Missing behavior columns" in e for e in report["errors"])

def test_validate_mind_news_missing_columns():
    df = pd.DataFrame({"item_id": ["N1"]})
    report = validate_mind_news(df)
    assert report["status"] == "failed"
    assert any("Missing news columns" in e for e in report["errors"])

def test_validate_mind_outputs_fk_report():
    train_int = pd.DataFrame({"item_id": ["N1", "N2"], "label": [1, 0]})
    valid_int = pd.DataFrame({"item_id": ["N1"], "label": [1]})
    items = pd.DataFrame({"item_id": ["N1"]}) # N2 missing
    
    report = validate_mind_outputs(train_int, valid_int, items)
    assert report["status"] == "ok"
    assert len(report["errors"]) == 0
    assert len(report["warnings"]) == 1
    assert "Found 1 items in train_interactions that are not in items DataFrame" in report["warnings"][0]

def test_validate_mind_outputs_invalid_labels():
    train_int = pd.DataFrame({"item_id": ["N1"], "label": [2]})
    items = pd.DataFrame({"item_id": ["N1"]})
    
    report = validate_mind_outputs(train_int, pd.DataFrame(), items)
    assert report["status"] == "failed"
    assert any("invalid labels" in e for e in report["errors"])

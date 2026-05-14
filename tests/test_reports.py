import pytest
import pandas as pd
from src.evaluation.reports import dataframe_to_markdown_table, build_retrieval_report_markdown

def test_dataframe_to_markdown_table():
    df = pd.DataFrame({
        "model_name": ["m1", "m2"],
        "k": [10, 10],
        "precision": [0.5, 0.4]
    })
    md = dataframe_to_markdown_table(df)
    assert "m1" in md
    assert "0.5" in md
    assert "---" in md
    assert md.startswith("| model_name | k | precision |")

def test_build_retrieval_report_markdown_without_tabulate():
    df = pd.DataFrame({
        "model_name": ["m1", "m2"],
        "k": [10, 10],
        "precision": [0.5, 0.4],
        "recall": [0.1, 0.2],
        "map": [0.3, 0.4],
        "ndcg": [0.8, 0.9],
        "coverage": [0.1, 0.2]
    })
    
    report = build_retrieval_report_markdown(df, dataset_name="Test")
    assert "Test" in report
    assert "m1" in report
    assert "m2" in report
    assert "0.5000" in report # checking formatting
    assert "Best model by NDCG@10**: m2" in report

import pandas as pd
from typing import Any

from src.data.mind_schema import get_mind_behavior_columns, get_mind_news_columns

def validate_mind_behaviors(df: pd.DataFrame) -> dict:
    report = {"status": "ok", "errors": []}
    
    if df.empty:
        report["status"] = "failed"
        report["errors"].append("Behaviors DataFrame is empty")
        return report
        
    required = set(get_mind_behavior_columns())
    missing = required - set(df.columns)
    if missing:
        report["status"] = "failed"
        report["errors"].append(f"Missing behavior columns: {missing}")
        
    return report

def validate_mind_news(df: pd.DataFrame) -> dict:
    report = {"status": "ok", "errors": []}
    
    if df.empty:
        report["status"] = "failed"
        report["errors"].append("News DataFrame is empty")
        return report
        
    required = set(get_mind_news_columns())
    missing = required - set(df.columns)
    if missing:
        report["status"] = "failed"
        report["errors"].append(f"Missing news columns: {missing}")
        
    return report

def validate_mind_outputs(
    train_interactions: pd.DataFrame,
    valid_interactions: pd.DataFrame,
    items: pd.DataFrame,
) -> dict:
    report = {"status": "ok", "errors": [], "warnings": []}
    
    if train_interactions.empty:
        report["status"] = "failed"
        report["errors"].append("train_interactions is empty")
        
    if items.empty:
        report["status"] = "failed"
        report["errors"].append("items DataFrame is empty")
        
    if not train_interactions.empty:
        invalid_labels = train_interactions[~train_interactions["label"].isin([0, 1])]
        if not invalid_labels.empty:
            report["status"] = "failed"
            report["errors"].append(f"Found invalid labels in train_interactions: {invalid_labels['label'].unique()}")
            
        train_items_set = set(train_interactions["item_id"])
        known_items_set = set(items["item_id"])
        unknown_items = train_items_set - known_items_set
        
        if unknown_items:
            # FK violations are just warnings if strict_foreign_keys=false
            report["warnings"].append(f"Found {len(unknown_items)} items in train_interactions that are not in items DataFrame")
            
    return report

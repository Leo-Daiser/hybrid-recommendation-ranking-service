import yaml
import logging
from pathlib import Path
import pandas as pd
from typing import Any

from src.data.mind_schema import parse_behaviors_tsv, parse_news_tsv

logger = logging.getLogger(__name__)

def load_mind_config(config_path: str | Path = "configs/mind.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_impression_tokens(impressions: str) -> list[dict]:
    """Parse MIND impressions string like 'N12345-1 N67890-0 N11111-0'."""
    if not isinstance(impressions, str) or not impressions.strip():
        return []
    
    parsed = []
    tokens = impressions.split()
    for token in tokens:
        if "-" not in token:
            logger.warning(f"Skipping malformed token: {token}")
            continue
        parts = token.rsplit("-", 1)
        if len(parts) != 2:
            logger.warning(f"Skipping malformed token: {token}")
            continue
            
        item_id, label_str = parts
        try:
            label = int(label_str)
            parsed.append({"item_id": str(item_id), "label": int(label)})
        except ValueError:
            logger.warning(f"Skipping token with invalid label: {token}")
            
    return parsed

def convert_behaviors_to_interactions(
    behaviors: pd.DataFrame,
    positive_label: int = 1,
    negative_label: int = 0,
    source: str = "mind",
) -> pd.DataFrame:
    records = []
    
    for row in behaviors.itertuples(index=False):
        try:
            # pd.to_datetime can be slow, but let's try a simple approach
            timestamp = pd.to_datetime(row.time)
        except Exception:
            timestamp = pd.NaT
            
        impression_items = parse_impression_tokens(row.impressions)
        
        for imp in impression_items:
            records.append({
                "impression_id": row.impression_id,
                "user_id": row.user_id,
                "item_id": imp["item_id"],
                "timestamp": timestamp,
                "label": positive_label if imp["label"] == 1 else negative_label,
                "source": source
            })
            
    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["label"] = pd.Series(
            [int(x) for x in df["label"].tolist()],
            index=df.index,
            dtype=object,
        )
    else:
        df = pd.DataFrame(columns=["impression_id", "user_id", "item_id", "timestamp", "label", "source"])
        
    return df

def convert_behaviors_to_impressions(
    behaviors: pd.DataFrame,
    positive_label: int = 1,
    negative_label: int = 0,
) -> pd.DataFrame:
    records = []
    
    for row in behaviors.itertuples(index=False):
        try:
            timestamp = pd.to_datetime(row.time)
        except Exception:
            timestamp = pd.NaT
            
        impression_items = parse_impression_tokens(row.impressions)
        
        pos_items = [imp["item_id"] for imp in impression_items if imp["label"] == 1]
        neg_items = [imp["item_id"] for imp in impression_items if imp["label"] == 0]
        
        records.append({
            "impression_id": row.impression_id,
            "user_id": row.user_id,
            "timestamp": timestamp,
            "history": row.history if pd.notna(row.history) else "",
            "impressions": row.impressions if pd.notna(row.impressions) else "",
            "positive_items": pos_items,
            "negative_items": neg_items,
            "positive_count": len(pos_items),
            "negative_count": len(neg_items),
            "total_impression_items": len(impression_items)
        })
        
    df = pd.DataFrame(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df = pd.DataFrame(columns=["impression_id", "user_id", "timestamp", "history", "impressions", "positive_items", "negative_items", "positive_count", "negative_count", "total_impression_items"])
    return df

def prepare_mind_items(
    news_train: pd.DataFrame,
    news_valid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if news_valid is not None and not news_valid.empty:
        items = pd.concat([news_train, news_valid], ignore_index=True)
    else:
        items = news_train.copy()
        
    items = items.drop_duplicates(subset=["item_id"], keep="first").reset_index(drop=True)
    items["source"] = "mind"
    return items

def run_prepare_mind(mind_config_path: str | Path = "configs/mind.yaml") -> dict[str, pd.DataFrame]:
    config = load_mind_config(mind_config_path)["mind"]
    raw_dir = Path(config["raw_data_dir"])
    
    expected = config["expected_files"]
    train_behaviors_path = raw_dir / expected["train_behaviors"]
    train_news_path = raw_dir / expected["train_news"]
    valid_behaviors_path = raw_dir / expected["valid_behaviors"]
    valid_news_path = raw_dir / expected["valid_news"]
    
    # Check if raw files exist
    if not train_behaviors_path.exists() or not train_news_path.exists():
        raise FileNotFoundError("MIND raw files not found. Please download MINDsmall and place files under data/raw/mind/MINDsmall/")
        
    train_behaviors = parse_behaviors_tsv(train_behaviors_path)
    train_news = parse_news_tsv(train_news_path)
    
    valid_behaviors = parse_behaviors_tsv(valid_behaviors_path) if valid_behaviors_path.exists() else None
    valid_news = parse_news_tsv(valid_news_path) if valid_news_path.exists() else None
    
    pos_label = config["parsing"]["positive_label"]
    neg_label = config["parsing"]["negative_label"]
    source = config["parsing"]["source"]
    
    train_interactions = convert_behaviors_to_interactions(train_behaviors, pos_label, neg_label, source)
    
    if valid_behaviors is not None:
        valid_interactions = convert_behaviors_to_interactions(valid_behaviors, pos_label, neg_label, source)
    else:
        valid_interactions = pd.DataFrame(columns=train_interactions.columns)
        
    impressions = pd.concat([
        convert_behaviors_to_impressions(train_behaviors, pos_label, neg_label),
        convert_behaviors_to_impressions(valid_behaviors, pos_label, neg_label) if valid_behaviors is not None else pd.DataFrame()
    ], ignore_index=True)
    
    items = prepare_mind_items(train_news, valid_news)
    
    # Save outputs
    out_cfg = config["output"]
    
    Path(out_cfg["train_interactions_path"]).parent.mkdir(parents=True, exist_ok=True)
    
    train_interactions.to_parquet(out_cfg["train_interactions_path"], index=False)
    valid_interactions.to_parquet(out_cfg["valid_interactions_path"], index=False)
    items.to_parquet(out_cfg["items_path"], index=False)
    impressions.to_parquet(out_cfg["impressions_path"], index=False)
    
    logger.info(f"Saved MIND train interactions to {out_cfg['train_interactions_path']}")
    logger.info(f"Saved MIND valid interactions to {out_cfg['valid_interactions_path']}")
    logger.info(f"Saved MIND items to {out_cfg['items_path']}")
    logger.info(f"Saved MIND impressions to {out_cfg['impressions_path']}")
    
    return {
        "train_interactions": train_interactions,
        "valid_interactions": valid_interactions,
        "items": items,
        "impressions": impressions
    }

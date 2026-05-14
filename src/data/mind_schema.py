import pandas as pd
from pathlib import Path

def get_mind_behavior_columns() -> list[str]:
    return [
        "impression_id",
        "user_id",
        "time",
        "history",
        "impressions"
    ]

def get_mind_news_columns() -> list[str]:
    return [
        "item_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities"
    ]

def parse_behaviors_tsv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(
        path,
        sep="\t",
        names=get_mind_behavior_columns(),
        header=None,
    )
    # Don't dropna here, we handle empty strings or nans properly later.
    return df

def parse_news_tsv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    df = pd.read_csv(
        path,
        sep="\t",
        names=get_mind_news_columns(),
        header=None,
    )
    
    # Fill missing string fields to avoid dropping valid news without abstract
    df["abstract"] = df["abstract"].fillna("")
    df["title_entities"] = df["title_entities"].fillna("")
    df["abstract_entities"] = df["abstract_entities"].fillna("")
    
    return df

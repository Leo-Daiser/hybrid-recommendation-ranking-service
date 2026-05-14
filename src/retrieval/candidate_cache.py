import pandas as pd
from pathlib import Path

def save_candidate_cache(
    candidate_cache: pd.DataFrame,
    output_path: str | Path,
) -> None:
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    candidate_cache.to_parquet(out_p, index=False)

def load_candidate_cache(
    path: str | Path,
) -> pd.DataFrame:
    return pd.read_parquet(path)

def validate_candidate_cache(
    candidate_cache: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rank_col: str = "rank",
    score_col: str = "retrieval_score",
) -> None:
    if candidate_cache.empty:
        return
        
    required = [user_col, item_col, rank_col, score_col, "retrieval_model", "generated_at"]
    for col in required:
        if col not in candidate_cache.columns:
            raise ValueError(f"Missing required column: {col}")
            
    if (candidate_cache[rank_col] < 1).any():
        raise ValueError(f"{rank_col} must be >= 1")
        
    if candidate_cache.duplicated(subset=[user_col, item_col]).any():
        raise ValueError("Duplicate user-item pairs found.")
        
    if candidate_cache.duplicated(subset=[user_col, rank_col]).any():
        raise ValueError("Duplicate ranks for the same user found.")
        
    if candidate_cache[score_col].isnull().any():
        raise ValueError(f"{score_col} contains null values.")

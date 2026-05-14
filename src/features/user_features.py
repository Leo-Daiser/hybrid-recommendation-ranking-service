import pandas as pd
import numpy as np

def build_user_features(
    train_interactions: pd.DataFrame,
    user_col: str = "user_id",
    rating_col: str = "rating",
    label_col: str = "label",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    df = train_interactions.copy()
    
    aggs = {
        rating_col: ["count", "mean", "min", "max", "std"],
        label_col: ["sum"],
        timestamp_col: ["min", "max"]
    }
    
    user_stats = df.groupby(user_col).agg(aggs)
    user_stats.columns = [
        "user_rating_count", "user_mean_rating", "user_min_rating", "user_max_rating", "user_std_rating",
        "user_positive_count", "user_first_interaction_ts", "user_last_interaction_ts"
    ]
    
    user_stats["user_std_rating"] = user_stats["user_std_rating"].fillna(0.0)
    
    user_stats["user_positive_ratio"] = user_stats["user_positive_count"] / user_stats["user_rating_count"]
    
    user_stats["user_activity_span"] = user_stats["user_last_interaction_ts"] - user_stats["user_first_interaction_ts"]
    user_stats["user_active_days_approx"] = user_stats["user_activity_span"] / 86400.0
    
    return user_stats.reset_index()

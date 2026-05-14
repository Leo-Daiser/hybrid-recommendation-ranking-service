import pandas as pd

def parse_genres(movies: pd.DataFrame, item_col: str = "movieId", genres_col: str = "genres") -> pd.DataFrame:
    df = movies[[item_col, genres_col]].copy()
    
    df["genre"] = df[genres_col].str.split("|")
    genre_df = df.explode("genre")
    
    # drop the original genres_col
    genre_df = genre_df.drop(columns=[genres_col])
    genre_df["has_genre"] = 1
    
    return genre_df.reset_index(drop=True)

def build_item_features(
    train_interactions: pd.DataFrame,
    movies: pd.DataFrame,
    item_col: str = "item_id",
    original_item_col: str = "movieId",
    rating_col: str = "rating",
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    unknown_genre_token: str = "(no genres listed)",
) -> pd.DataFrame:
    
    df = train_interactions.copy()
    
    aggs = {
        rating_col: ["count", "mean", "min", "max", "std"],
        label_col: ["sum"],
        timestamp_col: ["min", "max"]
    }
    
    item_stats = df.groupby(item_col).agg(aggs)
    item_stats.columns = [
        "item_rating_count", "item_mean_rating", "item_min_rating", "item_max_rating", "item_std_rating",
        "item_positive_count", "item_first_interaction_ts", "item_last_interaction_ts"
    ]
    
    item_stats["item_std_rating"] = item_stats["item_std_rating"].fillna(0.0)
    item_stats["item_positive_ratio"] = item_stats["item_positive_count"] / item_stats["item_rating_count"]
    
    item_stats["item_popularity_rank"] = item_stats["item_rating_count"].rank(method="min", ascending=False)
    
    item_stats = item_stats.reset_index()
    
    # merge with movies
    m_df = movies.copy()
    m_df[item_col] = m_df[original_item_col]
    
    m_df["item_genres"] = m_df["genres"].fillna(unknown_genre_token)
    m_df.loc[m_df["item_genres"] == "", "item_genres"] = unknown_genre_token
    
    m_df["item_genre_count"] = m_df["item_genres"].apply(
        lambda g: 0 if g == unknown_genre_token else len(g.split("|"))
    )
    
    features = pd.merge(
        item_stats,
        m_df[[item_col, "item_genres", "item_genre_count"]],
        on=item_col,
        how="left"
    )
    
    features["item_genres"] = features["item_genres"].fillna(unknown_genre_token)
    features["item_genre_count"] = features["item_genre_count"].fillna(0)
    
    return features

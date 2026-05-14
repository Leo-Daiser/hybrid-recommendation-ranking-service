import yaml
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier

def load_ranker_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_ranker_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    exclude_columns: list[str],
) -> tuple:
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    features = [c for c in numeric_cols if c not in exclude_columns and c != target_col]
    
    X_train = train_df[features].copy()
    y_train = train_df[target_col].copy() if target_col in train_df else pd.Series(dtype=int)
    
    X_valid = valid_df[features].copy()
    y_valid = valid_df[target_col].copy() if target_col in valid_df else pd.Series(dtype=int)
    
    medians = X_train.median()
    medians = medians.fillna(0)
    
    X_train = X_train.fillna(medians)
    X_valid = X_valid.fillna(medians)
    
    fill_values = medians.to_dict()
    
    return X_train, y_train, X_valid, y_valid, features, fill_values

def train_logistic_regression_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int = 42,
):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=random_seed, max_iter=1000)
    )
    if len(np.unique(y_train)) > 1:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, [0]*len(y_train))
        model.classes_ = np.array([0, 1])
    return model

def train_challenger_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_seed: int = 42,
):
    if len(np.unique(y_train)) <= 1:
        model = LogisticRegression(random_state=random_seed)
        model.fit(X_train, [0]*len(y_train))
        model.classes_ = np.array([0, 1])
        return model

    if HAS_CATBOOST:
        model = CatBoostClassifier(
            iterations=100, 
            random_seed=random_seed, 
            verbose=False,
            auto_class_weights="Balanced"
        )
        model.fit(X_train, y_train)
        return model
    else:
        model = HistGradientBoostingClassifier(
            random_state=random_seed,
            max_iter=100,
        )
        model.fit(X_train, y_train)
        return model

def predict_ranker_scores(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    if len(X) == 0:
        return np.array([])
    probas = model.predict_proba(X)
    if probas.shape[1] > 1:
        return probas[:, 1]
    return np.zeros(len(X))

def save_model_bundle(
    model,
    path: str | Path,
    feature_columns: list[str],
    fill_values: dict,
) -> None:
    """Save model as a bundle with feature metadata for runtime scoring."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "fill_values": fill_values,
    }
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

def save_model(model, path: str | Path) -> None:
    """Legacy save — kept for backward compatibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model_bundle(path: str | Path) -> dict:
    """Load a model bundle. Backward-compatible with plain model pickles."""
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    # Legacy format: bare model without metadata
    return {"model": obj, "feature_columns": None, "fill_values": None}

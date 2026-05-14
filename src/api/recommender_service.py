import yaml
import uuid
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from src.ranking.train_ranker import load_model_bundle

logger = logging.getLogger(__name__)

def load_api_config(config_path: str | Path = "configs/api.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_recommender_artifacts(config: dict) -> dict:
    if not config or "recommendation" not in config:
        return {}

    rec_cfg = config["recommendation"]
    artifacts: dict = {}

    def _try_load_parquet(key: str, path_key: str) -> None:
        p = Path(rec_cfg.get(path_key, ""))
        if p.exists():
            artifacts[key] = pd.read_parquet(p)
            logger.info("Loaded %s from %s (shape=%s)", key, p, artifacts[key].shape)
        else:
            logger.warning("Artifact not found: %s (key=%s)", p, path_key)

    _try_load_parquet("candidate_cache", "candidate_cache_path")
    _try_load_parquet("popularity_cache", "popularity_cache_path")
    _try_load_parquet("user_features", "user_features_path")
    _try_load_parquet("item_features", "item_features_path")

    model_path = Path(rec_cfg.get("ranker_model_path", ""))
    if model_path.exists():
        try:
            bundle = load_model_bundle(model_path)
            artifacts["ranker_model"] = bundle["model"]
            artifacts["ranker_feature_columns"] = bundle.get("feature_columns")
            artifacts["ranker_fill_values"] = bundle.get("fill_values")
            logger.info(
                "Loaded ranker model bundle. feature_columns=%s fill_values_keys=%s",
                bundle.get("feature_columns"),
                list(bundle.get("fill_values", {}).keys()) if bundle.get("fill_values") else None,
            )
        except Exception as e:
            logger.error("Failed to load ranker model from %s: %s", model_path, e)
    else:
        logger.warning("Ranker model not found at %s", model_path)

    return artifacts


def get_user_candidates(
    user_id: int,
    candidate_cache: pd.DataFrame | None,
    popularity_cache: pd.DataFrame | None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    k: int = 10,
) -> Tuple[pd.DataFrame, bool]:
    fallback_used = False

    if (
        candidate_cache is not None
        and not candidate_cache.empty
        and user_col in candidate_cache.columns
    ):
        user_cands = candidate_cache[candidate_cache[user_col] == user_id]
        if not user_cands.empty:
            return user_cands.head(k * 2).copy(), fallback_used

    fallback_used = True
    if popularity_cache is not None and not popularity_cache.empty:
        pop = popularity_cache.head(k * 2).copy()
        pop[user_col] = user_id
        return pop, fallback_used

    return pd.DataFrame(), fallback_used


def _assemble_runtime_features(
    candidates: pd.DataFrame,
    user_features: pd.DataFrame | None,
    item_features: pd.DataFrame | None,
    feature_columns: list[str] | None,
    fill_values: dict | None,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build (enriched_df, X, feature_cols) by joining user/item features onto candidates
    and assembling a numeric feature matrix that matches training schema.

    Returns:
        enriched_df: candidates joined with user/item features + derived retrieval cols
        X:           ready-to-score numeric DataFrame
        used_cols:   feature column names actually used
    """
    df = candidates.copy()

    # --- Derived retrieval features (mirror ranking/dataset.py) ---
    if "retrieval_score" in df.columns:
        df["max_retrieval_score"] = df["retrieval_score"]
    if "rank" in df.columns:
        df["min_rank"] = df["rank"]
    df["came_from_itemknn"] = 1
    df["came_from_popularity"] = 0
    df["retrieval_model_count"] = 1

    # --- Left-join user features ---
    if user_features is not None and not user_features.empty and user_col in user_features.columns:
        df = pd.merge(df, user_features, on=user_col, how="left")

    # --- Left-join item features ---
    if item_features is not None and not item_features.empty and item_col in item_features.columns:
        df = pd.merge(df, item_features, on=item_col, how="left")

    # --- Determine feature columns ---
    if feature_columns:
        # Strict: use exactly the training columns
        missing_cols = [c for c in feature_columns if c not in df.columns]
        for c in missing_cols:
            df[c] = np.nan
        X = df[feature_columns].copy()
        used_cols = feature_columns
    else:
        # Fallback: all numeric columns except known ID / meta columns
        _exclude = {
            user_col, item_col, "target", "generated_at",
            "retrieval_model", "item_genres", "rank", "retrieval_score",
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        used_cols = [c for c in numeric_cols if c not in _exclude]
        X = df[used_cols].copy()

    # --- Fill NaN ---
    if fill_values:
        for col in X.columns:
            if col in fill_values:
                X[col] = X[col].fillna(fill_values[col])
    X = X.fillna(0.0)

    return df, X, used_cols


def score_candidates_with_ranker(
    candidates: pd.DataFrame,
    ranker_model,
    user_features: pd.DataFrame | None,
    item_features: pd.DataFrame | None,
    feature_columns: list[str] | None,
    fill_values: dict | None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    score_col: str = "ranking_score",
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Returns (scored_df, warnings).
    scored_df is empty if scoring failed.
    """
    warnings: list[str] = []

    if ranker_model is None:
        warnings.append("Ranker model is not loaded.")
        return pd.DataFrame(), warnings

    if candidates.empty:
        warnings.append("No candidates to score.")
        return pd.DataFrame(), warnings

    # Check user/item feature availability
    if user_features is None or user_features.empty:
        warnings.append("user_features not available; NaN fill will be used for user columns.")
    if item_features is None or item_features.empty:
        warnings.append("item_features not available; NaN fill will be used for item columns.")

    enriched_df, X, used_cols = _assemble_runtime_features(
        candidates=candidates,
        user_features=user_features,
        item_features=item_features,
        feature_columns=feature_columns,
        fill_values=fill_values,
        user_col=user_col,
        item_col=item_col,
    )

    if len(X) == 0 or len(X.columns) == 0:
        warnings.append("Feature matrix is empty after assembly.")
        return pd.DataFrame(), warnings

    logger.debug(
        "Scoring %d candidates with %d features: %s",
        len(X), len(X.columns), list(X.columns)[:10],
    )

    try:
        if hasattr(ranker_model, "predict_proba"):
            probas = ranker_model.predict_proba(X)
            scores = probas[:, 1] if probas.shape[1] > 1 else np.zeros(len(X))
        elif hasattr(ranker_model, "decision_function"):
            scores = ranker_model.decision_function(X)
        else:
            scores = ranker_model.predict(X).astype(float)

        enriched_df = enriched_df.copy()
        enriched_df[score_col] = scores
        return enriched_df, warnings

    except Exception as e:
        msg = (
            f"Ranker predict_proba failed: {type(e).__name__}: {e}. "
            f"X shape={X.shape}, columns={list(X.columns)}, "
            f"expected feature_columns={feature_columns}."
        )
        logger.error(msg)
        warnings.append(msg)
        return pd.DataFrame(), warnings


def build_recommendation_response(
    user_id: int,
    ranked_candidates: pd.DataFrame,
    k: int,
    model_version: str,
    fallback_used: bool,
    scoring_mode: str = "retrieval_only",
    warnings: list[str] | None = None,
) -> dict:
    items = []
    if not ranked_candidates.empty:
        top_k = ranked_candidates.head(k)
        for _, row in top_k.iterrows():
            item_id = int(row.get("item_id", 0))
            score = float(row.get("ranking_score", row.get("retrieval_score", 0.0)))
            retrieval_score = (
                float(row["retrieval_score"]) if "retrieval_score" in row.index else None
            )
            items.append({
                "item_id": item_id,
                "score": score,
                "rank": len(items) + 1,
                "retrieval_score": retrieval_score,
                "explanation": {"source": "popularity" if fallback_used else "itemknn"},
            })

    return {
        "request_id": str(uuid.uuid4()),
        "user_id": user_id,
        "items": items,
        "model_version": model_version,
        "fallback_used": fallback_used,
        "scoring_mode": scoring_mode,
        "warnings": warnings or [],
    }


def recommend_for_user(
    user_id: int,
    k: int,
    artifacts: dict,
    config: dict,
) -> dict:
    if not config or "recommendation" not in config:
        raise ValueError("Invalid configuration")

    rec_cfg = config["recommendation"]
    max_k = rec_cfg.get("max_k", 50)
    k = min(k, max_k)

    user_col = rec_cfg.get("user_id_column", "user_id")
    item_col = rec_cfg.get("item_id_column", "item_id")
    score_col = rec_cfg.get("score_column", "ranking_score")
    fallback_col = rec_cfg.get("fallback_score_column", "retrieval_score")
    model_version = rec_cfg.get("model_version", "unknown")

    cand_cache = artifacts.get("candidate_cache")
    pop_cache = artifacts.get("popularity_cache")

    cands, fallback_used = get_user_candidates(
        user_id=user_id,
        candidate_cache=cand_cache,
        popularity_cache=pop_cache,
        user_col=user_col,
        item_col=item_col,
        k=k,
    )

    if cands.empty:
        return build_recommendation_response(
            user_id, cands, k, model_version, fallback_used,
            warnings=["No candidates available."],
        )

    scored, scoring_warnings = score_candidates_with_ranker(
        candidates=cands,
        ranker_model=artifacts.get("ranker_model"),
        user_features=artifacts.get("user_features"),
        item_features=artifacts.get("item_features"),
        feature_columns=artifacts.get("ranker_feature_columns"),
        fill_values=artifacts.get("ranker_fill_values"),
        user_col=user_col,
        item_col=item_col,
        score_col=score_col,
    )

    if not scored.empty:
        scored = scored.sort_values(by=score_col, ascending=False)
        return build_recommendation_response(
            user_id, scored, k, model_version, fallback_used,
            scoring_mode="ranker",
            warnings=scoring_warnings,
        )
    else:
        # Retrieval-only fallback
        if fallback_col in cands.columns:
            cands = cands.sort_values(by=fallback_col, ascending=False)
        return build_recommendation_response(
            user_id, cands, k, model_version, fallback_used,
            scoring_mode="retrieval_only",
            warnings=scoring_warnings,
        )

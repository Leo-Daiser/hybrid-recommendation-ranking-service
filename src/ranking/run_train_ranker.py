import pandas as pd
from pathlib import Path
from src.ranking.train_ranker import (
    load_ranker_config,
    prepare_ranker_features,
    train_logistic_regression_ranker,
    train_challenger_ranker,
    predict_ranker_scores,
    save_model_bundle
)
from src.ranking.evaluate_ranker import (
    attach_scores,
    rank_by_model_score,
    evaluate_ranked_candidates
)
from src.ranking.reports import save_ranking_metrics, save_ranking_report

def run_train_ranker(
    ranker_config_path: str | Path = "configs/ranker.yaml",
) -> pd.DataFrame:
    cfg = load_ranker_config(ranker_config_path)["ranker"]
    
    train_df = pd.read_parquet(cfg["input"]["train_path"])
    valid_df = pd.read_parquet(cfg["input"]["valid_path"])
    
    target_col = cfg["columns"]["target_column"]
    user_col = cfg["columns"]["user_id_column"]
    item_col = cfg["columns"]["item_id_column"]
    
    target_ratio_train = (train_df[target_col] == 1).mean() if not train_df.empty else 0.0
    target_ratio_valid = (valid_df[target_col] == 1).mean() if not valid_df.empty else 0.0
    
    X_train, y_train, X_valid, y_valid, features, fill_values = prepare_ranker_features(
        train_df=train_df,
        valid_df=valid_df,
        target_col=target_col,
        exclude_columns=cfg["features"]["exclude_columns"]
    )
    
    seed = cfg["training"]["random_seed"]
    
    logreg = train_logistic_regression_ranker(X_train, y_train, random_seed=seed)
    challenger = train_challenger_ranker(X_train, y_train, random_seed=seed)
    
    lr_scores = predict_ranker_scores(logreg, X_valid)
    valid_lr = attach_scores(valid_df, lr_scores, "ranking_score")
    valid_lr = rank_by_model_score(valid_lr, user_col, "ranking_score")
    
    ch_scores = predict_ranker_scores(challenger, X_valid)
    valid_ch = attach_scores(valid_df, ch_scores, "ranking_score")
    valid_ch = rank_by_model_score(valid_ch, user_col, "ranking_score")
    
    k_vals = cfg["evaluation"]["k_values"]
    
    lr_metrics = evaluate_ranked_candidates(
        scored_df=valid_lr,
        k_values=k_vals,
        user_col=user_col,
        item_col=item_col,
        target_col=target_col,
        rank_col="model_rank"
    )
    if not lr_metrics.empty:
        lr_metrics.insert(0, "model_name", cfg["training"]["baseline_model"])
        
    ch_metrics = evaluate_ranked_candidates(
        scored_df=valid_ch,
        k_values=k_vals,
        user_col=user_col,
        item_col=item_col,
        target_col=target_col,
        rank_col="model_rank"
    )
    if not ch_metrics.empty:
        ch_metrics.insert(0, "model_name", cfg["training"]["challenger_model"])
        
    all_metrics = pd.concat([lr_metrics, ch_metrics], ignore_index=True)
    
    save_model_bundle(logreg, cfg["output"]["logreg_model_path"], features, fill_values)
    save_model_bundle(challenger, cfg["output"]["challenger_model_path"], features, fill_values)
    
    save_ranking_metrics(
        metrics=all_metrics,
        json_path=cfg["output"]["metrics_json_path"],
        csv_path=cfg["output"]["metrics_csv_path"],
    )
    
    save_ranking_report(
        metrics=all_metrics,
        report_path=cfg["output"]["report_path"],
        train_shape=train_df.shape,
        valid_shape=valid_df.shape,
        target_ratio_train=target_ratio_train,
        target_ratio_valid=target_ratio_valid
    )
    
    return all_metrics

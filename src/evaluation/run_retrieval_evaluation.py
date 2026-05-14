import pandas as pd
import yaml
from pathlib import Path
from src.evaluation.offline_evaluator import build_ground_truth, compare_retrieval_models
from src.evaluation.reports import save_evaluation_results, save_retrieval_report

def load_evaluation_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_retrieval_evaluation(
    evaluation_config_path: str | Path = "configs/evaluation.yaml",
) -> pd.DataFrame:
    cfg = load_evaluation_config(evaluation_config_path)["evaluation"]
    
    valid_path = Path(cfg["interactions"]["valid_path"])
    item_feat_path = Path(cfg["item_features_path"])
    
    valid_interactions = pd.read_parquet(valid_path)
    item_features = pd.read_parquet(item_feat_path)
    total_items = item_features[cfg["item_id_column"]].nunique()
    
    ground_truth = build_ground_truth(
        interactions=valid_interactions,
        user_col=cfg["user_id_column"],
        item_col=cfg["item_id_column"],
        label_col=cfg["label_column"],
        positive_label=cfg["positive_label"]
    )
    
    caches = {}
    for model_name, cache_path in cfg["candidate_caches"].items():
        caches[model_name] = pd.read_parquet(Path(cache_path))
        
    results = compare_retrieval_models(
        candidate_caches=caches,
        ground_truth=ground_truth,
        total_items=total_items,
        k_values=cfg["k_values"],
        exclude_users_without_ground_truth=cfg["behavior"]["exclude_users_without_ground_truth"]
    )
    
    save_evaluation_results(
        results,
        json_path=cfg["output"]["valid_json_path"],
        csv_path=cfg["output"]["valid_csv_path"]
    )
    
    save_retrieval_report(
        results,
        report_path=cfg["output"]["valid_report_path"]
    )
    
    return results

import json
import pandas as pd
from pathlib import Path

def save_ranking_metrics(
    metrics: pd.DataFrame,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics.to_csv(csv_path, index=False)
    
    records = metrics.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

def build_ranking_report_markdown(
    metrics: pd.DataFrame,
    train_shape: tuple,
    valid_shape: tuple,
    target_ratio_train: float,
    target_ratio_valid: float,
) -> str:
    lines = [
        "# Ranking Model Evaluation Report",
        "",
        "## Dataset Statistics",
        f"- **Train Shape**: {train_shape[0]} rows, {train_shape[1]} columns",
        f"- **Valid Shape**: {valid_shape[0]} rows, {valid_shape[1]} columns",
        f"- **Train Target Ratio**: {target_ratio_train:.4f}",
        f"- **Valid Target Ratio**: {target_ratio_valid:.4f}",
        ""
    ]
    
    if train_shape[0] < 1000:
        lines.append("> [!WARNING]")
        lines.append("> **Small Dataset**")
        lines.append("> The ranking dataset has fewer than 1000 rows. The model quality evaluation might not be robust or representative of large-scale performance. Please treat this as a pipeline demonstration.")
        lines.append("")
        
    lines.extend([
        "## Metrics Comparison",
        ""
    ])
    
    if metrics.empty:
        lines.append("_No metrics available._")
    else:
        cols = list(metrics.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        
        for _, row in metrics.iterrows():
            row_str = " | ".join(str(row[c]) if not isinstance(row[c], float) else f"{row[c]:.4f}" for c in cols)
            lines.append(f"| {row_str} |")
            
    lines.append("")
    
    if not metrics.empty and "ndcg" in metrics.columns and "k" in metrics.columns:
        m10 = metrics[metrics["k"] == 10]
        if not m10.empty:
            best_idx = m10["ndcg"].idxmax()
            best_model = m10.loc[best_idx, "model_name"]
            best_ndcg = m10.loc[best_idx, "ndcg"]
            lines.append(f"**Best Model by NDCG@10**: {best_model} ({best_ndcg:.4f})")
            
    return "\n".join(lines)

def save_ranking_report(
    metrics: pd.DataFrame,
    report_path: str | Path,
    train_shape: tuple,
    valid_shape: tuple,
    target_ratio_train: float,
    target_ratio_valid: float,
) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_str = build_ranking_report_markdown(
        metrics, train_shape, valid_shape, target_ratio_train, target_ratio_valid
    )
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_str)

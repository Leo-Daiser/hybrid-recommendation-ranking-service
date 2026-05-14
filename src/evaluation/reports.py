import pandas as pd
import json
from pathlib import Path

def save_evaluation_results(
    results: pd.DataFrame,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    jp = Path(json_path)
    cp = Path(csv_path)
    
    jp.parent.mkdir(parents=True, exist_ok=True)
    cp.parent.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(cp, index=False)
    
    res_dict = results.to_dict(orient="records")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)

def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No results available._"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)

def build_retrieval_report_markdown(
    results: pd.DataFrame,
    dataset_name: str = "MovieLens ml-latest-small",
) -> str:
    models = results["model_name"].unique().tolist()
    k_vals = sorted(results["k"].unique().tolist())
    
    table = results.copy()
    for col in ["precision", "recall", "map", "ndcg", "coverage"]:
        if col in table.columns:
            table[col] = table[col].apply(lambda x: f"{x:.4f}")
            
    table_md = dataframe_to_markdown_table(table)
    
    k10 = results[results["k"] == 10]
    best_ndcg = "N/A"
    best_map = "N/A"
    if not k10.empty:
        best_ndcg = k10.loc[k10["ndcg"].idxmax()]["model_name"]
        best_map = k10.loc[k10["map"].idxmax()]["model_name"]
        
    md = f"""# Retrieval Evaluation Report

**Dataset**: {dataset_name}
**Models compared**: {', '.join(models)}
**K values**: {', '.join(map(str, k_vals))}

## Metrics

{table_md}

## Summary
- **Best model by NDCG@10**: {best_ndcg}
- **Best model by MAP@10**: {best_map}

## Known Limitations
- Only offline retrieval evaluation is performed.
- No ranking layer yet.
- No online feedback.
- No A/B testing.
"""
    return md

def save_retrieval_report(
    results: pd.DataFrame,
    report_path: str | Path,
) -> None:
    rp = Path(report_path)
    rp.parent.mkdir(parents=True, exist_ok=True)
    md = build_retrieval_report_markdown(results)
    with open(rp, "w", encoding="utf-8") as f:
        f.write(md)

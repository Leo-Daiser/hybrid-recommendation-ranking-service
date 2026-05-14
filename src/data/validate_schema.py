import pandas as pd

def validate_required_columns(table_name: str, df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Table '{table_name}' is missing required columns: {missing}")

def validate_non_empty(table_name: str, df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError(f"Table '{table_name}' is empty.")

def validate_unique_key(table_name: str, df: pd.DataFrame, unique_key: list[str]) -> None:
    if df.duplicated(subset=unique_key).any():
        raise ValueError(f"Table '{table_name}' has duplicate entries for key: {unique_key}")

def validate_foreign_key_relationship(
    child_df: pd.DataFrame, child_column: str,
    parent_df: pd.DataFrame, parent_column: str,
    relationship_name: str, strict: bool = True
) -> dict:
    orphans = child_df[~child_df[child_column].isin(parent_df[parent_column])]
    orphan_count = len(orphans)
    
    if orphan_count > 0 and strict:
        raise ValueError(f"Strict FK violation in {relationship_name}: {orphan_count} orphans found.")
        
    return {
        "orphan_count": orphan_count,
        "orphan_ratio": orphan_count / len(child_df) if len(child_df) > 0 else 0.0,
        "sample_orphans": orphans[child_column].head(5).tolist() if orphan_count > 0 else []
    }

def validate_raw_tables(tables: dict[str, pd.DataFrame], config: dict, strict_foreign_keys: bool | None = None) -> dict:
    report = {"tables": {}, "foreign_keys": {}, "warnings": []}
    
    for table_name, table_config in config.get("tables", {}).items():
        if table_name not in tables:
            continue
            
        df = tables[table_name]
        req_cols = table_config.get("required_columns", [])
        
        try:
            if table_name != "tags":
                validate_non_empty(table_name, df)
            elif df.empty:
                report["warnings"].append("tags table is empty")
                
            validate_required_columns(table_name, df, req_cols)
            
            if "unique_key" in table_config:
                validate_unique_key(table_name, df, table_config["unique_key"])
                
            if table_name == "ratings":
                if "rating" in df.columns and not pd.api.types.is_numeric_dtype(df["rating"]):
                    raise ValueError("ratings.rating must be numeric")
                if "timestamp" in df.columns and not pd.api.types.is_numeric_dtype(df["timestamp"]):
                    raise ValueError("ratings.timestamp must be numeric")
                    
                if "rating" in df.columns:
                    out_of_bounds = df[(df["rating"] < 0.5) | (df["rating"] > 5.0)]
                    if not out_of_bounds.empty:
                        report["warnings"].append(f"ratings table has {len(out_of_bounds)} ratings out of [0.5, 5.0] bounds.")
                        
            report["tables"][table_name] = {
                "rows": len(df),
                "columns": list(df.columns)
            }
            
        except ValueError as e:
            report["tables"][table_name] = {"error": str(e)}
            raise e

    if strict_foreign_keys is None:
        strict_foreign_keys = config.get("validation", {}).get("strict_foreign_keys", False)
        
    movies_df = tables.get("movies")
    
    if movies_df is not None:
        for t_name in ["ratings", "tags", "links"]:
            if t_name in tables:
                rel_name = f"{t_name}.movieId -> movies.movieId"
                fk_report = validate_foreign_key_relationship(
                    child_df=tables[t_name], child_column="movieId",
                    parent_df=movies_df, parent_column="movieId",
                    relationship_name=rel_name, strict=strict_foreign_keys
                )
                report["foreign_keys"][rel_name] = fk_report
                
    return report

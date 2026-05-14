import typer
from pathlib import Path
from src.db.init_db import init_db as init_database
from src.data.download import download_movielens, load_data_config, expected_raw_files
from src.data.load_raw import load_raw_tables
from src.data.validate_schema import validate_raw_tables
from src.data.prepare_interactions import run_prepare_interactions

app = typer.Typer()

@app.command()
def init_db():
    """Initialize the database schema."""
    typer.echo("Creating database tables...")
    init_database()
    typer.echo("Database tables created successfully.")

@app.command()
def download_data(
    config: Path = typer.Option(
        Path("configs/data.yaml"),
        "--config",
        help="Path to data config."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-download of the dataset."
    )
):
    """Download and extract raw dataset."""
    typer.echo("Downloading data...")
    try:
        raw_dir = download_movielens(config_path=config, force=force)
        typer.echo(f"Data ready at: {raw_dir}")
        cfg = load_data_config(config)
        expected = expected_raw_files(cfg)
        for f in expected:
            if f.exists():
                typer.echo(f"Found: {f.name}")
            else:
                typer.echo(f"Missing: {f.name}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def validate_raw(
    config: Path = typer.Option(
        Path("configs/data.yaml"),
        "--config",
        help="Path to data config."
    )
):
    """Validate raw data schema."""
    typer.echo("Validating raw data...")
    try:
        cfg = load_data_config(config)
        tables = load_raw_tables(config)
        
        for name, df in tables.items():
            typer.echo(f"Loaded {name}: {df.shape}")
            
        report = validate_raw_tables(tables, cfg)
        
        typer.echo("\n--- Validation Report ---")
        typer.echo("Foreign Keys:")
        for rel, rel_data in report.get("foreign_keys", {}).items():
            typer.echo(f"  {rel}: {rel_data['orphan_count']} orphans")
            
        if report.get("warnings"):
            typer.echo("Warnings:")
            for w in report["warnings"]:
                typer.echo(f"  - {w}")
                
        typer.echo("Validation complete.")
    except Exception as e:
        typer.echo(f"Validation Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def prepare_interactions(
    data_config: Path = typer.Option(Path("configs/data.yaml"), "--data-config", help="Path to data config."),
    interactions_config: Path = typer.Option(Path("configs/interactions.yaml"), "--interactions-config", help="Path to interactions config.")
):
    """Prepare implicit interactions and perform temporal split."""
    typer.echo("Preparing interactions...")
    try:
        splits = run_prepare_interactions(data_config_path=data_config, interactions_config_path=interactions_config)
        full = splits["full"]
        train = splits["train"]
        valid = splits["valid"]
        test = splits["test"]
        
        pos_ratio = full['label'].mean()
        
        typer.echo(f"Full interactions: shape={full.shape}")
        typer.echo(f"Train: shape={train.shape}")
        typer.echo(f"Valid: shape={valid.shape}")
        typer.echo(f"Test: shape={test.shape}")
        typer.echo(f"Positive ratio full: {pos_ratio:.4f}")
        typer.echo("Temporal boundaries:")
        typer.echo(f"  train max timestamp: {train['timestamp'].max()}")
        typer.echo(f"  valid min/max timestamp: {valid['timestamp'].min()} / {valid['timestamp'].max()}")
        typer.echo(f"  test min timestamp: {test['timestamp'].min()}")
        typer.echo("Saved to data/processed/")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def build_features(
    data_config: Path = typer.Option(Path("configs/data.yaml"), "--data-config", help="Path to data config."),
    features_config: Path = typer.Option(Path("configs/features.yaml"), "--features-config", help="Path to features config.")
):
    """Build user, item and genre features."""
    typer.echo("Building features...")
    try:
        from src.features.build_features import run_build_features
        res = run_build_features(data_config_path=data_config, feature_config_path=features_config)
        
        typer.echo(f"User features: shape={res['user_features'].shape}")
        typer.echo(f"Item features: shape={res['item_features'].shape}")
        typer.echo(f"Genre features: shape={res['genre_features'].shape}")
        typer.echo("Saved to data/processed/")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def build_popularity_candidates(
    retrieval_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--retrieval-config", help="Path to retrieval config.")
):
    """Build popularity retrieval baseline candidates."""
    typer.echo("Building popularity candidates...")
    try:
        from src.retrieval.build_candidates import run_build_popularity_candidates
        res = run_build_popularity_candidates(retrieval_config_path=retrieval_config)
        
        users_covered = res["user_id"].nunique()
        items_rec = res["item_id"].nunique()
        avg_cands = len(res) / users_covered if users_covered > 0 else 0
        
        typer.echo(f"Candidate cache: shape={res.shape}")
        typer.echo(f"Users covered: {users_covered}")
        typer.echo(f"Unique items recommended: {items_rec}")
        typer.echo(f"Average candidates per user: {avg_cands:.1f}")
        typer.echo("Saved to data/processed/candidate_cache_popularity.parquet")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def build_itemknn_candidates(
    retrieval_config: Path = typer.Option(Path("configs/retrieval.yaml"), "--retrieval-config", help="Path to retrieval config.")
):
    """Build item-item similarity candidates."""
    typer.echo("Building item-item similarity candidates...")
    try:
        from src.retrieval.build_itemknn_candidates import run_build_itemknn_candidates
        res = run_build_itemknn_candidates(retrieval_config_path=retrieval_config)
        
        sim_df = res["item_similarity"]
        cache_df = res["candidate_cache"]
        
        users_covered = cache_df["user_id"].nunique()
        items_rec = cache_df["item_id"].nunique()
        avg_cands = len(cache_df) / users_covered if users_covered > 0 else 0
        
        typer.echo(f"Item similarity: shape={sim_df.shape}")
        typer.echo(f"Candidate cache: shape={cache_df.shape}")
        typer.echo(f"Users covered: {users_covered}")
        typer.echo(f"Unique items recommended: {items_rec}")
        typer.echo(f"Average candidates per user: {avg_cands:.1f}")
        typer.echo("Saved similarity to data/processed/item_similarity_topk.parquet")
        typer.echo("Saved candidates to data/processed/candidate_cache_itemknn.parquet")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def evaluate_retrieval(
    evaluation_config: Path = typer.Option(Path("configs/evaluation.yaml"), "--evaluation-config", help="Path to evaluation config.")
):
    """Evaluate retrieval models offline."""
    typer.echo("Evaluating retrieval models...")
    try:
        from src.evaluation.run_retrieval_evaluation import run_retrieval_evaluation
        res = run_retrieval_evaluation(evaluation_config_path=evaluation_config)
        
        typer.echo("Results:")
        
        # Format the output table cleanly
        summary = res[["model_name", "k", "precision", "recall", "map", "ndcg", "coverage"]].copy()
        for col in ["precision", "recall", "map", "ndcg", "coverage"]:
            summary[col] = summary[col].apply(lambda x: f"{x:.4f}")
            
        typer.echo(summary.to_string(index=False))
        typer.echo("Saved metrics to artifacts/metrics/retrieval_evaluation_valid.json")
        typer.echo("Saved report to artifacts/reports/retrieval_evaluation_report.md")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def build_ranking_dataset(
    ranking_config: Path = typer.Option(Path("configs/ranking.yaml"), "--ranking-config", help="Path to ranking config.")
):
    """Build dataset for ranking models."""
    typer.echo("Building ranking datasets...")
    try:
        from src.ranking.build_ranking_dataset import run_build_ranking_datasets
        res = run_build_ranking_datasets(ranking_config_path=ranking_config)
        
        rt = res["ranking_train"]
        rv = res["ranking_valid"]
        
        rt_pos = (rt["target"] == 1).mean() if not rt.empty else 0
        rv_pos = (rv["target"] == 1).mean() if not rv.empty else 0
        
        typer.echo(f"Ranking train: shape={rt.shape}")
        typer.echo(f"Ranking valid: shape={rv.shape}")
        typer.echo(f"Train positive ratio: {rt_pos:.4f}")
        typer.echo(f"Valid positive ratio: {rv_pos:.4f}")
        typer.echo("Saved to data/processed/ranking_train.parquet")
        typer.echo("Saved to data/processed/ranking_valid.parquet")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def train_ranker(
    ranker_config: Path = typer.Option(Path("configs/ranker.yaml"), "--ranker-config", help="Path to ranker config.")
):
    """Train ranking model."""
    typer.echo("Training ranking models...")
    try:
        from src.ranking.run_train_ranker import run_train_ranker
        from src.ranking.train_ranker import load_ranker_config
        cfg = load_ranker_config(ranker_config)["ranker"]
        
        metrics = run_train_ranker(ranker_config_path=ranker_config)
        
        if not metrics.empty:
            typer.echo("Results:")
            typer.echo(metrics.to_string(index=False))
            
        typer.echo(f"Saved models to {Path(cfg['output']['logreg_model_path']).parent}/")
        typer.echo(f"Saved metrics to {cfg['output']['metrics_json_path']}")
        typer.echo(f"Saved report to {cfg['output']['report_path']}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def db_check():
    """Verify database connection is alive."""
    typer.echo("Checking DB connection...")
    try:
        from sqlalchemy import text
        from src.db.session import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        typer.echo("DB connection OK")
    except Exception as e:
        typer.echo(f"DB connection FAILED: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def run_full_pipeline(
    config: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Path to pipeline config."),
    skip_existing: bool = typer.Option(False, "--skip-existing", help="Skip steps if outputs already exist.")
):
    """Run full ML pipeline sequentially."""
    typer.echo("Running full pipeline...")
    try:
        from src.jobs.pipeline import run_full_pipeline as run_pipeline
        res = run_pipeline(pipeline_config_path=config, skip_existing_outputs=skip_existing)
        
        for i, step in enumerate(res["steps"]):
            status = step["status"]
            step_name = step["step_name"]
            typer.echo(f"[{i+1}/{len(res['steps'])}] {step_name}: {status}")
            if status == "failed":
                typer.echo(f"Error in {step_name}: {step.get('error')}", err=True)
        
        if res["status"] == "success":
            typer.echo("Pipeline completed successfully.")
        else:
            typer.echo(f"Pipeline failed at step: {res['failed_step']}", err=True)
            raise typer.Exit(code=1)
            
        typer.echo(f"Saved metadata to {res.get('latest_status_path', 'artifacts/pipeline_runs/latest_status.json')}")
    except Exception as e:
        typer.echo(f"Pipeline execution error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def pipeline_status(
    config: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Path to pipeline config.")
):
    """Check pipeline status and missing outputs."""
    try:
        from src.jobs.pipeline import check_pipeline_status
        status = check_pipeline_status(pipeline_config_path=config)
        
        typer.echo(f"Pipeline: {status['pipeline_name']}")
        typer.echo(f"{'Step Name':<30} | {'Status':<12} | {'Outputs'}")
        typer.echo("-" * 60)
        
        for step in status["steps"]:
            outputs_str = "Present" if step["outputs_present"] else f"Missing: {len(step['missing_outputs'])}"
            typer.echo(f"{step['step_name']:<30} | {step['status']:<12} | {outputs_str}")
            if step["missing_outputs"]:
                for m in step["missing_outputs"][:3]: # Show only first 3
                    typer.echo(f"  - {m}")
                if len(step["missing_outputs"]) > 3:
                    typer.echo(f"  ... and {len(step['missing_outputs']) - 3} more")
                    
    except Exception as e:
        typer.echo(f"Error checking status: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def prepare_mind(
    config: Path = typer.Option(Path("configs/mind.yaml"), "--config", help="Path to MIND config.")
):
    """Prepare MIND dataset for the hybrid recommendation system."""
    typer.echo("Preparing MIND dataset...")
    try:
        from src.data.prepare_mind import run_prepare_mind
        res = run_prepare_mind(mind_config_path=config)
        
        train_interactions = res["train_interactions"]
        valid_interactions = res["valid_interactions"]
        items = res["items"]
        impressions = res["impressions"]
        
        typer.echo(f"Train interactions: shape={train_interactions.shape}")
        typer.echo(f"Valid interactions: shape={valid_interactions.shape}")
        typer.echo(f"Items: shape={items.shape}")
        typer.echo(f"Impressions: shape={impressions.shape}")
        
        import yaml
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)["mind"]
        out_dir = Path(cfg["processed_data_dir"])
        typer.echo(f"Saved to {out_dir}/")
        
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error preparing MIND dataset: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()

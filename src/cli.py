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

if __name__ == "__main__":
    app()

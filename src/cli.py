import typer
from src.db.init_db import init_db as init_database

app = typer.Typer()

@app.command()
def init_db():
    """Initialize the database schema."""
    typer.echo("Creating database tables...")
    init_database()
    typer.echo("Database tables created successfully.")

if __name__ == "__main__":
    app()

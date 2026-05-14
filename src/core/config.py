import os
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_user: str = Field(default="recsys", env="DB_USER")
    db_password: str = Field(default="recsys_pass", env="DB_PASSWORD")
    db_name: str = Field(default="recsys_db", env="DB_NAME")
    
    app_name: str = "Hybrid Recommendation API"
    version: str = "0.1.0"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

def load_yaml_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

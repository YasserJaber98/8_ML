from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:8000"
    model_uri: str = "models:/churn_predictor/latest"
    model_version: str = "1.0.0"
    model_updated: str = "2024-01-01"
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/churn_db"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"

settings = Settings()
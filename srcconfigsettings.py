"""
Configuration management for AGTA system using Pydantic settings.
Provides type-safe configuration with environment variable parsing.
"""
import os
from typing import Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseSettings, Field, validator
from pydantic.types import SecretStr


class Settings(BaseSettings):
    """Main configuration settings for AGTA system."""
    
    # API Configuration
    alpha_vantage_api_key: Optional[SecretStr] = None
    finnhub_api_key: Optional[SecretStr] = None
    newsapi_key: Optional[SecretStr] = None
    
    # Firebase Configuration
    firebase_project_id: str = Field(default="agta-dev")
    firebase_credentials_path: Path = Field(default="./serviceAccountKey.json")
    
    # Trading Configuration
    risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.05)
    max_position_size: float = Field(default=0.1, ge=0.01, le=0.5)
    default_timeframe: str = Field(default="1h", regex=r"^\d+[mhdwM]$")
    supported_timeframes: List[str] = Field(default=["1m", "5m", "15m", "1h", "4h", "1d"])
    
    # Model Configuration
    transformer_model_name: str = Field(default="distilbert-base-uncased")
    embedding_dim: int = Field(default=768)
    sequence_length: int = Field(default=256)
    batch_size: int = Field(default=32)
    
    # RL Configuration
    rl_learning_rate: float = Field(default=0.0003, ge=1e-6, le=0.01)
    rl_gamma: float = Field(default=0.99, ge=0.9, le=0.999)
    rl_buffer_size: int = Field(default=10000)
    
    # Data Configuration
    max_data_points: int = Field(default=10000)
    data_retention_days: int = Field(default=30)
    
    # Logging Configuration
    log_level: str = Field(default="INFO", regex=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file_path: Path = Field(default="./logs/agta.log")
    
    # System Configuration
    max_workers: int = Field(default=4, ge=1, le=16)
    cache_ttl: int = Field(default=300, ge=60, le=3600)
    
    @validator("firebase_credentials_path")
    def validate_firebase_creds(cls, v: Path) -> Path:
        """Validate Firebase credentials path exists."""
        if not v.exists():
            raise ValueError(f"Firebase credentials file not found: {v}")
        return v
    
    @validator("log_file_path")
    def create_log_directory(cls, v: Path) -> Path:
        """Ensure log directory exists."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
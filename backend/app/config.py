"""
Backend Configuration for MLOps Events FastAPI Application

This module contains configuration settings using Pydantic Settings
for environment variable management.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # External API Keys
    GOOGLE_API_KEY: str
    APERTUREDB_KEY: str
    HF_TOKEN: str
    TL_API_KEY: Optional[str] = None

    # API Security
    API_SECRET_KEY: str = "your-secret-key-change-in-production"
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://localhost:3000"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Model Configuration
    MODEL_PROVIDER: str = "google-genai"
    MODEL_NAME: str = "gemini-2.5-pro"
    MODEL_TEMPERATURE: float = 0.7

    # Agent Configuration
    MAX_ITERATIONS: int = 20
    ENABLE_VERBOSE_OUTPUT: bool = True

    # Tool Configuration
    EMBED_MODEL: str = "google/embeddinggemma-300m"
    EMBED_DIM: int = 768
    USE_HF_API_EMBEDDINGS: bool = False

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    def get_allowed_origins_list(self) -> list[str]:
        """Parse ALLOWED_ORIGINS string into list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]


# Global settings instance
settings = Settings()

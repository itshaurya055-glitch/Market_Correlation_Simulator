"""
EPC Intelligence Core — Application Configuration

Loads all settings from .env using Pydantic Settings.
Provides a singleton `get_settings()` for app-wide access.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Groq LLM ---
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Paths ---
    chroma_db_path: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"
    standards_dir: str = "./data/standards"
    test_records_dir: str = "./data/test_records"

    # --- Database ---
    database_url: str = "sqlite:///./epc.db"

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- RAG Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_retrieval: int = 5

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for dir_path in [
            self.chroma_db_path,
            self.upload_dir,
            self.standards_dir,
            self.test_records_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — call this anywhere to get app settings."""
    return Settings()

"""
EPC Intelligence Core — Entry Point

Run with: uv run uvicorn main:app --reload --port 8000
"""

# Re-export the FastAPI app from the backend package
from backend.main import app  # noqa: F401

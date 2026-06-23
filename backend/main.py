"""
EPC Intelligence Core — FastAPI Application Entry Point

Initializes the FastAPI app with:
  - CORS middleware for frontend dev server
  - All API routers
  - Database table creation on startup
  - Groq connectivity health check
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.db.models import create_tables

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("epc_intelligence")


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()

    # Ensure data directories exist
    settings.ensure_directories()
    logger.info("Data directories verified.")

    # Create database tables
    create_tables()
    logger.info("Database tables created/verified.")

    # Verify Groq connection
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.1,
        )
        response = llm.invoke("Respond with only: OK")
        logger.info(f"Groq connection verified. Model: {settings.groq_model}")
        logger.info(f"Groq response: {response.content[:50]}")
    except Exception as e:
        logger.warning(f"Groq connection check failed: {e}")
        logger.warning("The app will start but LLM features may not work.")

    yield

    logger.info("EPC Intelligence Core shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EPC Intelligence Core",
    description=(
        "Specification & Quality Compliance Agent + Commissioning QA Copilot "
        "for data centre EPC projects. Powered by Groq (Llama-3.3-70b) "
        "with RAG over TIA-942, BIS IS 3043, and Uptime Tier standards."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Frontend SPA Route ─────────────────────────────────────────────────────────

from fastapi.responses import HTMLResponse
import os

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serve the root dashboard unified SPA HTML."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    index_path = os.path.join(root_dir, "index.html")
    
    if not os.path.exists(index_path):
        index_path = "index.html"
        
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h3>Failed to read index.html: {e}</h3>"


# ── Health Check ───────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    """Health check endpoint — verifies app is running and returns config info."""
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "EPC Intelligence Core",
        "version": "0.1.0",
        "groq_model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "database": settings.database_url,
    }


# ── Include Routers ───────────────────────────────────────────────────────────

from backend.api.routes_documents import router as documents_router
from backend.api.routes_documents import standards_router, projects_router
from backend.api.routes_compliance import router as compliance_router
from backend.api.routes_commissioning import router as commissioning_router
from backend.api.routes_rfi import router as rfi_router

app.include_router(documents_router)
app.include_router(standards_router)
app.include_router(projects_router)
app.include_router(compliance_router)
app.include_router(commissioning_router)
app.include_router(rfi_router)

import os
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Create a temp directory for test files
test_dir = tempfile.TemporaryDirectory()
db_file = Path(test_dir.name) / "test.db"

# Set environment variables for testing before importing settings
os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
os.environ["CHROMA_DB_PATH"] = str(Path(test_dir.name) / "chroma")
os.environ["UPLOAD_DIR"] = str(Path(test_dir.name) / "uploads")
os.environ["STANDARDS_DIR"] = str(Path(test_dir.name) / "standards")
os.environ["TEST_RECORDS_DIR"] = str(Path(test_dir.name) / "test_records")

from backend.config import get_settings
from backend.db.models import Base, Project, Document, NCR, CommissioningSession, TestRecord, DocType, Severity, NCRStatus, SessionStatus, TestResult, SystemType
from backend.db.ncr_store import create_ncr, create_ncrs_from_compliance_result, list_ncrs, update_ncr_status, get_ncr_summary
from backend.db.test_record_store import create_session, complete_session, add_test_result, get_session_records, export_test_record_pdf
from backend.rag.document_ingestion import chunk_document, ingest_pdf
from backend.rag.vector_store import add_documents, get_chroma_client, get_or_create_collection, delete_collection, list_collections
from backend.rag.retriever import retrieve, format_context
from backend.agents.orchestrator import classify_intent, route_by_intent, AgentState
from backend.main import app

# Prevent pytest from collecting imported model classes
TestRecord.__test__ = False
TestResult.__test__ = False

# Setup FastAPI TestClient
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def init_db():
    """Create all tables in the file-based SQLite test DB."""
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)
    yield
    # Cleanup will happen when the temp directory is cleaned up

@pytest.fixture(scope="module")
def db_session():
    """Create SQLite session for testing."""
    engine = create_engine(f"sqlite:///{db_file}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── 1. Config Tests ─────────────────────────────────────────────────────────────

def test_settings_load():
    settings = get_settings()
    assert "test.db" in settings.database_url
    assert "chroma" in settings.chroma_db_path
    settings.ensure_directories()
    assert Path(settings.chroma_db_path).exists()
    assert Path(settings.upload_dir).exists()

# ── 2. Database Models and Stores Tests ──────────────────────────────────────────

def test_project_crud(db_session):
    # Create
    project = Project(name="Mumbai DC", location="Mumbai", tier_level="III")
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)
    
    assert project.id is not None
    assert project.name == "Mumbai DC"
    
    # Read
    saved = db_session.query(Project).filter(Project.id == project.id).first()
    assert saved.location == "Mumbai"

def test_ncr_store(db_session):
    # Create Project and Document first
    project = Project(name="Pune DC")
    db_session.add(project)
    db_session.commit()
    
    doc = Document(project_id=project.id, filename="ups_spec.pdf", doc_type=DocType.SUBMITTAL)
    db_session.add(doc)
    db_session.commit()
    
    # Create NCR
    ncr = create_ncr(
        db=db_session,
        project_id=project.id,
        doc_id=doc.id,
        clause_ref="TIA-942 Section 5.3",
        severity="critical",
        submittal_value="100kVA",
        required_value="120kVA",
        deviation_type="below_rating",
        recommendation="Upgrade capacity"
    )
    
    assert ncr.id is not None
    assert ncr.severity == Severity.CRITICAL
    assert ncr.status == NCRStatus.OPEN
    
    # List NCRs
    ncrs = list_ncrs(db_session, project_id=project.id)
    assert len(ncrs) == 1
    assert ncrs[0]["clause_ref"] == "TIA-942 Section 5.3"
    
    # Update Status
    update_result = update_ncr_status(db_session, ncr.id, "resolved")
    assert update_result["status"] == "resolved"
    
    # Summary
    summary = get_ncr_summary(db_session, project.id)
    assert summary["total"] == 1
    assert summary["by_status"]["resolved"] == 1

def test_test_record_store(db_session):
    project = Project(name="Bangalore DC")
    db_session.add(project)
    db_session.commit()
    
    # Create session
    session = create_session(db_session, project.id, "ups")
    assert session.id is not None
    assert session.status == SessionStatus.IN_PROGRESS
    
    # Add test results
    add_test_result(
        db=db_session,
        session_id=session.id,
        system_type="ups",
        step_number=1,
        procedure="Measure voltage",
        expected_range="230V +/- 5%",
        measured_value=228.5,
        result="pass"
    )
    
    records = get_session_records(db_session, session.id)
    assert len(records) == 1
    assert records[0]["measured_value"] == 228.5
    assert records[0]["result"] == "pass"
    
    # Complete session
    completed = complete_session(db_session, session.id)
    assert completed.status == SessionStatus.COMPLETED
    assert completed.completed_at is not None
    
    # Test PDF Generation
    pdf_path = export_test_record_pdf(db_session, session.id)
    assert os.path.exists(pdf_path)

# ── 3. RAG and Vector Store Tests ──────────────────────────────────────────────

def test_chunk_document():
    pages = [
        {"page_number": 1, "text": "This is page 1 content. It contains specifications for UPS system."},
        {"page_number": 2, "text": "This is page 2 content. Battery runtime should be 15 minutes."}
    ]
    chunks = chunk_document(pages, "doc.pdf", "spec", project_id=42)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["source"] == "doc.pdf"
    assert chunks[0]["metadata"]["project_id"] == "42"
    assert "page_number" in chunks[0]["metadata"]

def test_vector_store_operations():
    # Verify Chroma client
    client_chroma = get_chroma_client()
    assert client_chroma is not None
    
    # Create collection & add docs
    col_name = "test_collection"
    delete_collection(col_name) # Ensure clean state
    
    chunks = [
        {
            "text": "The generator system must start within 10 seconds of power failure.",
            "metadata": {"source": "gen_spec.pdf", "page_number": 1, "doc_type": "standard"}
        },
        {
            "text": "Cooling units require N+1 redundancy for Tier III design.",
            "metadata": {"source": "cooling_spec.pdf", "page_number": 3, "doc_type": "standard"}
        }
    ]
    
    added = add_documents(chunks, col_name)
    assert added == 2
    
    # List collections
    cols = list_collections()
    assert col_name in cols
    
    # Retrieve
    results = retrieve("How fast should generator start?", [col_name], top_k=1)
    assert len(results) == 1
    assert "generator" in results[0]["text"]
    assert results[0]["collection"] == col_name
    
    # Clean up
    delete_collection(col_name)

# ── 4. Agent Router/Orchestrator Tests ──────────────────────────────────────────

def test_route_by_intent():
    state_spec: AgentState = {
        "user_input": "check this generator spec sheet",
        "intent": "spec_compliance",
        "project_id": 1,
        "context": {},
        "result": {}
    }
    route = route_by_intent(state_spec)
    assert route == "spec_agent"
    
    state_rfi: AgentState = {
        "user_input": "what is the required temperature range?",
        "intent": "rfi",
        "project_id": 1,
        "context": {},
        "result": {}
    }
    assert route_by_intent(state_rfi) == "rfi_agent"

# ── 5. API Client Endpoints Tests ───────────────────────────────────────────────

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "EPC Intelligence Core"

def test_api_project_lifecycle():
    # Create project
    res = client.post(
        "/api/projects/create",
        data={"name": "Chennai DC", "location": "Chennai", "tier_level": "IV"}
    )
    assert res.status_code == 200
    p_data = res.json()
    p_id = p_data["project_id"]
    assert p_data["name"] == "Chennai DC"
    
    # List projects
    res = client.get("/api/projects/list")
    assert res.status_code == 200
    projects_response = res.json()
    projects = projects_response["projects"]
    assert len(projects) >= 1
    assert any(p["id"] == p_id for p in projects)

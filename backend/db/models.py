"""
EPC Intelligence Core — SQLAlchemy ORM Models

Defines all database tables:
  - Project: top-level project container
  - Document: uploaded/indexed documents
  - NCR: non-conformance reports from spec compliance checks
  - CommissioningSession: commissioning test sessions
  - TestRecord: individual test step results within a session
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from backend.config import get_settings


# ── Enums ──────────────────────────────────────────────────────────────────────


class DocType(str, enum.Enum):
    SUBMITTAL = "submittal"
    SPEC = "spec"
    STANDARD = "standard"
    RFI_LOG = "rfi_log"
    SCHEDULE = "schedule"


class Severity(str, enum.Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class NCRStatus(str, enum.Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    WAIVED = "waived"


class SessionStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABORTED = "aborted"


class TestResult(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


class SystemType(str, enum.Enum):
    UPS = "ups"
    GENERATOR = "generator"
    COOLING = "cooling"
    FIRE_SUPPRESSION = "fire_suppression"
    BMS = "bms"


# ── Base ───────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


# ── Models ─────────────────────────────────────────────────────────────────────


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    tier_level = Column(String(10), nullable=True)  # e.g. "III", "IV"
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    ncrs = relationship("NCR", back_populates="project", cascade="all, delete-orphan")
    sessions = relationship(
        "CommissioningSession", back_populates="project", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    doc_type = Column(Enum(DocType), nullable=False)
    upload_date = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    chunk_count = Column(Integer, default=0)
    file_path = Column(String(1000), nullable=True)

    # Relationships
    project = relationship("Project", back_populates="documents")
    ncrs = relationship("NCR", back_populates="document", cascade="all, delete-orphan")


class NCR(Base):
    __tablename__ = "ncrs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    doc_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    clause_ref = Column(String(255), nullable=False)
    submittal_value = Column(Text, nullable=True)
    required_value = Column(Text, nullable=True)
    deviation_type = Column(String(255), nullable=True)
    severity = Column(Enum(Severity), nullable=False, default=Severity.MINOR)
    status = Column(Enum(NCRStatus), nullable=False, default=NCRStatus.OPEN)
    recommendation = Column(Text, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    project = relationship("Project", back_populates="ncrs")
    document = relationship("Document", back_populates="ncrs")


class CommissioningSession(Base):
    __tablename__ = "commissioning_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    system_type = Column(Enum(SystemType), nullable=False)
    status = Column(
        Enum(SessionStatus), nullable=False, default=SessionStatus.IN_PROGRESS
    )
    started_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="sessions")
    test_records = relationship(
        "TestRecord", back_populates="session", cascade="all, delete-orphan"
    )


class TestRecord(Base):
    __tablename__ = "test_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        Integer, ForeignKey("commissioning_sessions.id"), nullable=False
    )
    system_type = Column(Enum(SystemType), nullable=False)
    step_number = Column(Integer, nullable=False)
    procedure = Column(Text, nullable=False)
    expected_range = Column(String(255), nullable=True)
    measured_value = Column(Float, nullable=True)
    measured_value_text = Column(String(500), nullable=True)  # for non-numeric values
    result = Column(Enum(TestResult), nullable=True)
    notes = Column(Text, nullable=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    session = relationship("CommissioningSession", back_populates="test_records")


# ── Database Engine & Session Factory ──────────────────────────────────────────


def get_engine():
    """Create SQLAlchemy engine from settings."""
    settings = get_settings()
    return create_engine(settings.database_url, echo=False)


def create_tables():
    """Create all tables in the database."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory():
    """Return a sessionmaker bound to the engine."""
    engine = get_engine()
    return sessionmaker(bind=engine)


def get_db() -> Session:
    """FastAPI dependency — yields a DB session and closes it after use."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

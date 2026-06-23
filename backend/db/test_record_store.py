"""
EPC Intelligence Core — Test Record Store

CRUD operations for CommissioningSession and TestRecord models.
Includes PDF export via ReportLab for formal as-commissioned test records.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.db.models import (
    CommissioningSession,
    SessionStatus,
    SystemType,
    TestRecord,
    TestResult,
)

logger = logging.getLogger("epc_intelligence.db.test_record_store")


# ── Session CRUD ───────────────────────────────────────────────────────────────


def create_session(
    db: Session,
    project_id: int,
    system_type: str,
) -> CommissioningSession:
    """Create a new commissioning session."""
    try:
        sys_type = SystemType(system_type)
    except ValueError:
        raise ValueError(
            f"Invalid system_type '{system_type}'. "
            f"Must be one of: {[s.value for s in SystemType]}"
        )

    session = CommissioningSession(
        project_id=project_id,
        system_type=sys_type,
        status=SessionStatus.IN_PROGRESS,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(
        f"Created commissioning session #{session.id} "
        f"(project={project_id}, system={system_type})"
    )
    return session


def complete_session(db: Session, session_id: int) -> Optional[CommissioningSession]:
    """Mark a commissioning session as completed."""
    session = (
        db.query(CommissioningSession)
        .filter(CommissioningSession.id == session_id)
        .first()
    )
    if not session:
        return None

    session.status = SessionStatus.COMPLETED
    session.completed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return session


# ── Test Record CRUD ───────────────────────────────────────────────────────────


def add_test_result(
    db: Session,
    session_id: int,
    system_type: str,
    step_number: int,
    procedure: str,
    expected_range: Optional[str] = None,
    measured_value: Optional[float] = None,
    measured_value_text: Optional[str] = None,
    result: Optional[str] = None,
    notes: Optional[str] = None,
) -> TestRecord:
    """Add a test result record to a session."""
    try:
        sys_type = SystemType(system_type)
    except ValueError:
        sys_type = SystemType.UPS

    test_result_enum = None
    if result:
        try:
            test_result_enum = TestResult(result)
        except ValueError:
            pass

    record = TestRecord(
        session_id=session_id,
        system_type=sys_type,
        step_number=step_number,
        procedure=procedure,
        expected_range=expected_range,
        measured_value=measured_value,
        measured_value_text=measured_value_text,
        result=test_result_enum,
        notes=notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_session_records(db: Session, session_id: int) -> list[dict]:
    """Get all test records for a commissioning session."""
    records = (
        db.query(TestRecord)
        .filter(TestRecord.session_id == session_id)
        .order_by(TestRecord.step_number)
        .all()
    )

    return [
        {
            "id": r.id,
            "session_id": r.session_id,
            "system_type": r.system_type.value if r.system_type else None,
            "step_number": r.step_number,
            "procedure": r.procedure,
            "expected_range": r.expected_range,
            "measured_value": r.measured_value,
            "measured_value_text": r.measured_value_text,
            "result": r.result.value if r.result else None,
            "notes": r.notes,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        }
        for r in records
    ]


# ── PDF Export ─────────────────────────────────────────────────────────────────


def export_test_record_pdf(db: Session, session_id: int) -> str:
    """
    Generate a formal as-commissioned test record PDF.

    Returns the path to the generated PDF file.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    # Get session and records
    session = (
        db.query(CommissioningSession)
        .filter(CommissioningSession.id == session_id)
        .first()
    )
    if not session:
        raise ValueError(f"Session {session_id} not found.")

    records = get_session_records(db, session_id)

    # Generate PDF
    settings = get_settings()
    output_dir = Path(settings.test_records_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"test_record_session_{session_id}_{session.system_type.value}.pdf"
    output_path = output_dir / filename

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceAfter=6,
        spaceBefore=12,
    )
    body_style = styles["Normal"]

    elements = []

    # Title
    elements.append(Paragraph("As-Commissioned Test Record", title_style))
    elements.append(Spacer(1, 5 * mm))

    # Session info
    info_data = [
        ["Session ID:", str(session.id)],
        ["System Type:", session.system_type.value.upper() if session.system_type else "N/A"],
        ["Status:", session.status.value if session.status else "N/A"],
        ["Started:", session.started_at.strftime("%Y-%m-%d %H:%M UTC") if session.started_at else "N/A"],
        [
            "Completed:",
            session.completed_at.strftime("%Y-%m-%d %H:%M UTC")
            if session.completed_at
            else "In Progress",
        ],
    ]
    info_table = Table(info_data, colWidths=[35 * mm, 120 * mm])
    info_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    elements.append(info_table)
    elements.append(Spacer(1, 8 * mm))

    # Test Results Table
    elements.append(Paragraph("Test Results", heading_style))

    if records:
        header = ["Step", "Procedure", "Expected", "Measured", "Result", "Notes"]
        table_data = [header]

        for r in records:
            measured = str(r["measured_value"]) if r["measured_value"] is not None else (r["measured_value_text"] or "-")
            result_val = (r["result"] or "-").upper()
            table_data.append([
                str(r["step_number"]),
                Paragraph(r["procedure"][:80] + ("..." if len(r["procedure"]) > 80 else ""), body_style),
                r["expected_range"] or "-",
                measured,
                result_val,
                Paragraph(r["notes"][:50] if r["notes"] else "-", body_style),
            ])

        result_table = Table(
            table_data,
            colWidths=[12 * mm, 55 * mm, 30 * mm, 25 * mm, 18 * mm, 30 * mm],
        )
        result_table.setStyle(
            TableStyle([
                # Header
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                # Body
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(result_table)
    else:
        elements.append(Paragraph("No test records found for this session.", body_style))

    # Summary
    elements.append(Spacer(1, 10 * mm))
    elements.append(Paragraph("Summary", heading_style))

    total = len(records)
    passed = sum(1 for r in records if r["result"] == "pass")
    failed = sum(1 for r in records if r["result"] == "fail")
    pending = total - passed - failed

    summary_text = (
        f"Total tests: {total} | Passed: {passed} | Failed: {failed} | Pending: {pending}"
    )
    elements.append(Paragraph(summary_text, body_style))

    # Footer
    elements.append(Spacer(1, 15 * mm))
    elements.append(
        Paragraph(
            f"Generated by EPC Intelligence Core | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.grey),
        )
    )

    doc.build(elements)
    logger.info(f"Test record PDF generated: {output_path}")
    return str(output_path)

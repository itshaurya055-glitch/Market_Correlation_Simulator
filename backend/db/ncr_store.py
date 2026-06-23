"""
EPC Intelligence Core — NCR (Non-Conformance Report) Store

CRUD operations for NCR records with severity-based sorting
and filtering capabilities.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from backend.db.models import NCR, NCRStatus, Severity

logger = logging.getLogger("epc_intelligence.db.ncr_store")

# Severity sort order (critical first)
SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.MAJOR: 1,
    Severity.MINOR: 2,
}


def create_ncr(
    db: Session,
    project_id: int,
    doc_id: int,
    clause_ref: str,
    severity: str,
    submittal_value: Optional[str] = None,
    required_value: Optional[str] = None,
    deviation_type: Optional[str] = None,
    recommendation: Optional[str] = None,
) -> NCR:
    """Create a new NCR record."""
    try:
        severity_enum = Severity(severity)
    except ValueError:
        severity_enum = Severity.MINOR

    ncr = NCR(
        project_id=project_id,
        doc_id=doc_id,
        clause_ref=clause_ref,
        submittal_value=submittal_value,
        required_value=required_value,
        deviation_type=deviation_type,
        severity=severity_enum,
        status=NCRStatus.OPEN,
        recommendation=recommendation,
    )
    db.add(ncr)
    db.commit()
    db.refresh(ncr)
    logger.info(f"Created NCR #{ncr.id} (severity={severity}, clause={clause_ref})")
    return ncr


def create_ncrs_from_compliance_result(
    db: Session,
    project_id: int,
    doc_id: int,
    compliance_result: dict,
) -> list[NCR]:
    """
    Bulk-create NCRs from a compliance agent result dict.
    Expects the standard format with 'deviations' array.
    """
    deviations = compliance_result.get("deviations", [])
    created = []

    for dev in deviations:
        ncr = create_ncr(
            db=db,
            project_id=project_id,
            doc_id=doc_id,
            clause_ref=dev.get("clause", "Unknown"),
            severity=dev.get("severity", "minor"),
            submittal_value=dev.get("submittal_value"),
            required_value=dev.get("required_value", dev.get("requirement")),
            deviation_type=dev.get("deviation_type"),
            recommendation=dev.get("recommendation"),
        )
        created.append(ncr)

    logger.info(f"Created {len(created)} NCRs for doc_id={doc_id}")
    return created


def list_ncrs(
    db: Session,
    project_id: int,
    severity_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> list[dict]:
    """
    List all NCRs for a project, sorted by severity (critical first).

    Returns list of dicts for JSON serialization.
    """
    query = db.query(NCR).filter(NCR.project_id == project_id)

    if severity_filter:
        try:
            query = query.filter(NCR.severity == Severity(severity_filter))
        except ValueError:
            pass

    if status_filter:
        try:
            query = query.filter(NCR.status == NCRStatus(status_filter))
        except ValueError:
            pass

    ncrs = query.all()

    # Sort by severity (critical → major → minor), then by creation date
    ncrs.sort(
        key=lambda n: (
            SEVERITY_ORDER.get(n.severity, 99),
            n.created_at or "",
        )
    )

    return [
        {
            "ncr_id": ncr.id,
            "project_id": ncr.project_id,
            "doc_id": ncr.doc_id,
            "clause_ref": ncr.clause_ref,
            "submittal_value": ncr.submittal_value,
            "required_value": ncr.required_value,
            "deviation_type": ncr.deviation_type,
            "severity": ncr.severity.value if ncr.severity else None,
            "status": ncr.status.value if ncr.status else None,
            "recommendation": ncr.recommendation,
            "created_at": ncr.created_at.isoformat() if ncr.created_at else None,
        }
        for ncr in ncrs
    ]


def update_ncr_status(
    db: Session,
    ncr_id: int,
    new_status: str,
) -> Optional[dict]:
    """Update the status of an NCR (open → resolved/waived)."""
    ncr = db.query(NCR).filter(NCR.id == ncr_id).first()
    if not ncr:
        return None

    try:
        ncr.status = NCRStatus(new_status)
    except ValueError:
        raise ValueError(f"Invalid status '{new_status}'. Must be one of: {[s.value for s in NCRStatus]}")

    db.commit()
    db.refresh(ncr)
    logger.info(f"Updated NCR #{ncr_id} status to {new_status}")

    return {
        "ncr_id": ncr.id,
        "status": ncr.status.value,
    }


def get_ncr_summary(db: Session, project_id: int) -> dict:
    """Get a summary count of NCRs by severity and status for a project."""
    ncrs = db.query(NCR).filter(NCR.project_id == project_id).all()

    by_severity = {"critical": 0, "major": 0, "minor": 0}
    by_status = {"open": 0, "resolved": 0, "waived": 0}

    for ncr in ncrs:
        if ncr.severity:
            by_severity[ncr.severity.value] = by_severity.get(ncr.severity.value, 0) + 1
        if ncr.status:
            by_status[ncr.status.value] = by_status.get(ncr.status.value, 0) + 1

    return {
        "total": len(ncrs),
        "by_severity": by_severity,
        "by_status": by_status,
    }

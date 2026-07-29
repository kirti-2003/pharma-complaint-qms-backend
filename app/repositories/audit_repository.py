from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.complaint_audit_log import ComplaintAuditLog
from app.schemas.audit import AuditLogCreate


class AuditRepository:
    """
    Handles database operations related to complaint audit logs.

    Audit logs record workflow-level actions such as complaint creation,
    AI processing, status changes, document uploads, field updates,
    failures, and commitment to the QMS ledger.
    """

    def create_audit_log(
        self,
        db: Session,
        audit_data: AuditLogCreate,
    ) -> ComplaintAuditLog:
        """
        Create a new complaint audit-log record.
        """

        audit_log = ComplaintAuditLog(
            **audit_data.model_dump(exclude_unset=True)
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def get_audit_log_by_id(
        self,
        db: Session,
        audit_log_id: UUID,
    ) -> Optional[ComplaintAuditLog]:
        """
        Retrieve an audit log using its UUID.
        """

        return (
            db.query(ComplaintAuditLog)
            .filter(
                ComplaintAuditLog.audit_log_id == audit_log_id
            )
            .first()
        )

    def get_audit_logs_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> List[ComplaintAuditLog]:
        """
        Retrieve all audit logs belonging to a complaint.

        Logs are returned from oldest to newest.
        """

        return (
            db.query(ComplaintAuditLog)
            .filter(
                ComplaintAuditLog.complaint_id == complaint_id
            )
            .order_by(
                ComplaintAuditLog.created_at.asc()
            )
            .all()
        )

    def get_latest_audit_log(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Optional[ComplaintAuditLog]:
        """
        Retrieve the latest audit log for a complaint.
        """

        return (
            db.query(ComplaintAuditLog)
            .filter(
                ComplaintAuditLog.complaint_id == complaint_id
            )
            .order_by(
                ComplaintAuditLog.created_at.desc()
            )
            .first()
        )

    def get_audit_logs_by_action(
        self,
        db: Session,
        complaint_id: UUID,
        action: str,
    ) -> List[ComplaintAuditLog]:
        """
        Retrieve audit logs for a particular workflow action.
        """

        return (
            db.query(ComplaintAuditLog)
            .filter(
                ComplaintAuditLog.complaint_id == complaint_id,
                ComplaintAuditLog.action == action,
            )
            .order_by(
                ComplaintAuditLog.created_at.asc()
            )
            .all()
        )

    def get_audit_logs_by_performed_by(
        self,
        db: Session,
        complaint_id: UUID,
        performed_by: str,
    ) -> List[ComplaintAuditLog]:
        """
        Retrieve audit logs according to who performed the action.

        Supported values:

        USER
        AI
        SYSTEM
        """

        return (
            db.query(ComplaintAuditLog)
            .filter(
                ComplaintAuditLog.complaint_id == complaint_id,
                ComplaintAuditLog.performed_by == performed_by,
            )
            .order_by(
                ComplaintAuditLog.created_at.asc()
            )
            .all()
        )

    def get_status_change_logs(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> List[ComplaintAuditLog]:
        """
        Retrieve audit logs that contain a complaint status change.
        """

        return (
            db.query(ComplaintAuditLog)
            .filter(
                ComplaintAuditLog.complaint_id == complaint_id,
                ComplaintAuditLog.new_status.isnot(None),
            )
            .order_by(
                ComplaintAuditLog.created_at.asc()
            )
            .all()
        )

    def create_user_audit_log(
        self,
        db: Session,
        complaint_id: UUID,
        action: str,
        description: Optional[str] = None,
        previous_status: Optional[str] = None,
        new_status: Optional[str] = None,
        audit_metadata: Optional[dict] = None,
    ) -> ComplaintAuditLog:
        """
        Create an audit log for an action performed by the user.
        """

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action=action,
            performed_by="USER",
            previous_status=previous_status,
            new_status=new_status,
            description=description,
            audit_metadata=audit_metadata or {},
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_ai_audit_log(
        self,
        db: Session,
        complaint_id: UUID,
        action: str,
        description: Optional[str] = None,
        previous_status: Optional[str] = None,
        new_status: Optional[str] = None,
        audit_metadata: Optional[dict] = None,
    ) -> ComplaintAuditLog:
        """
        Create an audit log for an action performed by the AI workflow.
        """

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action=action,
            performed_by="AI",
            previous_status=previous_status,
            new_status=new_status,
            description=description,
            audit_metadata=audit_metadata or {},
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_system_audit_log(
        self,
        db: Session,
        complaint_id: UUID,
        action: str,
        description: Optional[str] = None,
        previous_status: Optional[str] = None,
        new_status: Optional[str] = None,
        audit_metadata: Optional[dict] = None,
    ) -> ComplaintAuditLog:
        """
        Create an audit log for an action performed automatically
        by the application.
        """

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action=action,
            performed_by="SYSTEM",
            previous_status=previous_status,
            new_status=new_status,
            description=description,
            audit_metadata=audit_metadata or {},
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_status_change_log(
        self,
        db: Session,
        complaint_id: UUID,
        previous_status: Optional[str],
        new_status: str,
        performed_by: str,
        description: Optional[str] = None,
        audit_metadata: Optional[dict] = None,
    ) -> ComplaintAuditLog:
        """
        Record a complaint status transition.
        """

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action="STATUS_CHANGED",
            performed_by=performed_by,
            previous_status=previous_status,
            new_status=new_status,
            description=description,
            audit_metadata=audit_metadata or {},
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_complaint_created_log(
        self,
        db: Session,
        complaint_id: UUID,
        audit_metadata: Optional[dict] = None,
    ) -> ComplaintAuditLog:
        """
        Record the creation of a complaint draft.
        """

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action="COMPLAINT_CREATED",
            performed_by="USER",
            new_status="PENDING_TRIAGE",
            description="Complaint draft created.",
            audit_metadata=audit_metadata or {},
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_ai_processing_started_log(
        self,
        db: Session,
        complaint_id: UUID,
        ai_run_id: Optional[UUID] = None,
        trigger_type: Optional[str] = None,
    ) -> ComplaintAuditLog:
        """
        Record the start of an AI-processing workflow.
        """

        metadata = {}

        if ai_run_id is not None:
            metadata["ai_run_id"] = str(ai_run_id)

        if trigger_type is not None:
            metadata["trigger_type"] = trigger_type

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action="AI_PROCESSING_STARTED",
            performed_by="AI",
            description="AI complaint processing started.",
            audit_metadata=metadata,
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_ai_processing_completed_log(
        self,
        db: Session,
        complaint_id: UUID,
        ai_run_id: Optional[UUID] = None,
        audit_metadata: Optional[dict] = None,
    ) -> ComplaintAuditLog:
        """
        Record the successful completion of AI processing.
        """

        metadata = audit_metadata.copy() if audit_metadata else {}

        if ai_run_id is not None:
            metadata["ai_run_id"] = str(ai_run_id)

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action="AI_PROCESSING_COMPLETED",
            performed_by="AI",
            description="AI complaint processing completed.",
            audit_metadata=metadata,
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_ai_processing_failed_log(
        self,
        db: Session,
        complaint_id: UUID,
        error_message: str,
        ai_run_id: Optional[UUID] = None,
    ) -> ComplaintAuditLog:
        """
        Record a failed AI-processing workflow.
        """

        metadata = {
            "error_message": error_message,
        }

        if ai_run_id is not None:
            metadata["ai_run_id"] = str(ai_run_id)

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action="AI_PROCESSING_FAILED",
            performed_by="AI",
            description=error_message,
            audit_metadata=metadata,
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_attachment_uploaded_log(
        self,
        db: Session,
        complaint_id: UUID,
        attachment_id: UUID,
        original_file_name: Optional[str] = None,
    ) -> ComplaintAuditLog:
        """
        Record the upload of a complaint attachment.
        """

        metadata = {
            "attachment_id": str(attachment_id),
        }

        if original_file_name is not None:
            metadata["original_file_name"] = original_file_name

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action="ATTACHMENT_UPLOADED",
            performed_by="USER",
            description="A complaint attachment was uploaded.",
            audit_metadata=metadata,
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def create_complaint_committed_log(
        self,
        db: Session,
        complaint_id: UUID,
        previous_status: Optional[str] = None,
        audit_metadata: Optional[dict] = None,
    ) -> ComplaintAuditLog:
        """
        Record commitment of the complaint to the QMS ledger.
        """

        audit_log = ComplaintAuditLog(
            complaint_id=complaint_id,
            action="COMPLAINT_COMMITTED",
            performed_by="USER",
            previous_status=previous_status,
            new_status="COMMITTED",
            description="Complaint committed to the QMS ledger.",
            audit_metadata=audit_metadata or {},
        )

        db.add(audit_log)
        db.flush()
        db.refresh(audit_log)

        return audit_log

    def count_audit_logs_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> int:
        """
        Count audit logs belonging to a complaint.
        """

        return (
            db.query(ComplaintAuditLog)
            .filter(
                ComplaintAuditLog.complaint_id == complaint_id
            )
            .count()
        )

    def audit_log_belongs_to_complaint(
        self,
        db: Session,
        audit_log_id: UUID,
        complaint_id: UUID,
    ) -> bool:
        """
        Check whether an audit log belongs to a complaint.
        """

        audit_log = (
            db.query(ComplaintAuditLog.audit_log_id)
            .filter(
                ComplaintAuditLog.audit_log_id == audit_log_id,
                ComplaintAuditLog.complaint_id == complaint_id,
            )
            .first()
        )

        return audit_log is not None
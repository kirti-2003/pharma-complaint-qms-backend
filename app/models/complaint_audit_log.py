from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    String,
    Text,
    TIMESTAMP,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ComplaintAuditLog(Base):
    """
    Stores workflow-level events for a complaint.

    Examples include complaint creation, AI processing,
    status changes, and commitment to the QMS ledger.
    """

    __tablename__ = "complaint_audit_logs"

    __table_args__ = (
        CheckConstraint(
            """
            performed_by IN (
                'USER',
                'AI',
                'SYSTEM'
            )
            """,
            name="chk_audit_performed_by",
        ),
        {"schema": "public"},
    )

    audit_log_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    complaint_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaints.complaint_id",
            name="fk_audit_complaint",
        ),
        nullable=False,
        index=True,
    )

    action = Column(
        String(100),
        nullable=False,
    )

    performed_by = Column(
        String(20),
        nullable=False,
    )

    previous_status = Column(
        String(30),
        nullable=True,
    )

    new_status = Column(
        String(30),
        nullable=True,
    )

    description = Column(
        Text,
        nullable=True,
    )

    audit_metadata = Column(
        JSONB,
        nullable=True,
    )

    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    complaint = relationship(
        "Complaint",
        back_populates="audit_logs",
    )

    def __repr__(self):
        return (
            f"<ComplaintAuditLog("
            f"audit_log_id={self.audit_log_id}, "
            f"action='{self.action}', "
            f"performed_by='{self.performed_by}'"
            f")>"
        )
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
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ComplaintFieldChange(Base):
    """
    Stores the history of individual complaint field changes.

    This identifies the old value, new value, who made the change,
    and the source from which the change originated.
    """

    __tablename__ = "complaint_field_changes"

    __table_args__ = (
        CheckConstraint(
            """
            changed_by IN (
                'USER',
                'AI',
                'SYSTEM'
            )
            """,
            name="chk_field_changed_by",
        ),
        CheckConstraint(
            """
            change_source IN (
                'INITIAL_EXTRACTION',
                'CHAT_CORRECTION',
                'MANUAL_EDIT',
                'FILE_REPROCESSING',
                'SYSTEM_UPDATE'
            )
            """,
            name="chk_field_change_source",
        ),
        {"schema": "public"},
    )

    field_change_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    complaint_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaints.complaint_id",
            name="fk_field_change_complaint",
        ),
        nullable=False,
        index=True,
    )

    ai_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaint_ai_runs.ai_run_id",
            name="fk_field_change_ai_run",
        ),
        nullable=True,
        index=True,
    )

    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaint_messages.message_id",
            name="fk_field_change_message",
        ),
        nullable=True,
        index=True,
    )

    field_name = Column(
        String(100),
        nullable=False,
    )

    old_value = Column(
        Text,
        nullable=True,
    )

    new_value = Column(
        Text,
        nullable=True,
    )

    changed_by = Column(
        String(20),
        nullable=False,
    )

    change_source = Column(
        String(30),
        nullable=False,
    )

    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    complaint = relationship(
        "Complaint",
        back_populates="field_changes",
    )

    ai_run = relationship(
        "ComplaintAIRun",
        back_populates="field_changes",
    )

    message = relationship(
        "ComplaintMessage",
        back_populates="field_changes",
    )

    def __repr__(self):
        return (
            f"<ComplaintFieldChange("
            f"field_change_id={self.field_change_id}, "
            f"field_name='{self.field_name}', "
            f"changed_by='{self.changed_by}'"
            f")>"
        )
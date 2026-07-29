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


class ComplaintMessage(Base):
    """
    Stores user, assistant, and system messages associated
    with a complaint.
    """

    __tablename__ = "complaint_messages"

    __table_args__ = (
        CheckConstraint(
            """
            sender_type IN (
                'USER',
                'ASSISTANT',
                'SYSTEM'
            )
            """,
            name="chk_message_sender_type",
        ),
        CheckConstraint(
            """
            message_type IN (
                'TEXT',
                'FILE',
                'EXTRACTION_RESULT',
                'FIELD_UPDATE',
                'ERROR'
            )
            """,
            name="chk_message_type",
        ),
        {"schema": "public"},
    )

    message_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    complaint_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaints.complaint_id",
            name="fk_message_complaint",
        ),
        nullable=False,
        index=True,
    )

    attachment_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaint_attachments.attachment_id",
            name="fk_message_attachment",
        ),
        nullable=True,
        index=True,
    )

    sender_type = Column(
        String(20),
        nullable=False,
    )

    message_type = Column(
        String(30),
        nullable=False,
        server_default=text("'TEXT'"),
    )

    message_text = Column(
        Text,
        nullable=True,
    )

    message_metadata = Column(
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
        back_populates="messages",
    )

    attachment = relationship(
        "ComplaintAttachment",
        back_populates="messages",
    )

    field_changes = relationship(
        "ComplaintFieldChange",
        back_populates="message",
    )

    def __repr__(self):
        return (
            f"<ComplaintMessage("
            f"message_id={self.message_id}, "
            f"sender_type='{self.sender_type}', "
            f"message_type='{self.message_type}'"
            f")>"
        )
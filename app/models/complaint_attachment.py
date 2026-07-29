from sqlalchemy import (
    BigInteger,
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


class ComplaintAttachment(Base):
    """
    Stores files attached to a complaint.

    Supported files may include PDFs, emails, images,
    DOCX files, and text documents.
    """

    __tablename__ = "complaint_attachments"

    __table_args__ = (
        CheckConstraint(
            """
            document_type IN (
                'COMPLAINT_PDF',
                'EMAIL',
                'IMAGE',
                'DOCX',
                'TXT',
                'OTHER'
            )
            """,
            name="chk_attachment_document_type",
        ),
        CheckConstraint(
            """
            extraction_status IN (
                'PENDING',
                'PROCESSING',
                'COMPLETED',
                'FAILED'
            )
            """,
            name="chk_attachment_extraction_status",
        ),
        {"schema": "public"},
    )

    attachment_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    complaint_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaints.complaint_id",
            name="fk_attachment_complaint",
        ),
        nullable=False,
        index=True,
    )

    original_file_name = Column(
        String(255),
        nullable=False,
    )

    stored_file_name = Column(
        String(255),
        nullable=True,
    )

    storage_path = Column(
        Text,
        nullable=False,
    )

    mime_type = Column(
        String(150),
        nullable=True,
    )

    file_extension = Column(
        String(20),
        nullable=True,
    )

    file_size_bytes = Column(
        BigInteger,
        nullable=True,
    )

    document_type = Column(
        String(30),
        nullable=False,
        server_default=text("'OTHER'"),
    )

    extracted_text = Column(
        Text,
        nullable=True,
    )

    extraction_status = Column(
        String(30),
        nullable=False,
        server_default=text("'PENDING'"),
    )

    extraction_error = Column(
        Text,
        nullable=True,
    )

    uploaded_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    complaint = relationship(
        "Complaint",
        back_populates="attachments",
    )

    messages = relationship(
        "ComplaintMessage",
        back_populates="attachment",
    )

    def __repr__(self):
        return (
            f"<ComplaintAttachment("
            f"attachment_id={self.attachment_id}, "
            f"original_file_name='{self.original_file_name}', "
            f"extraction_status='{self.extraction_status}'"
            f")>"
        )
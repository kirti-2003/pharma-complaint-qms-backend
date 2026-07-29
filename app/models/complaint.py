from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    Numeric,
    String,
    Text,
    TIMESTAMP,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Complaint(Base):
    """
    Main complaint record.

    Stores the structured complaint details extracted from text,
    email, PDF, image, or document input.
    """

    __tablename__ = "complaints"

    __table_args__ = (
        CheckConstraint(
            """
            suggested_severity IS NULL
            OR suggested_severity IN ('MINOR', 'MAJOR', 'CRITICAL')
            """,
            name="chk_complaint_severity",
        ),
        CheckConstraint(
            """
            status IN (
                'PENDING_TRIAGE',
                'PROCESSING',
                'NEEDS_INFORMATION',
                'READY_TO_COMMIT',
                'COMMITTED',
                'PROCESSING_FAILED'
            )
            """,
            name="chk_complaint_status",
        ),
        CheckConstraint(
            """
            input_type IN (
                'TEXT',
                'EMAIL',
                'PDF',
                'IMAGE',
                'DOCUMENT'
            )
            """,
            name="chk_complaint_input_type",
        ),
        {"schema": "public"},
    )

    complaint_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    complaint_number = Column(
        String(30),
        unique=True,
        nullable=True,
    )

    # -----------------------
    # Origin and customer
    # -----------------------
    complaint_source = Column(String(100), nullable=True)
    customer_name = Column(String(255), nullable=True)

    # -----------------------
    # Product and batch
    # -----------------------
    product_name = Column(String(255), nullable=True)
    product_strength_grade = Column(String(100), nullable=True)
    batch_lot_number = Column(String(100), nullable=True)

    affected_quantity_text = Column(String(100), nullable=True)
    affected_quantity_value = Column(Numeric, nullable=True)
    affected_quantity_unit = Column(String(50), nullable=True)

    manufacturing_date_text = Column(String(50), nullable=True)
    manufacturing_date = Column(Date, nullable=True)

    expiry_date_text = Column(String(50), nullable=True)
    expiry_date = Column(Date, nullable=True)

    # -----------------------
    # Facility and material
    # -----------------------
    originating_site_block = Column(String(150), nullable=True)
    impacted_non_product_materials = Column(String(255), nullable=True)

    # -----------------------
    # Complaint analysis
    # -----------------------
    complaint_category = Column(String(255), nullable=True)
    complaint_description = Column(Text, nullable=True)

    suggested_severity = Column(String(20), nullable=True)
    suggested_next_action = Column(Text, nullable=True)
    initial_risk_assessment = Column(Text, nullable=True)

    # -----------------------
    # Raw input and workflow
    # -----------------------
    raw_complaint_text = Column(Text, nullable=True)

    input_type = Column(
        String(20),
        nullable=False,
        server_default=text("'TEXT'"),
    )

    status = Column(
        String(30),
        nullable=False,
        server_default=text("'PENDING_TRIAGE'"),
    )

    is_committed = Column(
        Boolean,
        nullable=False,
        server_default=text("false"),
    )

    committed_at = Column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )

    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # -----------------------
    # Relationships
    # -----------------------
    attachments = relationship(
        "ComplaintAttachment",
        back_populates="complaint",
    )

    messages = relationship(
        "ComplaintMessage",
        back_populates="complaint",
    )

    ai_runs = relationship(
        "ComplaintAIRun",
        back_populates="complaint",
    )

    field_changes = relationship(
        "ComplaintFieldChange",
        back_populates="complaint",
    )

    audit_logs = relationship(
        "ComplaintAuditLog",
        back_populates="complaint",
    )

    def __repr__(self):
        return (
            f"<Complaint("
            f"complaint_id={self.complaint_id}, "
            f"complaint_number='{self.complaint_number}', "
            f"status='{self.status}'"
            f")>"
        )
from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Integer,
    String,
    Text,
    TIMESTAMP,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ComplaintAIRun(Base):
    """
    Stores one execution of the LangGraph complaint-processing workflow.

    A complaint can have multiple AI runs, such as initial processing,
    chat correction, reanalysis, or file reprocessing.
    """

    __tablename__ = "complaint_ai_runs"

    __table_args__ = (
        CheckConstraint(
            """
            trigger_type IN (
                'TEXT_SUBMISSION',
                'FILE_UPLOAD',
                'CHAT_CORRECTION',
                'REANALYSIS'
            )
            """,
            name="chk_ai_trigger_type",
        ),
        CheckConstraint(
            """
            run_status IN (
                'STARTED',
                'PROCESSING',
                'WAITING_FOR_USER',
                'COMPLETED',
                'FAILED'
            )
            """,
            name="chk_ai_run_status",
        ),
        {"schema": "public"},
    )

    ai_run_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
    )

    complaint_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "public.complaints.complaint_id",
            name="fk_ai_run_complaint",
        ),
        nullable=False,
        index=True,
    )

    trigger_type = Column(
        String(30),
        nullable=False,
    )

    model_provider = Column(
        String(50),
        nullable=False,
        server_default=text("'GROQ'"),
    )

    model_name = Column(
        String(100),
        nullable=False,
        server_default=text("'gemma2-9b-it'"),
    )

    langgraph_thread_id = Column(
        String(255),
        nullable=True,
    )

    langgraph_run_id = Column(
        String(255),
        nullable=True,
    )

    run_status = Column(
        String(30),
        nullable=False,
        server_default=text("'STARTED'"),
    )

    input_payload = Column(JSONB, nullable=True)
    extracted_fields = Column(JSONB, nullable=True)
    missing_fields = Column(JSONB, nullable=True)
    classification_result = Column(JSONB, nullable=True)
    risk_assessment_result = Column(JSONB, nullable=True)
    final_output = Column(JSONB, nullable=True)

    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)

    error_message = Column(
        Text,
        nullable=True,
    )

    started_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    completed_at = Column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )

    complaint = relationship(
        "Complaint",
        back_populates="ai_runs",
    )

    field_changes = relationship(
        "ComplaintFieldChange",
        back_populates="ai_run",
    )

    def __repr__(self):
        return (
            f"<ComplaintAIRun("
            f"ai_run_id={self.ai_run_id}, "
            f"trigger_type='{self.trigger_type}', "
            f"run_status='{self.run_status}'"
            f")>"
        )
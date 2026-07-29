from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.complaint import ComplaintStatus


AuditPerformedBy = Literal[
    "USER",
    "AI",
    "SYSTEM",
]


# -----------------------
# Create Audit Log
# -----------------------
class AuditLogCreate(BaseModel):
    complaint_id: UUID

    action: str = Field(
        min_length=1,
        max_length=100,
    )

    performed_by: AuditPerformedBy

    previous_status: ComplaintStatus | None = None

    new_status: ComplaintStatus | None = None

    description: str | None = None

    audit_metadata: dict[str, Any] | None = None


# -----------------------
# Audit Log Response
# -----------------------
class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    audit_log_id: UUID
    complaint_id: UUID

    action: str
    performed_by: AuditPerformedBy

    previous_status: ComplaintStatus | None
    new_status: ComplaintStatus | None

    description: str | None

    audit_metadata: dict[str, Any] | None

    created_at: datetime
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


FieldChangedBy = Literal[
    "USER",
    "AI",
    "SYSTEM",
]

FieldChangeSource = Literal[
    "INITIAL_EXTRACTION",
    "CHAT_CORRECTION",
    "MANUAL_EDIT",
    "FILE_REPROCESSING",
    "SYSTEM_UPDATE",
]


# -----------------------
# Create Field Change
# -----------------------
class FieldChangeCreate(BaseModel):
    complaint_id: UUID

    ai_run_id: UUID | None = None

    message_id: UUID | None = None

    field_name: str = Field(
        min_length=1,
        max_length=100,
    )

    old_value: str | None = None

    new_value: str | None = None

    changed_by: FieldChangedBy

    change_source: FieldChangeSource


# -----------------------
# Field Change Response
# -----------------------
class FieldChangeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    field_change_id: UUID
    complaint_id: UUID

    ai_run_id: UUID | None
    message_id: UUID | None

    field_name: str

    old_value: str | None
    new_value: str | None

    changed_by: FieldChangedBy
    change_source: FieldChangeSource

    created_at: datetime
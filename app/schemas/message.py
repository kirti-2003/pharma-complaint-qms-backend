from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


MessageSenderType = Literal[
    "USER",
    "ASSISTANT",
    "SYSTEM",
]

MessageType = Literal[
    "TEXT",
    "FILE",
    "EXTRACTION_RESULT",
    "FIELD_UPDATE",
    "ERROR",
]


# -----------------------
# Create Message
# -----------------------
class MessageCreate(BaseModel):
    complaint_id: UUID

    attachment_id: UUID | None = None

    sender_type: MessageSenderType = "USER"

    message_type: MessageType = "TEXT"

    message_text: str | None = None

    message_metadata: dict[str, Any] | None = None


# -----------------------
# Chat Request
# -----------------------
class ComplaintChatRequest(BaseModel):
    """
    Request sent when the user asks the AI assistant to correct
    or update complaint information.
    """

    message: str = Field(
        min_length=1,
        description="User message sent to the complaint assistant.",
    )


# -----------------------
# Message Response
# -----------------------
class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: UUID
    complaint_id: UUID
    attachment_id: UUID | None

    sender_type: MessageSenderType
    message_type: MessageType

    message_text: str | None
    message_metadata: dict[str, Any] | None

    created_at: datetime
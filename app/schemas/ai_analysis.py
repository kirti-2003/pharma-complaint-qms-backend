from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


AIRunTriggerType = Literal[
    "TEXT_SUBMISSION",
    "FILE_UPLOAD",
    "CHAT_CORRECTION",
    "REANALYSIS",
]

AIRunStatus = Literal[
    "STARTED",
    "PROCESSING",
    "WAITING_FOR_USER",
    "COMPLETED",
    "FAILED",
]


# -----------------------
# Create AI Run
# -----------------------
class AIRunCreate(BaseModel):
    complaint_id: UUID

    trigger_type: AIRunTriggerType

    model_provider: str = "GROQ"

    model_name: str = "gemma2-9b-it"

    langgraph_thread_id: str | None = None

    langgraph_run_id: str | None = None

    input_payload: dict[str, Any] | None = None


# -----------------------
# Update AI Run
# -----------------------
class AIRunUpdate(BaseModel):
    run_status: AIRunStatus | None = None

    langgraph_thread_id: str | None = None

    langgraph_run_id: str | None = None

    extracted_fields: dict[str, Any] | None = None

    missing_fields: list[str] | dict[str, Any] | None = None

    classification_result: dict[str, Any] | None = None

    risk_assessment_result: dict[str, Any] | None = None

    final_output: dict[str, Any] | None = None

    prompt_tokens: int | None = None

    completion_tokens: int | None = None

    total_tokens: int | None = None

    error_message: str | None = None

    completed_at: datetime | None = None


# -----------------------
# AI Run Response
# -----------------------
class AIRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ai_run_id: UUID
    complaint_id: UUID

    trigger_type: AIRunTriggerType

    model_provider: str
    model_name: str

    langgraph_thread_id: str | None
    langgraph_run_id: str | None

    run_status: AIRunStatus

    input_payload: dict[str, Any] | None
    extracted_fields: dict[str, Any] | None
    missing_fields: list[str] | dict[str, Any] | None
    classification_result: dict[str, Any] | None
    risk_assessment_result: dict[str, Any] | None
    final_output: dict[str, Any] | None

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    error_message: str | None

    started_at: datetime
    completed_at: datetime | None
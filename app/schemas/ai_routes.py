from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


AITriggerType = Literal[
    "TEXT_SUBMISSION",
    "FILE_UPLOAD",
    "REANALYSIS",
]

AIRunStatus = Literal[
    "STARTED",
    "PROCESSING",
    "WAITING_FOR_USER",
    "COMPLETED",
    "FAILED",
]


class ProcessComplaintRequest(BaseModel):
    """
    Request body for starting or rerunning AI complaint processing.
    """

    trigger_type: AITriggerType = "TEXT_SUBMISSION"


class ChatCorrectionRequest(BaseModel):
    """
    Request body for applying a user correction through chat.
    """

    message_text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="User correction or additional complaint information.",
        examples=[
            (
                "Correction: the affected quantity was 15 capsules, "
                "not 12."
            )
        ],
    )


class AIRunResponse(BaseModel):
    """
    Public API representation of a complaint AI run.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    ai_run_id: UUID
    complaint_id: UUID

    trigger_type: str
    model_provider: str
    model_name: str
    run_status: str

    input_payload: dict[str, Any] | None = None
    extracted_fields: dict[str, Any] | None = None
    missing_fields: list[str] | None = None

    classification_result: dict[str, Any] | None = None
    risk_assessment_result: dict[str, Any] | None = None
    final_output: dict[str, Any] | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    error_message: str | None = None

    langgraph_thread_id: str | None = None
    langgraph_run_id: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AIRunListResponse(BaseModel):
    """
    Response returned when listing complaint AI runs.
    """

    complaint_id: UUID
    total: int
    items: list[AIRunResponse]


class AIProcessingResponse(BaseModel):
    """
    Standard response for initial processing and chat correction.
    """

    message: str
    complaint_id: UUID
    ai_run: AIRunResponse
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.ai_routes import (
    AIProcessingResponse,
    AIRunListResponse,
    AIRunResponse,
    ProcessComplaintRequest,
    ChatCorrectionRequest,
)
from app.services.ai_service import AIService


router = APIRouter(
    prefix="/ai",
    tags=["AI Complaint Processing"],
)

ai_service = AIService()


@router.post(
    "/complaints/{complaint_id}/process",
    response_model=AIProcessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Process complaint with AI",
    description=(
        "Run complaint extraction, validation, classification, "
        "risk assessment, and persistence through LangGraph."
    ),
)
def process_complaint_with_ai(
    complaint_id: UUID,
    request_data: ProcessComplaintRequest,
    db: Session = Depends(get_db),
) -> AIProcessingResponse:
    ai_run = ai_service.process_complaint(
        db=db,
        complaint_id=complaint_id,
        trigger_type=request_data.trigger_type,
    )

    return AIProcessingResponse(
        message="Complaint AI processing completed.",
        complaint_id=complaint_id,
        ai_run=AIRunResponse.model_validate(ai_run),
    )


@router.post(
    "/complaints/{complaint_id}/chat",
    response_model=AIProcessingResponse,
    status_code=status.HTTP_200_OK,
    summary="Apply complaint chat correction",
    description=(
        "Apply a user correction or additional information to an "
        "existing AI-processed complaint."
    ),
)
def process_complaint_chat_correction(
    complaint_id: UUID,
    request_data: ChatCorrectionRequest,
    db: Session = Depends(get_db),
) -> AIProcessingResponse:
    ai_run = ai_service.process_chat_correction(
        db=db,
        complaint_id=complaint_id,
        message_text=request_data.message_text,
    )

    return AIProcessingResponse(
        message="Complaint chat correction processed successfully.",
        complaint_id=complaint_id,
        ai_run=AIRunResponse.model_validate(ai_run),
    )


@router.get(
    "/complaints/{complaint_id}/runs",
    response_model=AIRunListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get complaint AI runs",
    description=(
        "Return all AI processing runs associated with a complaint, "
        "including initial processing and chat corrections."
    ),
)
def get_complaint_ai_runs(
    complaint_id: UUID,
    db: Session = Depends(get_db),
) -> AIRunListResponse:
    ai_runs = ai_service.get_complaint_ai_runs(
        db=db,
        complaint_id=complaint_id,
    )

    return AIRunListResponse(
        complaint_id=complaint_id,
        total=len(ai_runs),
        items=[
            AIRunResponse.model_validate(ai_run)
            for ai_run in ai_runs
        ],
    )
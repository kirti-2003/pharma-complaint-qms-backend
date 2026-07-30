from __future__ import annotations
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.complaint import (
    ComplaintCreate,
    ComplaintCommitRequest,
    ComplaintResponse,
)
from app.services.complaint_service import ComplaintService


router = APIRouter(
    prefix="/complaints",
    tags=["Complaints"],
)

complaint_service = ComplaintService()


@router.post(
    "",
    response_model=ComplaintResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create complaint",
    description=(
        "Create a new complaint draft. The returned complaint ID "
        "can then be used for AI processing."
    ),
)
def create_complaint(
    complaint_data: ComplaintCreate,
    db: Session = Depends(get_db),
) -> ComplaintResponse:
    complaint = complaint_service.create_complaint(
        db=db,
        complaint_data=complaint_data,
    )

    return ComplaintResponse.model_validate(complaint)


@router.get(
    "/{complaint_id}",
    response_model=ComplaintResponse,
    status_code=status.HTTP_200_OK,
    summary="Get complaint by ID",
)
def get_complaint_by_id(
    complaint_id: UUID,
    db: Session = Depends(get_db),
) -> ComplaintResponse:
    complaint = complaint_service.get_complaint_by_id(
        db=db,
        complaint_id=complaint_id,
    )

    return ComplaintResponse.model_validate(complaint)


@router.post(
    "/{complaint_id}/commit",
    response_model=ComplaintResponse,
    status_code=status.HTTP_200_OK,
    summary="Commit complaint to QMS",
    description=(
        "Commit a completed complaint to the QMS ledger. "
        "Only complaints with READY_TO_COMMIT status can be committed."
    ),
)
def commit_complaint_to_qms(
    complaint_id: UUID,
    request: ComplaintCommitRequest,
    db: Session = Depends(get_db),
) -> ComplaintResponse:
    if not request.confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Commit confirmation is required.",
        )

    complaint = complaint_service.commit_complaint_to_qms(
        db=db,
        complaint_id=complaint_id,
    )

    return ComplaintResponse.model_validate(complaint)
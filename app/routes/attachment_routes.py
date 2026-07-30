from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.attachment import AttachmentResponse
from app.services.attachment_service import AttachmentService


router = APIRouter(
    prefix="/complaints",
    tags=["Complaint Attachments"],
)

attachment_service = AttachmentService()


@router.post(
    "/{complaint_id}/attachments",
    response_model=AttachmentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_complaint_attachment(
    complaint_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AttachmentResponse:
    attachment = await attachment_service.upload_attachment(
        db=db,
        complaint_id=complaint_id,
        file=file,
    )

    attachment = attachment_service.extract_attachment_text(
        db=db,
        attachment_id=attachment.attachment_id,
    )

    return AttachmentResponse.model_validate(attachment)
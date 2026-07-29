from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.complaint_attachment import ComplaintAttachment
from app.schemas.attachment import (
    AttachmentCreate,
    AttachmentExtractionUpdate,
)


class AttachmentRepository:
    """
    Handles database operations related to complaint attachments.

    This repository manages attachment metadata and extraction results.
    File storage, validation, OCR, PDF parsing, and document processing
    should remain in the service layer.
    """

    def create_attachment(
        self,
        db: Session,
        attachment_data: AttachmentCreate,
    ) -> ComplaintAttachment:
        """
        Create a new complaint attachment record.
        """

        attachment = ComplaintAttachment(
            **attachment_data.model_dump(exclude_unset=True)
        )

        db.add(attachment)
        db.flush()
        db.refresh(attachment)

        return attachment

    def get_attachment_by_id(
        self,
        db: Session,
        attachment_id: UUID,
    ) -> Optional[ComplaintAttachment]:
        """
        Retrieve an attachment using its UUID.
        """

        return (
            db.query(ComplaintAttachment)
            .filter(
                ComplaintAttachment.attachment_id == attachment_id
            )
            .first()
        )

    def get_attachments_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> List[ComplaintAttachment]:
        """
        Retrieve all attachments belonging to a complaint.
        """

        return (
            db.query(ComplaintAttachment)
            .filter(
                ComplaintAttachment.complaint_id == complaint_id
            )
            .order_by(
                ComplaintAttachment.uploaded_at.asc()
            )
            .all()
        )

    def get_attachment_by_stored_file_name(
        self,
        db: Session,
        stored_file_name: str,
    ) -> Optional[ComplaintAttachment]:
        """
        Retrieve an attachment using its stored filename.
        """

        return (
            db.query(ComplaintAttachment)
            .filter(
                ComplaintAttachment.stored_file_name == stored_file_name
            )
            .first()
        )

    def update_extraction_status(
        self,
        db: Session,
        attachment: ComplaintAttachment,
        extraction_status: str,
    ) -> ComplaintAttachment:
        """
        Update only the extraction status of an attachment.
        """

        attachment.extraction_status = extraction_status

        db.flush()
        db.refresh(attachment)

        return attachment

    def update_extraction_result(
        self,
        db: Session,
        attachment: ComplaintAttachment,
        extraction_data: AttachmentExtractionUpdate,
    ) -> ComplaintAttachment:
        """
        Update extracted text, status, or extraction error.
        """

        update_data = extraction_data.model_dump(
            exclude_unset=True
        )

        for field_name, field_value in update_data.items():
            setattr(
                attachment,
                field_name,
                field_value,
            )

        db.flush()
        db.refresh(attachment)

        return attachment

    def mark_extraction_as_processing(
        self,
        db: Session,
        attachment: ComplaintAttachment,
    ) -> ComplaintAttachment:
        """
        Mark an attachment as currently being processed.
        """

        attachment.extraction_status = "PROCESSING"
        attachment.extraction_error = None

        db.flush()
        db.refresh(attachment)

        return attachment

    def mark_extraction_as_completed(
        self,
        db: Session,
        attachment: ComplaintAttachment,
        extracted_text: str,
    ) -> ComplaintAttachment:
        """
        Save extracted text and mark extraction as completed.
        """

        attachment.extracted_text = extracted_text
        attachment.extraction_status = "COMPLETED"
        attachment.extraction_error = None

        db.flush()
        db.refresh(attachment)

        return attachment

    def mark_extraction_as_failed(
        self,
        db: Session,
        attachment: ComplaintAttachment,
        error_message: str,
    ) -> ComplaintAttachment:
        """
        Mark extraction as failed and store the error message.
        """

        attachment.extraction_status = "FAILED"
        attachment.extraction_error = error_message

        db.flush()
        db.refresh(attachment)

        return attachment

    def count_attachments_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> int:
        """
        Count attachments associated with a complaint.
        """

        return (
            db.query(ComplaintAttachment)
            .filter(
                ComplaintAttachment.complaint_id == complaint_id
            )
            .count()
        )

    def attachment_exists(
        self,
        db: Session,
        attachment_id: UUID,
        complaint_id: UUID,
    ) -> bool:
        """
        Check whether an attachment belongs to a complaint.
        """

        attachment = (
            db.query(ComplaintAttachment.attachment_id)
            .filter(
                ComplaintAttachment.attachment_id == attachment_id,
                ComplaintAttachment.complaint_id == complaint_id,
            )
            .first()
        )

        return attachment is not None

    def delete_attachment(
        self,
        db: Session,
        attachment: ComplaintAttachment,
    ) -> None:
        """
        Delete an attachment database record.

        Physical file deletion should be handled by the service layer.
        """

        db.delete(attachment)
        db.flush()
from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.complaint_message import ComplaintMessage
from app.repositories.attachment_repository import AttachmentRepository
from app.repositories.audit_repository import AuditRepository
from app.repositories.complaint_repository import ComplaintRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.message import MessageCreate


class MessageService:
    """
    Handles complaint conversation business logic.

    This service validates complaint and attachment relationships,
    creates user, assistant, and system messages, writes audit logs,
    and controls database transactions.
    """

    VALID_SENDER_TYPES = {
        "USER",
        "ASSISTANT",
        "SYSTEM",
    }

    VALID_MESSAGE_TYPES = {
        "TEXT",
        "FILE",
        "EXTRACTION_RESULT",
        "FIELD_UPDATE",
        "ERROR",
    }

    def __init__(self):
        self.message_repository = MessageRepository()
        self.complaint_repository = ComplaintRepository()
        self.attachment_repository = AttachmentRepository()
        self.audit_repository = AuditRepository()

    def get_message_by_id(
        self,
        db: Session,
        message_id: UUID,
    ) -> ComplaintMessage:
        """
        Retrieve a message by its UUID.

        Raises a 404 error when the message does not exist.
        """

        message = self.message_repository.get_message_by_id(
            db=db,
            message_id=message_id,
        )

        if message is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found.",
            )

        return message

    def get_messages_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> List[ComplaintMessage]:
        """
        Retrieve the complete conversation for a complaint.

        Messages are returned from oldest to newest.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        return self.message_repository.get_messages_by_complaint_id(
            db=db,
            complaint_id=complaint_id,
        )

    def get_latest_message(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> Optional[ComplaintMessage]:
        """
        Retrieve the latest message for a complaint.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        return self.message_repository.get_latest_message(
            db=db,
            complaint_id=complaint_id,
        )

    def create_message(
        self,
        db: Session,
        message_data: MessageCreate,
    ) -> ComplaintMessage:
        """
        Create a generic complaint message.

        This method is useful for internal application workflows.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=message_data.complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        if message_data.sender_type not in self.VALID_SENDER_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid message sender type.",
            )

        if message_data.message_type not in self.VALID_MESSAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid message type.",
            )

        if message_data.attachment_id is not None:
            self._validate_attachment(
                db=db,
                complaint_id=message_data.complaint_id,
                attachment_id=message_data.attachment_id,
            )

        if (
            not message_data.message_text
            and message_data.attachment_id is None
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "A message must contain text or an attachment."
                ),
            )

        try:
            message = self.message_repository.create_message(
                db=db,
                message_data=message_data,
            )

            db.commit()
            db.refresh(message)

            return message

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create complaint message.",
            ) from exc

    def create_user_message(
        self,
        db: Session,
        complaint_id: UUID,
        message_text: str,
        attachment_id: Optional[UUID] = None,
        message_metadata: Optional[dict] = None,
    ) -> ComplaintMessage:
        """
        Create a text message submitted by the user.

        User chat messages are allowed only while the complaint
        remains uncommitted.
        """

        complaint = self._get_editable_complaint(
            db=db,
            complaint_id=complaint_id,
        )

        cleaned_message = message_text.strip()

        if not cleaned_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message text cannot be empty.",
            )

        if attachment_id is not None:
            self._validate_attachment(
                db=db,
                complaint_id=complaint.complaint_id,
                attachment_id=attachment_id,
            )

        try:
            message = self.message_repository.create_user_message(
                db=db,
                complaint_id=complaint.complaint_id,
                message_text=cleaned_message,
                attachment_id=attachment_id,
                message_metadata=message_metadata,
            )

            self.audit_repository.create_user_audit_log(
                db=db,
                complaint_id=complaint.complaint_id,
                action="USER_MESSAGE_CREATED",
                description="User sent a complaint chat message.",
                audit_metadata={
                    "message_id": str(message.message_id),
                    "has_attachment": attachment_id is not None,
                },
            )

            db.commit()
            db.refresh(message)

            return message

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save user message.",
            ) from exc

    def create_assistant_message(
        self,
        db: Session,
        complaint_id: UUID,
        message_text: str,
        message_type: str = "TEXT",
        message_metadata: Optional[dict] = None,
    ) -> ComplaintMessage:
        """
        Save a message produced by the AI assistant.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        cleaned_message = message_text.strip()

        if not cleaned_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assistant message text cannot be empty.",
            )

        if message_type not in self.VALID_MESSAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid assistant message type.",
            )

        try:
            message = (
                self.message_repository.create_assistant_message(
                    db=db,
                    complaint_id=complaint_id,
                    message_text=cleaned_message,
                    message_type=message_type,
                    message_metadata=message_metadata,
                )
            )

            db.commit()
            db.refresh(message)

            return message

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save assistant message.",
            ) from exc

    def create_system_message(
        self,
        db: Session,
        complaint_id: UUID,
        message_text: str,
        message_type: str = "TEXT",
        message_metadata: Optional[dict] = None,
    ) -> ComplaintMessage:
        """
        Save an application-generated system message.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        cleaned_message = message_text.strip()

        if not cleaned_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="System message text cannot be empty.",
            )

        if message_type not in self.VALID_MESSAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid system message type.",
            )

        try:
            message = self.message_repository.create_system_message(
                db=db,
                complaint_id=complaint_id,
                message_text=cleaned_message,
                message_type=message_type,
                message_metadata=message_metadata,
            )

            db.commit()
            db.refresh(message)

            return message

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save system message.",
            ) from exc

    def create_extraction_result_message(
        self,
        db: Session,
        complaint_id: UUID,
        message_text: str,
        attachment_id: Optional[UUID] = None,
        message_metadata: Optional[dict] = None,
    ) -> ComplaintMessage:
        """
        Save a document extraction result in the conversation.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        if attachment_id is not None:
            self._validate_attachment(
                db=db,
                complaint_id=complaint_id,
                attachment_id=attachment_id,
            )

        cleaned_message = message_text.strip()

        if not cleaned_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Extraction result cannot be empty.",
            )

        try:
            message = (
                self.message_repository
                .create_extraction_result_message(
                    db=db,
                    complaint_id=complaint_id,
                    message_text=cleaned_message,
                    attachment_id=attachment_id,
                    message_metadata=message_metadata,
                )
            )

            self.audit_repository.create_ai_audit_log(
                db=db,
                complaint_id=complaint_id,
                action="EXTRACTION_RESULT_CREATED",
                description=(
                    "Extracted attachment text was added "
                    "to the complaint conversation."
                ),
                audit_metadata={
                    "message_id": str(message.message_id),
                    "attachment_id": (
                        str(attachment_id)
                        if attachment_id is not None
                        else None
                    ),
                },
            )

            db.commit()
            db.refresh(message)

            return message

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save extraction result.",
            ) from exc

    def create_field_update_message(
        self,
        db: Session,
        complaint_id: UUID,
        message_text: str,
        updated_fields: Optional[list[str]] = None,
        message_metadata: Optional[dict] = None,
    ) -> ComplaintMessage:
        """
        Save an AI response describing complaint field updates.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        cleaned_message = message_text.strip()

        if not cleaned_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field-update message cannot be empty.",
            )

        metadata = message_metadata.copy() if message_metadata else {}

        if updated_fields is not None:
            metadata["updated_fields"] = updated_fields

        try:
            message = (
                self.message_repository.create_field_update_message(
                    db=db,
                    complaint_id=complaint_id,
                    message_text=cleaned_message,
                    message_metadata=metadata,
                )
            )

            db.commit()
            db.refresh(message)

            return message

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save field-update message.",
            ) from exc

    def create_error_message(
        self,
        db: Session,
        complaint_id: UUID,
        error_message: str,
        message_metadata: Optional[dict] = None,
    ) -> ComplaintMessage:
        """
        Save a system error message in the complaint conversation.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        cleaned_error = error_message.strip()

        if not cleaned_error:
            cleaned_error = "An unexpected processing error occurred."

        try:
            message = self.message_repository.create_error_message(
                db=db,
                complaint_id=complaint_id,
                message_text=cleaned_error,
                message_metadata=message_metadata,
            )

            self.audit_repository.create_system_audit_log(
                db=db,
                complaint_id=complaint_id,
                action="ERROR_MESSAGE_CREATED",
                description=cleaned_error,
                audit_metadata={
                    "message_id": str(message.message_id),
                },
            )

            db.commit()
            db.refresh(message)

            return message

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save error message.",
            ) from exc

    def delete_message(
        self,
        db: Session,
        message_id: UUID,
    ) -> None:
        """
        Delete a message from an uncommitted complaint.

        Messages belonging to committed complaints are immutable.
        """

        message = self.get_message_by_id(
            db=db,
            message_id=message_id,
        )

        complaint = self._get_editable_complaint(
            db=db,
            complaint_id=message.complaint_id,
        )

        try:
            deleted_message_id = message.message_id
            deleted_message_type = message.message_type

            self.message_repository.delete_message(
                db=db,
                message=message,
            )

            self.audit_repository.create_user_audit_log(
                db=db,
                complaint_id=complaint.complaint_id,
                action="MESSAGE_DELETED",
                description="A complaint message was deleted.",
                audit_metadata={
                    "message_id": str(deleted_message_id),
                    "message_type": deleted_message_type,
                },
            )

            db.commit()

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete complaint message.",
            ) from exc

    def count_messages(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> int:
        """
        Count messages associated with a complaint.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        return (
            self.message_repository.count_messages_by_complaint_id(
                db=db,
                complaint_id=complaint_id,
            )
        )

    def _get_editable_complaint(
        self,
        db: Session,
        complaint_id: UUID,
    ):
        """
        Retrieve a complaint and verify that it can still be changed.
        """

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        if complaint.is_committed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Committed complaints cannot be modified.",
            )

        return complaint

    def _validate_attachment(
        self,
        db: Session,
        complaint_id: UUID,
        attachment_id: UUID,
    ) -> None:
        """
        Verify that an attachment exists and belongs to the complaint.
        """

        attachment = self.attachment_repository.get_attachment_by_id(
            db=db,
            attachment_id=attachment_id,
        )

        if attachment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Attachment not found.",
            )

        if attachment.complaint_id != complaint_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "The attachment does not belong to this complaint."
                ),
            )
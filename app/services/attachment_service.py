import os
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.complaint_attachment import ComplaintAttachment
from app.repositories.attachment_repository import AttachmentRepository
from app.repositories.audit_repository import AuditRepository
from app.repositories.complaint_repository import ComplaintRepository
from app.repositories.message_repository import MessageRepository
from app.schemas.attachment import AttachmentCreate
from app.utils.file_parser import extract_text_from_file

class AttachmentService:
    """
    Handles complaint attachment business logic.

    Responsibilities include:

    - Validating uploaded files
    - Saving physical files
    - Creating attachment database records
    - Creating complaint chat messages
    - Creating audit logs
    - Managing extraction statuses
    - Deleting attachment files and records
    - Controlling commit and rollback
    """

    ALLOWED_FILE_TYPES = {
        ".pdf": {
            "document_type": "COMPLAINT_PDF",
            "mime_types": {
                "application/pdf",
            },
        },
        ".eml": {
            "document_type": "EMAIL",
            "mime_types": {
                "message/rfc822",
                "application/octet-stream",
            },
        },
        ".msg": {
            "document_type": "EMAIL",
            "mime_types": {
                "application/vnd.ms-outlook",
                "application/octet-stream",
            },
        },
        ".png": {
            "document_type": "IMAGE",
            "mime_types": {
                "image/png",
            },
        },
        ".jpg": {
            "document_type": "IMAGE",
            "mime_types": {
                "image/jpeg",
            },
        },
        ".jpeg": {
            "document_type": "IMAGE",
            "mime_types": {
                "image/jpeg",
            },
        },
        ".webp": {
            "document_type": "IMAGE",
            "mime_types": {
                "image/webp",
            },
        },
        ".docx": {
            "document_type": "DOCX",
            "mime_types": {
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document",
                "application/octet-stream",
            },
        },
        ".txt": {
            "document_type": "TXT",
            "mime_types": {
                "text/plain",
                "application/octet-stream",
            },
        },
    }

    def __init__(
        self,
        upload_directory: str = "uploads/complaints",
        max_upload_size_mb: int = 10,
    ):
        self.attachment_repository = AttachmentRepository()
        self.complaint_repository = ComplaintRepository()
        self.message_repository = MessageRepository()
        self.audit_repository = AuditRepository()

        self.upload_directory = Path(upload_directory)
        self.max_upload_size_bytes = (
            max_upload_size_mb * 1024 * 1024
        )

    def get_attachment_by_id(
        self,
        db: Session,
        attachment_id: UUID,
    ) -> ComplaintAttachment:
        """
        Retrieve an attachment or raise a 404 response.
        """

        attachment = (
            self.attachment_repository.get_attachment_by_id(
                db=db,
                attachment_id=attachment_id,
            )
        )

        if attachment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Attachment not found.",
            )

        return attachment

    def get_attachments_by_complaint_id(
        self,
        db: Session,
        complaint_id: UUID,
    ) -> List[ComplaintAttachment]:
        """
        Retrieve all attachments belonging to a complaint.
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
            self.attachment_repository
            .get_attachments_by_complaint_id(
                db=db,
                complaint_id=complaint_id,
            )
        )

    def validate_file(
        self,
        file: UploadFile,
    ) -> tuple[str, str]:
        """
        Validate the uploaded filename, extension, and MIME type.

        Returns:

        file_extension, document_type
        """

        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a filename.",
            )

        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in self.ALLOWED_FILE_TYPES:
            supported_extensions = ", ".join(
                sorted(self.ALLOWED_FILE_TYPES.keys())
            )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Unsupported file type. Supported files are: "
                    f"{supported_extensions}"
                ),
            )

        file_config = self.ALLOWED_FILE_TYPES[file_extension]
        allowed_mime_types = file_config["mime_types"]

        if (
            file.content_type
            and file.content_type not in allowed_mime_types
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid MIME type for {file_extension} file: "
                    f"{file.content_type}"
                ),
            )

        return (
            file_extension,
            file_config["document_type"],
        )

    async def upload_attachment(
        self,
        db: Session,
        complaint_id: UUID,
        file: UploadFile,
    ) -> ComplaintAttachment:
        """
        Validate and save an uploaded complaint attachment.

        The attachment database record, file message, and audit log
        are committed together in one database transaction.
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
                detail=(
                    "Attachments cannot be added to a committed "
                    "complaint."
                ),
            )

        file_extension, document_type = self.validate_file(file)

        stored_file_name = (
            f"{uuid4().hex}{file_extension}"
        )

        complaint_directory = (
            self.upload_directory / str(complaint_id)
        )

        complaint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        storage_path = (
            complaint_directory / stored_file_name
        )

        file_size_bytes = 0

        try:
            with storage_path.open("wb") as destination:
                while True:
                    chunk = await file.read(1024 * 1024)

                    if not chunk:
                        break

                    file_size_bytes += len(chunk)

                    if (
                        file_size_bytes
                        > self.max_upload_size_bytes
                    ):
                        raise HTTPException(
                            status_code=(
                                status.HTTP_413_CONTENT_TOO_LARGE
                            ),
                            detail=(
                                "Uploaded file exceeds the maximum "
                                "allowed size."
                            ),
                        )

                    destination.write(chunk)

            if file_size_bytes == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty.",
                )

            attachment_data = AttachmentCreate(
                complaint_id=complaint_id,
                original_file_name=file.filename,
                stored_file_name=stored_file_name,
                storage_path=str(storage_path),
                mime_type=file.content_type,
                file_extension=file_extension,
                file_size_bytes=file_size_bytes,
                document_type=document_type,
                
            )

            attachment = (
                self.attachment_repository.create_attachment(
                    db=db,
                    attachment_data=attachment_data,
                )
            )

            self.message_repository.create_file_message(
                db=db,
                complaint_id=complaint_id,
                attachment_id=attachment.attachment_id,
                message_text=(
                    f"Uploaded file: {attachment.original_file_name}"
                ),
                message_metadata={
                    "original_file_name": (
                        attachment.original_file_name
                    ),
                    "document_type": attachment.document_type,
                    "mime_type": attachment.mime_type,
                    "file_size_bytes": (
                        attachment.file_size_bytes
                    ),
                },
            )

            self.audit_repository.create_attachment_uploaded_log(
                db=db,
                complaint_id=complaint_id,
                attachment_id=attachment.attachment_id,
                original_file_name=(
                    attachment.original_file_name
                ),
            )

            db.commit()
            db.refresh(attachment)

            return attachment

        except HTTPException:
            db.rollback()
            self._remove_file_if_exists(storage_path)
            raise

        except SQLAlchemyError as exc:
            db.rollback()
            self._remove_file_if_exists(storage_path)

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail="Failed to save attachment information.",
            ) from exc

        except OSError as exc:
            db.rollback()
            self._remove_file_if_exists(storage_path)

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail="Failed to save the uploaded file.",
            ) from exc

        finally:
            await file.close()

    def mark_extraction_as_processing(
        self,
        db: Session,
        attachment_id: UUID,
    ) -> ComplaintAttachment:
        """
        Mark an attachment extraction operation as processing.
        """

        attachment = self.get_attachment_by_id(
            db=db,
            attachment_id=attachment_id,
        )

        try:
            attachment = (
                self.attachment_repository
                .mark_extraction_as_processing(
                    db=db,
                    attachment=attachment,
                )
            )

            self.audit_repository.create_system_audit_log(
                db=db,
                complaint_id=attachment.complaint_id,
                action="ATTACHMENT_EXTRACTION_STARTED",
                description=(
                    "Text extraction started for attachment."
                ),
                audit_metadata={
                    "attachment_id": str(
                        attachment.attachment_id
                    ),
                    "original_file_name": (
                        attachment.original_file_name
                    ),
                },
            )

            db.commit()
            db.refresh(attachment)

            return attachment

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Failed to update attachment extraction status."
                ),
            ) from exc

    def mark_extraction_as_completed(
        self,
        db: Session,
        attachment_id: UUID,
        extracted_text: str,
    ) -> ComplaintAttachment:
        """
        Store extracted text and mark extraction as completed.
        """

        attachment = self.get_attachment_by_id(
            db=db,
            attachment_id=attachment_id,
        )

        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Extracted text cannot be empty.",
            )

        try:
            attachment = (
                self.attachment_repository
                .mark_extraction_as_completed(
                    db=db,
                    attachment=attachment,
                    extracted_text=extracted_text,
                )
            )

            self.audit_repository.create_system_audit_log(
                db=db,
                complaint_id=attachment.complaint_id,
                action="ATTACHMENT_EXTRACTION_COMPLETED",
                description=(
                    "Text extraction completed for attachment."
                ),
                audit_metadata={
                    "attachment_id": str(
                        attachment.attachment_id
                    ),
                    "original_file_name": (
                        attachment.original_file_name
                    ),
                    "extracted_character_count": len(
                        extracted_text
                    ),
                },
            )

            db.commit()
            db.refresh(attachment)

            return attachment

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail="Failed to save extracted attachment text.",
            ) from exc

    def mark_extraction_as_failed(
        self,
        db: Session,
        attachment_id: UUID,
        error_message: str,
    ) -> ComplaintAttachment:
        """
        Mark attachment extraction as failed.
        """

        attachment = self.get_attachment_by_id(
            db=db,
            attachment_id=attachment_id,
        )

        try:
            attachment = (
                self.attachment_repository
                .mark_extraction_as_failed(
                    db=db,
                    attachment=attachment,
                    error_message=error_message,
                )
            )

            self.audit_repository.create_system_audit_log(
                db=db,
                complaint_id=attachment.complaint_id,
                action="ATTACHMENT_EXTRACTION_FAILED",
                description=error_message,
                audit_metadata={
                    "attachment_id": str(
                        attachment.attachment_id
                    ),
                    "original_file_name": (
                        attachment.original_file_name
                    ),
                },
            )

            db.commit()
            db.refresh(attachment)

            return attachment

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Failed to record attachment extraction failure."
                ),
            ) from exc

    def delete_attachment(
        self,
        db: Session,
        attachment_id: UUID,
    ) -> None:
        """
        Delete an attachment record and its physical file.

        Attachments belonging to committed complaints cannot
        be deleted.
        """

        attachment = self.get_attachment_by_id(
            db=db,
            attachment_id=attachment_id,
        )

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=attachment.complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        if complaint.is_committed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Attachments cannot be deleted from a "
                    "committed complaint."
                ),
            )

        file_path = Path(attachment.storage_path)
        original_file_name = attachment.original_file_name
        attachment_uuid = attachment.attachment_id

        try:
            self.attachment_repository.delete_attachment(
                db=db,
                attachment=attachment,
            )

            self.audit_repository.create_user_audit_log(
                db=db,
                complaint_id=complaint.complaint_id,
                action="ATTACHMENT_DELETED",
                description=(
                    "A complaint attachment was deleted."
                ),
                audit_metadata={
                    "attachment_id": str(attachment_uuid),
                    "original_file_name": original_file_name,
                },
            )

            db.commit()

            self._remove_file_if_exists(file_path)

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail="Failed to delete attachment.",
            ) from exc

    def _remove_file_if_exists(
        self,
        file_path: Path,
    ) -> None:
        """
        Remove a physical file when it exists.

        File deletion errors are intentionally ignored during cleanup
        so that the original database or upload exception is preserved.
        """

        try:
            if file_path.exists() and file_path.is_file():
                os.remove(file_path)
        except OSError:
            pass


    def extract_attachment_text(
        self,
        db: Session,
        attachment_id: UUID,
    ) -> ComplaintAttachment:
        """
        Extract readable text from an uploaded attachment.

        The extracted text is stored on the attachment and copied to the
        complaint so that the AI workflow can process it.
        """

        attachment = self.get_attachment_by_id(
            db=db,
            attachment_id=attachment_id,
        )

        complaint = self.complaint_repository.get_complaint_by_id(
            db=db,
            complaint_id=attachment.complaint_id,
        )

        if complaint is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Complaint not found.",
            )

        try:
            attachment.extraction_status = "PROCESSING"
            attachment.extraction_error = None

            db.flush()

            extracted_text = extract_text_from_file(
                storage_path=attachment.storage_path,
                file_extension=attachment.file_extension,
            )

            attachment.extracted_text = extracted_text
            attachment.extraction_status = "COMPLETED"
            attachment.extraction_error = None

            complaint.raw_complaint_text = extracted_text

            if attachment.file_extension.lower() == ".pdf":
                complaint.input_type = "PDF"
            else:
                complaint.input_type = "DOCUMENT"

            self.audit_repository.create_system_audit_log(
                db=db,
                complaint_id=complaint.complaint_id,
                action="ATTACHMENT_EXTRACTION_COMPLETED",
                description="Text was extracted from the uploaded attachment.",
                audit_metadata={
                    "attachment_id": str(attachment.attachment_id),
                    "original_file_name": attachment.original_file_name,
                    "file_extension": attachment.file_extension,
                    "extracted_character_count": len(extracted_text),
                },
            )

            db.commit()
            db.refresh(attachment)

            return attachment

        except HTTPException as exc:
            db.rollback()

            try:
                attachment = self.get_attachment_by_id(
                    db=db,
                    attachment_id=attachment_id,
                )

                attachment.extraction_status = "FAILED"
                attachment.extraction_error = str(exc.detail)

                db.commit()
            except SQLAlchemyError:
                db.rollback()

            raise

        except SQLAlchemyError as exc:
            db.rollback()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save extracted attachment text.",
            ) from exc

        except Exception as exc:
            db.rollback()

            try:
                attachment = self.get_attachment_by_id(
                    db=db,
                    attachment_id=attachment_id,
                )

                attachment.extraction_status = "FAILED"
                attachment.extraction_error = str(exc)

                db.commit()
            except SQLAlchemyError:
                db.rollback()

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to extract text from the uploaded document.",
            ) from exc
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


AttachmentDocumentType = Literal[
    "COMPLAINT_PDF",
    "EMAIL",
    "IMAGE",
    "DOCX",
    "TXT",
    "OTHER",
]

AttachmentExtractionStatus = Literal[
    "PENDING",
    "PROCESSING",
    "COMPLETED",
    "FAILED",
]


# -----------------------
# Attachment Metadata
# -----------------------
class AttachmentCreate(BaseModel):
    """
    Internal schema used after an uploaded file has been saved.
    """

    complaint_id: UUID

    original_file_name: str = Field(
        min_length=1,
        max_length=255,
    )

    stored_file_name: str | None = Field(
        default=None,
        max_length=255,
    )

    storage_path: str

    mime_type: str | None = Field(
        default=None,
        max_length=150,
    )

    file_extension: str | None = Field(
        default=None,
        max_length=20,
    )

    file_size_bytes: int | None = Field(
        default=None,
        ge=0,
    )

    document_type: AttachmentDocumentType = "OTHER"


# -----------------------
# Extraction Update
# -----------------------
class AttachmentExtractionUpdate(BaseModel):
    extracted_text: str | None = None

    extraction_status: AttachmentExtractionStatus

    extraction_error: str | None = None


# -----------------------
# Attachment Response
# -----------------------
class AttachmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    attachment_id: UUID
    complaint_id: UUID

    original_file_name: str
    stored_file_name: str | None
    storage_path: str

    mime_type: str | None
    file_extension: str | None
    file_size_bytes: int | None

    document_type: AttachmentDocumentType

    extracted_text: str | None
    extraction_status: AttachmentExtractionStatus
    extraction_error: str | None

    uploaded_at: datetime


# -----------------------
# Upload Response
# -----------------------
class AttachmentUploadResponse(BaseModel):
    attachment_id: UUID
    complaint_id: UUID

    original_file_name: str

    document_type: AttachmentDocumentType
    extraction_status: AttachmentExtractionStatus

    message: str
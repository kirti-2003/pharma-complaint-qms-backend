from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


ComplaintInputType = Literal[
    "TEXT",
    "EMAIL",
    "PDF",
    "IMAGE",
    "DOCUMENT",
]

ComplaintStatus = Literal[
    "PENDING_TRIAGE",
    "PROCESSING",
    "NEEDS_INFORMATION",
    "READY_TO_COMMIT",
    "COMMITTED",
    "PROCESSING_FAILED",
]

ComplaintSeverity = Literal[
    "MINOR",
    "MAJOR",
    "CRITICAL",
]


# -----------------------
# Shared Complaint Fields
# -----------------------
class ComplaintBase(BaseModel):
    complaint_source: str | None = Field(
        default=None,
        max_length=100,
    )

    customer_name: str | None = Field(
        default=None,
        max_length=255,
    )

    product_name: str | None = Field(
        default=None,
        max_length=255,
    )

    product_strength_grade: str | None = Field(
        default=None,
        max_length=100,
    )

    batch_lot_number: str | None = Field(
        default=None,
        max_length=100,
    )

    affected_quantity_text: str | None = Field(
        default=None,
        max_length=100,
    )

    affected_quantity_value: Decimal | None = None

    affected_quantity_unit: str | None = Field(
        default=None,
        max_length=50,
    )

    manufacturing_date_text: str | None = Field(
        default=None,
        max_length=50,
    )

    manufacturing_date: date | None = None

    expiry_date_text: str | None = Field(
        default=None,
        max_length=50,
    )

    expiry_date: date | None = None

    originating_site_block: str | None = Field(
        default=None,
        max_length=150,
    )

    impacted_non_product_materials: str | None = Field(
        default=None,
        max_length=255,
    )

    complaint_category: str | None = Field(
        default=None,
        max_length=255,
    )

    complaint_description: str | None = None

    suggested_severity: ComplaintSeverity | None = None

    suggested_next_action: str | None = None

    initial_risk_assessment: str | None = None


# -----------------------
# Create Complaint
# -----------------------
class ComplaintCreate(ComplaintBase):
    """
    Request schema used to create a draft complaint.

    A complaint may initially contain only raw text. The remaining
    structured fields can be populated later by the AI workflow.
    """

    raw_complaint_text: str | None = None

    input_type: ComplaintInputType = "TEXT"


# -----------------------
# Create Text Complaint
# -----------------------
class TextComplaintCreate(BaseModel):
    """
    Simplified request used when the user pastes complaint text.
    """

    raw_complaint_text: str = Field(
        min_length=1,
        description="Original complaint text submitted by the user.",
    )

    complaint_source: str | None = Field(
        default=None,
        max_length=100,
    )

    customer_name: str | None = Field(
        default=None,
        max_length=255,
    )

    input_type: Literal["TEXT", "EMAIL"] = "TEXT"


# -----------------------
# Update Complaint
# -----------------------
class ComplaintUpdate(BaseModel):
    """
    Request schema used for partial complaint updates.

    Every field is optional because PATCH requests may update
    only one or a few fields.
    """

    complaint_source: str | None = Field(
        default=None,
        max_length=100,
    )

    customer_name: str | None = Field(
        default=None,
        max_length=255,
    )

    product_name: str | None = Field(
        default=None,
        max_length=255,
    )

    product_strength_grade: str | None = Field(
        default=None,
        max_length=100,
    )

    batch_lot_number: str | None = Field(
        default=None,
        max_length=100,
    )

    affected_quantity_text: str | None = Field(
        default=None,
        max_length=100,
    )

    affected_quantity_value: Decimal | None = None

    affected_quantity_unit: str | None = Field(
        default=None,
        max_length=50,
    )

    manufacturing_date_text: str | None = Field(
        default=None,
        max_length=50,
    )

    manufacturing_date: date | None = None

    expiry_date_text: str | None = Field(
        default=None,
        max_length=50,
    )

    expiry_date: date | None = None

    originating_site_block: str | None = Field(
        default=None,
        max_length=150,
    )

    impacted_non_product_materials: str | None = Field(
        default=None,
        max_length=255,
    )

    complaint_category: str | None = Field(
        default=None,
        max_length=255,
    )

    complaint_description: str | None = None

    suggested_severity: ComplaintSeverity | None = None

    suggested_next_action: str | None = None

    initial_risk_assessment: str | None = None

    raw_complaint_text: str | None = None

    status: ComplaintStatus | None = None


# -----------------------
# Status Update
# -----------------------
class ComplaintStatusUpdate(BaseModel):
    status: ComplaintStatus


# -----------------------
# Commit Complaint
# -----------------------
class ComplaintCommitRequest(BaseModel):
    """
    Request used to commit a completed complaint to the QMS ledger.
    """

    confirm: bool = Field(
        description="Must be true to commit the complaint.",
    )


# -----------------------
# Complaint List Response
# -----------------------
class ComplaintListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    complaint_id: UUID
    complaint_number: str | None
    customer_name: str | None
    product_name: str | None
    batch_lot_number: str | None
    complaint_category: str | None
    suggested_severity: ComplaintSeverity | None
    input_type: ComplaintInputType
    status: ComplaintStatus
    is_committed: bool
    created_at: datetime
    updated_at: datetime


# -----------------------
# Full Complaint Response
# -----------------------
class ComplaintResponse(ComplaintBase):
    model_config = ConfigDict(from_attributes=True)

    complaint_id: UUID
    complaint_number: str | None

    raw_complaint_text: str | None

    input_type: ComplaintInputType
    status: ComplaintStatus

    is_committed: bool
    committed_at: datetime | None

    created_at: datetime
    updated_at: datetime
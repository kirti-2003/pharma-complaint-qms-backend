from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ExtractedComplaintOutput(BaseModel):
    """
    Structured information extracted from complaint text,
    uploaded documents, emails, or images.

    Fields should remain None when the information is not
    explicitly available in the complaint.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    complainant_name: str | None = Field(
        default=None,
        description="Name of the person or organization submitting the complaint.",
    )

    complainant_email: str | None = Field(
        default=None,
        description="Email address of the complainant.",
    )

    complainant_phone: str | None = Field(
        default=None,
        description="Phone number of the complainant.",
    )

    customer_type: str | None = Field(
        default=None,
        description=(
            "Type of complainant, such as customer, distributor, hospital, "
            "pharmacy, regulatory authority, or internal employee."
        ),
    )

    product_name: str | None = Field(
        default=None,
        description="Name of the pharmaceutical product.",
    )

    product_strength_grade: str | None = Field(
        default=None,
        description="Product strength, grade, dosage, or concentration.",
    )

    dosage_form: str | None = Field(
        default=None,
        description="Dosage form such as tablet, capsule, injection, syrup, or API.",
    )

    batch_lot_number: str | None = Field(
        default=None,
        description="Batch number or lot number associated with the product.",
    )

    manufacturing_date: str | None = Field(
        default=None,
        description="Manufacturing date exactly as mentioned in the complaint.",
    )

    expiry_date: str | None = Field(
        default=None,
        description="Expiry date exactly as mentioned in the complaint.",
    )

    complaint_date: str | None = Field(
        default=None,
        description="Date on which the complaint was received or reported.",
    )

    incident_date: str | None = Field(
        default=None,
        description="Date on which the reported incident occurred.",
    )

    country: str | None = Field(
        default=None,
        description="Country associated with the complaint or incident.",
    )

    complaint_description: str | None = Field(
        default=None,
        description="Clear summary of the complaint described by the complainant.",
    )

    observed_issue: str | None = Field(
        default=None,
        description="The specific defect, failure, reaction, or issue observed.",
    )

    quantity_affected: str | None = Field(
        default=None,
        description="Quantity of product reportedly affected.",
    )

    patient_involved: bool | None = Field(
        default=None,
        description="Whether a patient was involved in the reported complaint.",
    )

    adverse_event_reported: bool | None = Field(
        default=None,
        description="Whether an adverse event or health impact was reported.",
    )

    patient_outcome: str | None = Field(
        default=None,
        description="Reported patient outcome, when available.",
    )

    storage_conditions: str | None = Field(
        default=None,
        description="Storage or transportation conditions mentioned in the complaint.",
    )

    supporting_evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence mentioned in the complaint, such as photographs, samples, "
            "invoices, labels, or medical reports."
        ),
    )

    source_reference: str | None = Field(
        default=None,
        description="Reference number, email subject, document ID, or external identifier.",
    )

    additional_information: dict[str, Any] = Field(
        default_factory=dict,
        description="Other relevant information that does not fit the defined fields.",
    )


class ComplaintClassificationOutput(BaseModel):
    """
    Structured classification produced after complaint extraction
    and required-field validation.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    complaint_category: str | None = Field(
        default=None,
        description=(
            "High-level complaint category such as product quality, packaging, "
            "labeling, delivery, documentation, or adverse event."
        ),
    )

    complaint_subcategory: str | None = Field(
        default=None,
        description="More specific complaint classification.",
    )

    complaint_type: str | None = Field(
        default=None,
        description=(
            "Complaint type such as API, finished dosage form, packaging, "
            "medical device, service, or logistics."
        ),
    )

    suggested_severity: Literal[
        "MINOR",
        "MAJOR",
        "CRITICAL",
    ] | None = Field(
        default=None,
        description=(
            "Suggested pharmaceutical complaint severity: "
            "MINOR, MAJOR, or CRITICAL."
        ),
    )

    is_quality_complaint: bool = Field(
        default=False,
        description="Whether the complaint relates to product quality.",
    )

    is_adverse_event: bool = Field(
        default=False,
        description="Whether the complaint contains a possible adverse event.",
    )

    requires_immediate_attention: bool = Field(
        default=False,
        description="Whether immediate human review is recommended.",
    )

    classification_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence score between 0 and 1.",
    )

    classification_reasoning: str | None = Field(
        default=None,
        description="Brief explanation supporting the classification.",
    )


class RiskAssessmentOutput(BaseModel):
    """
    Initial AI-generated risk assessment.

    This is a preliminary recommendation and should not be treated
    as the final regulatory or quality decision.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    risk_level: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ] | None = Field(
        default=None,
        description="Initial risk level.",
    )

    patient_safety_risk: str | None = Field(
        default=None,
        description="Assessment of possible patient safety impact.",
    )

    product_quality_risk: str | None = Field(
        default=None,
        description="Assessment of possible product quality impact.",
    )

    regulatory_risk: str | None = Field(
        default=None,
        description="Assessment of possible regulatory impact.",
    )

    business_risk: str | None = Field(
        default=None,
        description="Assessment of reputational, operational, or commercial impact.",
    )

    requires_escalation: bool = Field(
        default=False,
        description="Whether the complaint should be escalated immediately.",
    )

    requires_sample_collection: bool = Field(
        default=False,
        description="Whether collecting the affected product sample is recommended.",
    )

    requires_batch_investigation: bool = Field(
        default=False,
        description="Whether a batch-level investigation is recommended.",
    )

    requires_adverse_event_review: bool = Field(
        default=False,
        description="Whether pharmacovigilance or adverse-event review is recommended.",
    )

    suggested_next_action: str | None = Field(
        default=None,
        description="Recommended immediate next action.",
    )

    recommended_actions: list[str] = Field(
        default_factory=list,
        description="List of recommended investigation or follow-up actions.",
    )

    risk_factors: list[str] = Field(
        default_factory=list,
        description="Factors that contributed to the risk assessment.",
    )

    risk_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence score between 0 and 1.",
    )

    risk_reasoning: str | None = Field(
        default=None,
        description="Brief explanation supporting the risk assessment.",
    )


class ChatCorrectionOutput(BaseModel):
    """
    Structured result returned when the user corrects complaint
    information through the AI chat interface.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    user_intent: Literal[
        "UPDATE_FIELD",
        "PROVIDE_INFORMATION",
        "ASK_QUESTION",
        "CONFIRM_INFORMATION",
        "UNKNOWN",
    ] = Field(
        default="UNKNOWN",
        description="Detected purpose of the user's chat message.",
    )

    updated_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Complaint fields that should be updated.",
    )

    unchanged_fields: list[str] = Field(
        default_factory=list,
        description="Fields mentioned by the user that do not require an update.",
    )

    rejected_updates: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Rejected complaint updates with their supplied values "
            "and rejection reasons."
        ),
    )

    clarification_required: bool = Field(
        default=False,
        description="Whether additional clarification is required from the user.",
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question that should be asked when clarification is required.",
    )

    assistant_message: str | None = Field(
        default=None,
        description="Short response to show in the complaint chat interface.",
    )


class FieldValidationOutput(BaseModel):
    """
    Result of validating extracted complaint fields.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    is_complete: bool = Field(
        default=False,
        description="Whether all required complaint fields are available.",
    )

    missing_fields: list[str] = Field(
        default_factory=list,
        description="Required fields that are currently missing.",
    )

    invalid_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Invalid fields mapped to their validation errors.",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking data-quality warnings.",
    )


class ComplaintFinalOutput(BaseModel):
    """
    Final structured output returned by the LangGraph workflow.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    extracted_fields: ExtractedComplaintOutput | None = Field(
        default=None,
        description="Complaint information extracted or updated by the workflow.",
    )

    validation: FieldValidationOutput | None = Field(
        default=None,
        description="Result of deterministic complaint-field validation.",
    )

    classification: ComplaintClassificationOutput | None = Field(
        default=None,
        description="AI-generated complaint classification.",
    )

    risk_assessment: RiskAssessmentOutput | None = Field(
        default=None,
        description="AI-generated preliminary risk assessment.",
    )

    processing_status: Literal[
        "COMPLETED",
        "WAITING_FOR_USER",
        "FAILED",
    ] = Field(
        default="COMPLETED",
        description="Final LangGraph processing result.",
    )

    missing_fields: list[str] = Field(
        default_factory=list,
        description="Required complaint fields that remain missing.",
    )

    invalid_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Complaint fields that failed validation.",
    )

    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Non-blocking validation warnings.",
    )

    clarification_required: bool = Field(
        default=False,
        description="Whether more information is required from the user.",
    )

    clarification_question: str | None = Field(
        default=None,
        description="Question to display when more information is required.",
    )

    assistant_message: str | None = Field(
        default=None,
        description="Final user-facing workflow message.",
    )

    error_message: str | None = Field(
        default=None,
        description="Workflow error message when processing fails.",
    )
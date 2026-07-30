from typing import Any, TypedDict


class ComplaintGraphState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.

    Each node reads the values it needs and returns only the
    fields it updates.
    """

    # ---------------------------------------------------------
    # Workflow identification
    # ---------------------------------------------------------

    complaint_id: str
    ai_run_id: str

    trigger_type: str
    input_type: str

    # Possible trigger types:
    # TEXT_SUBMISSION
    # FILE_UPLOAD
    # CHAT_CORRECTION
    # REANALYSIS

    # ---------------------------------------------------------
    # Input data
    # ---------------------------------------------------------

    raw_text: str

    chat_message: str | None

    attachment_id: str | None

    attachment_ids: list[str]

    existing_fields: dict[str, Any]

    # ---------------------------------------------------------
    # Extraction result
    # ---------------------------------------------------------

    extracted_fields: dict[str, Any]

    extraction_confidence: float | None

    # ---------------------------------------------------------
    # Field validation result
    # ---------------------------------------------------------

    is_complete: bool

    missing_fields: list[str]

    invalid_fields: dict[str, str]

    validation_warnings: list[str]

    # ---------------------------------------------------------
    # Complaint classification result
    # ---------------------------------------------------------

    classification_result: dict[str, Any]

    # Expected structure:
    # {
    #     "complaint_category": "...",
    #     "complaint_subcategory": "...",
    #     "complaint_type": "...",
    #     "suggested_severity": "...",
    #     "is_quality_complaint": True,
    #     "is_adverse_event": False,
    #     "requires_immediate_attention": False,
    #     "classification_confidence": 0.95,
    #     "classification_reasoning": "..."
    # }

    # ---------------------------------------------------------
    # Risk assessment result
    # ---------------------------------------------------------

    risk_assessment_result: dict[str, Any]

    # Expected structure:
    # {
    #     "risk_level": "MEDIUM",
    #     "patient_safety_risk": "...",
    #     "product_quality_risk": "...",
    #     "regulatory_risk": "...",
    #     "business_risk": "...",
    #     "requires_escalation": False,
    #     "requires_sample_collection": True,
    #     "requires_batch_investigation": True,
    #     "requires_adverse_event_review": False,
    #     "suggested_next_action": "...",
    #     "recommended_actions": [],
    #     "risk_factors": [],
    #     "risk_confidence": 0.88,
    #     "risk_reasoning": "..."
    # }

    # ---------------------------------------------------------
    # Chat correction result
    # ---------------------------------------------------------

    chat_correction_result: dict[str, Any]

    updated_fields: dict[str, Any]

    rejected_updates: dict[str, str]

    clarification_required: bool

    clarification_question: str | None

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    final_output: dict[str, Any]

    assistant_message: str | None

    processing_status: str

    # Possible processing statuses:
    # STARTED
    # PROCESSING
    # WAITING_FOR_USER
    # COMPLETED
    # FAILED

    # ---------------------------------------------------------
    # Token usage
    # ---------------------------------------------------------

    prompt_tokens: int

    completion_tokens: int

    total_tokens: int

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    model_name: str | None

    # ---------------------------------------------------------
    # LangGraph execution information
    # ---------------------------------------------------------

    langgraph_thread_id: str | None

    langgraph_run_id: str | None

    current_node: str | None

    completed_nodes: list[str]

    # ---------------------------------------------------------
    # Error handling
    # ---------------------------------------------------------

    has_error: bool

    error_node: str | None

    error_message: str | None

    error_details: dict[str, Any]
from typing import Any

from app.ai.graph.state import ComplaintGraphState


def create_initial_complaint_state(
    complaint_id: str,
    trigger_type: str,
    raw_text: str = "",
    ai_run_id: str | None = None,
    input_type: str = "TEXT",
    chat_message: str | None = None,
    attachment_id: str | None = None,
    attachment_ids: list[str] | None = None,
    existing_fields: dict[str, Any] | None = None,
) -> ComplaintGraphState:
    """
    Create the initial LangGraph state for complaint processing.
    """

    return ComplaintGraphState(
        complaint_id=complaint_id,
        ai_run_id=ai_run_id or "",
        trigger_type=trigger_type,
        input_type=input_type,
        raw_text=raw_text,
        chat_message=chat_message,
        attachment_id=attachment_id,
        attachment_ids=attachment_ids or [],
        existing_fields=existing_fields or {},
        extracted_fields={},
        extraction_confidence=None,
        is_complete=False,
        missing_fields=[],
        invalid_fields={},
        validation_warnings=[],
        classification_result={},
        risk_assessment_result={},
        chat_correction_result={},
        updated_fields={},
        rejected_updates={},
        clarification_required=False,
        clarification_question=None,
        final_output={},
        assistant_message=None,
        processing_status="STARTED",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        model_name=None,
        langgraph_thread_id=None,
        langgraph_run_id=None,
        current_node=None,
        completed_nodes=[],
        has_error=False,
        error_node=None,
        error_message=None,
        error_details={},
    )
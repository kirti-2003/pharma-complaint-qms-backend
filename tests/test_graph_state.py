from app.ai.graph.state import ComplaintGraphState
from app.ai.graph.state_factory import create_initial_complaint_state


def test_create_initial_complaint_state():
    state = create_initial_complaint_state(
        complaint_id="complaint-001",
        ai_run_id="ai-run-001",
        trigger_type="TEXT_SUBMISSION",
        input_type="TEXT",
        raw_text="The tablets were broken inside the blister pack.",
    )

    assert state["complaint_id"] == "complaint-001"
    assert state["ai_run_id"] == "ai-run-001"
    assert state["trigger_type"] == "TEXT_SUBMISSION"
    assert state["input_type"] == "TEXT"

    assert state["raw_text"] == (
        "The tablets were broken inside the blister pack."
    )

    assert state["processing_status"] == "STARTED"

    assert state["extracted_fields"] == {}
    assert state["missing_fields"] == []
    assert state["classification_result"] == {}
    assert state["risk_assessment_result"] == {}

    assert state["prompt_tokens"] == 0
    assert state["completion_tokens"] == 0
    assert state["total_tokens"] == 0

    assert state["has_error"] is False
    assert state["error_message"] is None


def test_state_can_be_partially_created():
    state: ComplaintGraphState = {
        "complaint_id": "complaint-002",
        "trigger_type": "FILE_UPLOAD",
        "raw_text": "Complaint extracted from uploaded PDF.",
    }

    assert state["complaint_id"] == "complaint-002"
    assert state["trigger_type"] == "FILE_UPLOAD"
    assert "risk_assessment_result" not in state


def test_initial_state_with_existing_fields():
    state = create_initial_complaint_state(
        complaint_id="complaint-003",
        trigger_type="CHAT_CORRECTION",
        chat_message="The batch number should be BT-2026-009.",
        existing_fields={
            "product_name": "Paracetamol 500 mg",
            "batch_lot_number": "BT-2026-001",
        },
    )

    assert state["trigger_type"] == "CHAT_CORRECTION"

    assert state["chat_message"] == (
        "The batch number should be BT-2026-009."
    )

    assert state["existing_fields"]["product_name"] == (
        "Paracetamol 500 mg"
    )

    assert state["existing_fields"]["batch_lot_number"] == (
        "BT-2026-001"
    )
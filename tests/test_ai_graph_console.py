import json
from typing import Any

from app.ai.graph import (
    complaint_graph,
    create_initial_complaint_state,
)


def print_section(
    title: str,
    data: Any,
) -> None:
    """
    Print a formatted console section.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    print(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def run_initial_graph_test() -> dict[str, Any]:
    """
    Test the complete initial complaint workflow.
    """

    complaint_text = """
    Apollo Pharmacy reported 12 discolored capsules in a sealed bottle
    of Amoxicillin Capsules 500 mg.

    Batch number: AMX240602.
    Manufacturing date: March 2026.
    Expiry date: February 2028.

    The pharmacy requested an investigation and replacement.
    There was no patient injury or adverse event reported.
    """

    initial_state = (
        create_initial_complaint_state(
            complaint_id=(
                "graph-test-complaint"
            ),
            ai_run_id=(
                "graph-test-initial-run"
            ),
            trigger_type=(
                "TEXT_SUBMISSION"
            ),
            input_type="TEXT",
            raw_text=complaint_text,
        )
    )

    print_section(
        "INITIAL GRAPH STATE",
        initial_state,
    )

    result = complaint_graph.invoke(
        initial_state
    )

    print_section(
        "INITIAL GRAPH RESULT",
        result,
    )

    assert (
        result.get("has_error")
        is False
    )

    assert (
        result.get(
            "processing_status"
        )
        == "COMPLETED"
    )

    assert (
        result.get("current_node")
        == "build_final_output"
    )

    assert result.get(
        "final_output"
    )

    assert result.get(
        "classification_result"
    )

    assert result.get(
        "risk_assessment_result"
    )

    extracted_fields = result.get(
        "extracted_fields",
        {},
    )

    assert (
        extracted_fields.get(
            "quantity_affected"
        )
        == "12"
    )

    final_output = result.get(
        "final_output",
        {},
    )

    assert (
        final_output.get(
            "processing_status"
        )
        == "COMPLETED"
    )

    print(
        "\nInitial LangGraph workflow "
        "completed successfully."
    )

    return result


def run_chat_graph_test(
    previous_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Test chat correction and downstream reprocessing.
    """

    chat_state = dict(
        previous_result
    )

    chat_state.update(
        {
            "ai_run_id": (
                "graph-test-chat-run"
            ),
            "trigger_type": (
                "CHAT_CORRECTION"
            ),
            "input_type": "TEXT",
            "chat_message": (
                "Correction: the affected quantity was "
                "15 capsules, not 12. The product was "
                "stored at room temperature."
            ),
            "has_error": False,
            "error_node": None,
            "error_message": None,
            "error_details": {},
            "processing_status": (
                "PROCESSING"
            ),
        }
    )

    print_section(
        "CHAT GRAPH INPUT STATE",
        chat_state,
    )

    result = complaint_graph.invoke(
        chat_state
    )

    print_section(
        "CHAT GRAPH RESULT",
        result,
    )

    assert (
        result.get("has_error")
        is False
    )

    assert (
        result.get(
            "processing_status"
        )
        == "COMPLETED"
    )

    assert (
        result.get("current_node")
        == "build_final_output"
    )

    extracted_fields = result.get(
        "extracted_fields",
        {},
    )

    assert (
        extracted_fields.get(
            "quantity_affected"
        )
        == "15"
    )

    assert (
        extracted_fields.get(
            "storage_conditions"
        )
        == "room temperature"
    )

    assert result.get(
        "classification_result"
    )

    assert result.get(
        "risk_assessment_result"
    )

    assert result.get(
        "final_output"
    )

    print(
        "\nChat-update LangGraph workflow "
        "completed successfully."
    )

    return result


def run_incomplete_complaint_test() -> dict[str, Any]:
    """
    Test routing when required complaint fields are missing.
    """

    complaint_text = """
    Some capsules look unusual.
    Please investigate.
    """

    state = (
        create_initial_complaint_state(
            complaint_id=(
                "graph-test-incomplete"
            ),
            ai_run_id=(
                "graph-test-incomplete-run"
            ),
            trigger_type=(
                "TEXT_SUBMISSION"
            ),
            input_type="TEXT",
            raw_text=complaint_text,
        )
    )

    print_section(
        "INCOMPLETE GRAPH INPUT STATE",
        state,
    )

    result = complaint_graph.invoke(
        state
    )

    print_section(
        "INCOMPLETE COMPLAINT RESULT",
        result,
    )

    assert (
        result.get("has_error")
        is False
    )

    assert (
        result.get(
            "processing_status"
        )
        == "WAITING_FOR_USER"
    )

    assert (
        result.get(
            "clarification_required"
        )
        is True
    )

    assert result.get(
        "missing_fields"
    )

    assert not result.get(
        "classification_result"
    )

    assert not result.get(
        "risk_assessment_result"
    )

    final_output = result.get(
        "final_output",
        {},
    )

    assert (
        final_output.get(
            "processing_status"
        )
        == "WAITING_FOR_USER"
    )

    assert (
        final_output.get(
            "clarification_required"
        )
        is True
    )

    assert final_output.get(
        "assistant_message"
    )

    print(
        "\nIncomplete complaint routing "
        "completed successfully."
    )

    return result


def run_invalid_chat_test() -> dict[str, Any]:
    """
    Test structured error handling when a chat workflow starts
    without existing complaint fields.
    """

    state = (
        create_initial_complaint_state(
            complaint_id=(
                "graph-test-invalid-chat"
            ),
            ai_run_id=(
                "graph-test-invalid-chat-run"
            ),
            trigger_type=(
                "CHAT_CORRECTION"
            ),
            input_type="TEXT",
            raw_text=None,
        )
    )

    state["chat_message"] = (
        "Change the affected quantity to 20."
    )

    print_section(
        "INVALID CHAT INPUT STATE",
        state,
    )

    result = complaint_graph.invoke(
        state
    )

    print_section(
        "INVALID CHAT RESULT",
        result,
    )

    assert (
        result.get("has_error")
        is True
    )

    assert (
        result.get(
            "processing_status"
        )
        == "FAILED"
    )

    assert (
        result.get("current_node")
        == "build_final_output"
    )

    assert result.get(
        "error_message"
    )

    final_output = result.get(
        "final_output",
        {},
    )

    assert final_output

    assert (
        final_output.get(
            "processing_status"
        )
        == "FAILED"
    )

    assert final_output.get(
        "error_message"
    )

    assert final_output.get(
        "assistant_message"
    )

    print(
        "\nInvalid chat error routing "
        "completed successfully."
    )

    return result


def run_graph_tests() -> None:
    """
    Run all graph console tests.
    """

    initial_result = (
        run_initial_graph_test()
    )

    final_result = (
        run_chat_graph_test(
            initial_result
        )
    )

    run_incomplete_complaint_test()

    run_invalid_chat_test()

    print_section(
        "FINAL SUCCESSFUL GRAPH STATE",
        final_result,
    )

    print(
        "\nAll LangGraph console tests "
        "completed successfully."
    )


if __name__ == "__main__":
    run_graph_tests()
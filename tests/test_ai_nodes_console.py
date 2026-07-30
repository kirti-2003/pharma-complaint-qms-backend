import json
from typing import Any

from app.ai.graph.state_factory import (
    create_initial_complaint_state,
)
from app.ai.nodes.assess_risk import (
    assess_risk_node,
)
from app.ai.nodes.build_final_output import (
    build_final_output_node,
)
from app.ai.nodes.classify_complaint import (
    classify_complaint_node,
)
from app.ai.nodes.extract_complaint import (
    extract_complaint_node,
)
from app.ai.nodes.update_from_chat import (
    update_from_chat_node,
)
from app.ai.nodes.validate_fields import (
    validate_fields_node,
)


def merge_state(
    state: dict[str, Any],
    node_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Simulate how LangGraph merges a partial node result
    into the graph state.
    """

    state.update(
        node_result
    )

    return state


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


def run_node(
    state: dict[str, Any],
    node_result: dict[str, Any],
    section_title: str,
    failure_message: str,
) -> bool:
    """
    Print and merge a node result.

    Returns False when the node reports an error.
    """

    print_section(
        section_title,
        node_result,
    )

    merge_state(
        state=state,
        node_result=node_result,
    )

    if state.get("has_error"):
        print(
            f"\n{failure_message}"
        )

        print_section(
            "ERROR STATE",
            state,
        )

        return False

    return True


def run_initial_complaint_flow(
    state: dict[str, Any],
) -> bool:
    """
    Run the initial complaint processing workflow.
    """

    # --------------------------------------------------
    # 1. Extract complaint fields
    # --------------------------------------------------

    extraction_result = (
        extract_complaint_node(
            state
        )
    )

    extraction_success = run_node(
        state=state,
        node_result=extraction_result,
        section_title="EXTRACTION RESULT",
        failure_message=(
            "Extraction failed. Stopping test."
        ),
    )

    if not extraction_success:
        return False

    # --------------------------------------------------
    # 2. Validate extracted fields
    # --------------------------------------------------

    validation_result = (
        validate_fields_node(
            state
        )
    )

    validation_success = run_node(
        state=state,
        node_result=validation_result,
        section_title="VALIDATION RESULT",
        failure_message=(
            "Validation failed. Stopping test."
        ),
    )

    if not validation_success:
        return False

    # When information is missing, build a waiting response.
    if not state.get("is_complete"):
        print(
            "\nComplaint requires clarification."
        )

        final_output_result = (
            build_final_output_node(
                state
            )
        )

        run_node(
            state=state,
            node_result=final_output_result,
            section_title="FINAL OUTPUT RESULT",
            failure_message=(
                "Final output creation failed."
            ),
        )

        return False

    # --------------------------------------------------
    # 3. Classify complaint
    # --------------------------------------------------

    classification_result = (
        classify_complaint_node(
            state
        )
    )

    classification_success = run_node(
        state=state,
        node_result=classification_result,
        section_title="CLASSIFICATION RESULT",
        failure_message=(
            "Classification failed. Stopping test."
        ),
    )

    if not classification_success:
        return False

    # --------------------------------------------------
    # 4. Assess complaint risk
    # --------------------------------------------------

    risk_result = (
        assess_risk_node(
            state
        )
    )

    risk_success = run_node(
        state=state,
        node_result=risk_result,
        section_title="RISK ASSESSMENT RESULT",
        failure_message=(
            "Risk assessment failed. Stopping test."
        ),
    )

    if not risk_success:
        return False

    # --------------------------------------------------
    # 5. Build final output
    # --------------------------------------------------

    final_output_result = (
        build_final_output_node(
            state
        )
    )

    final_output_success = run_node(
        state=state,
        node_result=final_output_result,
        section_title="FINAL OUTPUT RESULT",
        failure_message=(
            "Final output creation failed."
        ),
    )

    if not final_output_success:
        return False

    return True


def run_chat_update_flow(
    state: dict[str, Any],
) -> bool:
    """
    Test updating complaint information through a chat message.

    After the update, validation, classification, risk assessment,
    and final-output generation are run again.
    """

    print_section(
        "STATE BEFORE CHAT UPDATE",
        state,
    )

    state["trigger_type"] = (
        "CHAT_CORRECTION"
    )

    state["input_type"] = (
        "TEXT"
    )

    state["chat_message"] = (
        "Correction: the affected quantity was 15 capsules, "
        "not 12. The product was stored at room temperature."
    )

    # --------------------------------------------------
    # 1. Apply chat correction
    # --------------------------------------------------

    chat_update_result = (
        update_from_chat_node(
            state
        )
    )

    chat_update_success = run_node(
        state=state,
        node_result=chat_update_result,
        section_title="CHAT UPDATE RESULT",
        failure_message=(
            "Chat update failed. Stopping test."
        ),
    )

    if not chat_update_success:
        return False

    # --------------------------------------------------
    # 2. Validate updated complaint fields
    # --------------------------------------------------

    validation_result = (
        validate_fields_node(
            state
        )
    )

    validation_success = run_node(
        state=state,
        node_result=validation_result,
        section_title=(
            "POST-CHAT VALIDATION RESULT"
        ),
        failure_message=(
            "Post-chat validation failed."
        ),
    )

    if not validation_success:
        return False

    if not state.get("is_complete"):
        final_output_result = (
            build_final_output_node(
                state
            )
        )

        run_node(
            state=state,
            node_result=final_output_result,
            section_title=(
                "POST-CHAT FINAL OUTPUT RESULT"
            ),
            failure_message=(
                "Post-chat final output failed."
            ),
        )

        return False

    # --------------------------------------------------
    # 3. Reclassify updated complaint
    # --------------------------------------------------

    classification_result = (
        classify_complaint_node(
            state
        )
    )

    classification_success = run_node(
        state=state,
        node_result=classification_result,
        section_title=(
            "POST-CHAT CLASSIFICATION RESULT"
        ),
        failure_message=(
            "Post-chat classification failed."
        ),
    )

    if not classification_success:
        return False

    # --------------------------------------------------
    # 4. Reassess updated complaint risk
    # --------------------------------------------------

    risk_result = (
        assess_risk_node(
            state
        )
    )

    risk_success = run_node(
        state=state,
        node_result=risk_result,
        section_title="POST-CHAT RISK RESULT",
        failure_message=(
            "Post-chat risk assessment failed."
        ),
    )

    if not risk_success:
        return False

    # --------------------------------------------------
    # 5. Rebuild final output
    # --------------------------------------------------

    final_output_result = (
        build_final_output_node(
            state
        )
    )

    final_output_success = run_node(
        state=state,
        node_result=final_output_result,
        section_title=(
            "POST-CHAT FINAL OUTPUT RESULT"
        ),
        failure_message=(
            "Post-chat final output creation failed."
        ),
    )

    if not final_output_success:
        return False

    return True


def run_ai_nodes() -> None:
    """
    Run the initial complaint flow and the chat-update flow.
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

    state = create_initial_complaint_state(
        complaint_id=(
            "console-test-complaint"
        ),
        ai_run_id=(
            "console-test-ai-run"
        ),
        trigger_type=(
            "TEXT_SUBMISSION"
        ),
        input_type="TEXT",
        raw_text=complaint_text,
    )

    print_section(
        "INITIAL STATE",
        state,
    )

    # --------------------------------------------------
    # Initial complaint workflow
    # --------------------------------------------------

    initial_success = (
        run_initial_complaint_flow(
            state
        )
    )

    print_section(
        "STATE AFTER INITIAL ANALYSIS",
        state,
    )

    if not initial_success:
        print(
            "\nInitial complaint workflow did not complete."
        )

        return

    print(
        "\nInitial complaint workflow completed successfully."
    )

    # --------------------------------------------------
    # Chat correction workflow
    # --------------------------------------------------

    chat_success = (
        run_chat_update_flow(
            state
        )
    )

    print_section(
        "FINAL STATE AFTER CHAT UPDATE",
        state,
    )

    if not chat_success:
        print(
            "\nChat update workflow did not complete."
        )

        return

    print(
        "\nAI node console test completed successfully."
    )


if __name__ == "__main__":
    run_ai_nodes()
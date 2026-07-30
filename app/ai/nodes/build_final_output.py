from typing import Any

from pydantic import ValidationError

from app.ai.graph.state import (
    ComplaintGraphState,
)
from app.ai.schemas.structured_outputs import (
    ComplaintFinalOutput,
)


def _add_completed_node(
    completed_nodes: list[str],
    node_name: str,
) -> list[str]:
    """
    Add a node to completed_nodes without creating duplicates.
    """

    normalized_nodes = list(
        completed_nodes
    )

    if node_name not in normalized_nodes:
        normalized_nodes.append(
            node_name
        )

    return normalized_nodes


def _build_success_message(
    classification_result: dict[str, Any],
    risk_assessment_result: dict[str, Any],
    validation_warnings: list[str],
) -> str:
    """
    Build a concise user-facing completion message.
    """

    category = (
        classification_result.get(
            "complaint_category"
        )
        or "Unclassified"
    )

    severity = (
        classification_result.get(
            "suggested_severity"
        )
        or "Not determined"
    )

    risk_level = (
        risk_assessment_result.get(
            "risk_level"
        )
        or "Not determined"
    )

    message = (
        "Complaint analysis completed successfully. "
        f"Category: {category}. "
        f"Suggested severity: {severity}. "
        f"Risk level: {risk_level}."
    )

    if validation_warnings:
        warning_count = len(
            validation_warnings
        )

        message += (
            f" The complaint contains "
            f"{warning_count} "
            "non-blocking validation warning"
        )

        if warning_count != 1:
            message += "s"

        message += "."

    return message


def _build_waiting_message(
    clarification_question: str | None,
) -> str:
    """
    Build the final message when more information is required.
    """

    if clarification_question:
        return clarification_question

    return (
        "Additional complaint information is required "
        "before processing can continue."
    )


def _build_error_message(
    state: ComplaintGraphState,
) -> str:
    """
    Build the final user-facing error message.
    """

    error_message = state.get(
        "error_message"
    )

    if error_message:
        return (
            "Complaint processing failed. "
            f"{error_message}"
        )

    return (
        "Complaint processing failed due to an "
        "unexpected workflow error."
    )


def _build_validation_result(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Build the validation section of the final output.
    """

    return {
        "is_complete": state.get(
            "is_complete",
            False,
        ),
        "missing_fields": state.get(
            "missing_fields",
            [],
        ),
        "invalid_fields": state.get(
            "invalid_fields",
            {},
        ),
        "warnings": state.get(
            "validation_warnings",
            [],
        ),
    }


def _build_final_output_payload(
    state: ComplaintGraphState,
    assistant_message: str,
    final_status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    """
    Construct the canonical final-output payload.
    """

    extracted_fields = state.get(
        "extracted_fields",
        {},
    )

    classification_result = state.get(
        "classification_result",
        {},
    )

    risk_assessment_result = state.get(
        "risk_assessment_result",
        {},
    )

    missing_fields = state.get(
        "missing_fields",
        [],
    )

    invalid_fields = state.get(
        "invalid_fields",
        {},
    )

    validation_warnings = state.get(
        "validation_warnings",
        [],
    )

    clarification_required = state.get(
        "clarification_required",
        False,
    )

    clarification_question = state.get(
        "clarification_question"
    )

    return {
        "extracted_fields": (
            extracted_fields
            or None
        ),
        "validation": (
            _build_validation_result(
                state
            )
        ),
        "classification": (
            classification_result
            or None
        ),
        "risk_assessment": (
            risk_assessment_result
            or None
        ),
        "processing_status": final_status,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "validation_warnings": (
            validation_warnings
        ),
        "clarification_required": (
            clarification_required
        ),
        "clarification_question": (
            clarification_question
        ),
        "assistant_message": (
            assistant_message
        ),
        "error_message": error_message,
    }


def _validate_final_output(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate the payload against ComplaintFinalOutput.
    """

    validated_output = (
        ComplaintFinalOutput.model_validate(
            payload
        )
    )

    return validated_output.model_dump()


def _build_failed_output(
    state: ComplaintGraphState,
    node_name: str,
) -> dict[str, Any]:
    """
    Build a structured failed final output when an upstream
    node has already reported an error.
    """

    original_error_message = (
        state.get("error_message")
        or "Unknown complaint workflow error."
    )

    assistant_message = (
        _build_error_message(
            state
        )
    )

    final_payload = (
        _build_final_output_payload(
            state=state,
            assistant_message=(
                assistant_message
            ),
            final_status="FAILED",
            error_message=(
                original_error_message
            ),
        )
    )

    final_output = (
        _validate_final_output(
            final_payload
        )
    )

    completed_nodes = (
        _add_completed_node(
            completed_nodes=state.get(
                "completed_nodes",
                [],
            ),
            node_name=node_name,
        )
    )

    return {
        "final_output": final_output,
        "assistant_message": (
            assistant_message
        ),
        "current_node": node_name,
        "completed_nodes": (
            completed_nodes
        ),
        "processing_status": "FAILED",

        # Keep the original upstream error information.
        "has_error": True,
        "error_node": (
            state.get("error_node")
            or node_name
        ),
        "error_message": (
            original_error_message
        ),
        "error_details": state.get(
            "error_details",
            {},
        ),
    }


def build_final_output_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Build the final structured response for the complaint workflow.

    This deterministic node does not call the LLM and therefore
    does not consume additional tokens.
    """

    node_name = "build_final_output"

    try:
        # An upstream node failed.
        if state.get("has_error"):
            return _build_failed_output(
                state=state,
                node_name=node_name,
            )

        clarification_required = state.get(
            "clarification_required",
            False,
        )

        is_complete = state.get(
            "is_complete",
            False,
        )

        if (
            clarification_required
            or not is_complete
        ):
            final_status = (
                "WAITING_FOR_USER"
            )

            assistant_message = (
                _build_waiting_message(
                    state.get(
                        "clarification_question"
                    )
                )
            )

        else:
            classification_result = (
                state.get(
                    "classification_result",
                    {},
                )
            )

            risk_assessment_result = (
                state.get(
                    "risk_assessment_result",
                    {},
                )
            )

            if not classification_result:
                raise ValueError(
                    "Classification result is required "
                    "before building the final output."
                )

            if not risk_assessment_result:
                raise ValueError(
                    "Risk assessment result is required "
                    "before building the final output."
                )

            final_status = "COMPLETED"

            assistant_message = (
                _build_success_message(
                    classification_result=(
                        classification_result
                    ),
                    risk_assessment_result=(
                        risk_assessment_result
                    ),
                    validation_warnings=(
                        state.get(
                            "validation_warnings",
                            [],
                        )
                    ),
                )
            )

        final_payload = (
            _build_final_output_payload(
                state=state,
                assistant_message=(
                    assistant_message
                ),
                final_status=final_status,
                error_message=None,
            )
        )

        final_output = (
            _validate_final_output(
                final_payload
            )
        )

        completed_nodes = (
            _add_completed_node(
                completed_nodes=state.get(
                    "completed_nodes",
                    [],
                ),
                node_name=node_name,
            )
        )

        return {
            "final_output": final_output,
            "assistant_message": (
                assistant_message
            ),
            "current_node": node_name,
            "completed_nodes": (
                completed_nodes
            ),
            "processing_status": (
                final_status
            ),
            "has_error": False,
            "error_node": None,
            "error_message": None,
            "error_details": {},
        }

    except (
        ValidationError,
        ValueError,
        TypeError,
    ) as exc:
        error_message = str(exc)

        failed_state = dict(
            state
        )

        failed_state.update(
            {
                "has_error": True,
                "error_node": node_name,
                "error_message": (
                    error_message
                ),
                "error_details": {
                    "exception_type": (
                        type(exc).__name__
                    ),
                },
            }
        )

        try:
            return _build_failed_output(
                state=failed_state,
                node_name=node_name,
            )

        except ValidationError:
            return {
                "current_node": node_name,
                "completed_nodes": (
                    _add_completed_node(
                        completed_nodes=state.get(
                            "completed_nodes",
                            [],
                        ),
                        node_name=node_name,
                    )
                ),
                "processing_status": "FAILED",
                "has_error": True,
                "error_node": node_name,
                "error_message": (
                    error_message
                ),
                "error_details": {
                    "exception_type": (
                        type(exc).__name__
                    ),
                },
            }
import json
from typing import Any

from pydantic import ValidationError

from app.ai.clients.groq_client import (
    groq_client,
)
from app.ai.graph.state import (
    ComplaintGraphState,
)
from app.ai.prompts.chat_prompt import (
    CHAT_SYSTEM_PROMPT,
    build_chat_user_prompt,
)
from app.ai.schemas.structured_outputs import (
    ChatCorrectionOutput,
)


ALLOWED_COMPLAINT_FIELDS: set[str] = {
    "complainant_name",
    "complainant_email",
    "complainant_phone",
    "customer_type",
    "product_name",
    "product_strength_grade",
    "dosage_form",
    "batch_lot_number",
    "manufacturing_date",
    "expiry_date",
    "complaint_date",
    "incident_date",
    "country",
    "complaint_description",
    "observed_issue",
    "quantity_affected",
    "patient_involved",
    "adverse_event_reported",
    "patient_outcome",
    "storage_conditions",
    "supporting_evidence",
    "source_reference",
    "additional_information",
}


def _parse_json_response(
    content: str,
) -> dict[str, Any]:
    """
    Parse a JSON object returned by Groq.

    Handles plain JSON and Markdown-wrapped JSON.
    """

    if not content or not content.strip():
        raise ValueError(
            "Groq chat-correction response was empty."
        )

    cleaned_content = content.strip()

    if cleaned_content.startswith(
        "```json"
    ):
        cleaned_content = (
            cleaned_content[7:]
        )

    elif cleaned_content.startswith(
        "```"
    ):
        cleaned_content = (
            cleaned_content[3:]
        )

    if cleaned_content.endswith(
        "```"
    ):
        cleaned_content = (
            cleaned_content[:-3]
        )

    cleaned_content = (
        cleaned_content.strip()
    )

    parsed_data = json.loads(
        cleaned_content
    )

    if not isinstance(
        parsed_data,
        dict,
    ):
        raise ValueError(
            "Groq chat-correction response "
            "must be a JSON object."
        )

    return parsed_data


def _normalize_confidence(
    value: Any,
) -> float | None:
    """
    Normalize confidence into a value between 0 and 1.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        numeric_value = float(value)

        if 0 <= numeric_value <= 1:
            return numeric_value

        if 1 < numeric_value <= 100:
            return numeric_value / 100

        return None

    if not isinstance(value, str):
        return None

    normalized_value = (
        value.strip().lower()
    )

    confidence_mapping = {
        "low": 0.4,
        "medium": 0.7,
        "moderate": 0.7,
        "high": 0.9,
        "very high": 0.95,
    }

    if normalized_value in confidence_mapping:
        return confidence_mapping[
            normalized_value
        ]

    if normalized_value.endswith("%"):
        try:
            percentage = float(
                normalized_value[:-1]
            )
        except ValueError:
            return None

        if 0 <= percentage <= 100:
            return percentage / 100

        return None

    try:
        numeric_value = float(
            normalized_value
        )
    except ValueError:
        return None

    if 0 <= numeric_value <= 1:
        return numeric_value

    if 1 < numeric_value <= 100:
        return numeric_value / 100

    return None


def _normalize_chat_output(
    parsed_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize common LLM output variations before validation.
    """

    normalized_data = dict(
        parsed_data
    )

    if "confidence" in normalized_data:
        normalized_data["confidence"] = (
            _normalize_confidence(
                normalized_data.get(
                    "confidence"
                )
            )
        )

    if (
        "update_confidence"
        in normalized_data
    ):
        normalized_data[
            "update_confidence"
        ] = _normalize_confidence(
            normalized_data.get(
                "update_confidence"
            )
        )

    user_intent = normalized_data.get(
        "user_intent"
    )

    if isinstance(user_intent, str):
        normalized_data[
            "user_intent"
        ] = (
            user_intent
            .strip()
            .upper()
            .replace(" ", "_")
            .replace("-", "_")
        )

    return normalized_data


def _get_schema_updated_fields(
    chat_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Read accepted field updates from common schema field names.
    """

    possible_field_names = (
        "updated_fields",
        "field_updates",
        "accepted_updates",
    )

    for field_name in possible_field_names:
        value = chat_result.get(
            field_name
        )

        if isinstance(value, dict):
            return value

    return {}


def _get_schema_rejected_updates(
    chat_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Read rejected updates from common schema field names.
    """

    possible_field_names = (
        "rejected_updates",
        "rejected_fields",
    )

    for field_name in possible_field_names:
        value = chat_result.get(
            field_name
        )

        if isinstance(value, dict):
            return value

    return {}


def _filter_field_updates(
    requested_updates: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """
    Separate valid complaint fields from unknown fields.
    """

    accepted_updates: dict[
        str,
        Any,
    ] = {}

    rejected_updates: dict[
        str,
        Any,
    ] = {}

    for field_name, field_value in (
        requested_updates.items()
    ):
        if (
            field_name
            in ALLOWED_COMPLAINT_FIELDS
        ):
            accepted_updates[
                field_name
            ] = field_value

        else:
            rejected_updates[
                field_name
            ] = {
                "value": field_value,
                "reason": (
                    "Field is not supported by "
                    "the complaint schema."
                ),
            }

    return (
        accepted_updates,
        rejected_updates,
    )


def _merge_fields(
    current_fields: dict[str, Any],
    updated_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge accepted chat corrections into extracted complaint data.
    """

    merged_fields = dict(
        current_fields
    )

    for field_name, field_value in (
        updated_fields.items()
    ):
        merged_fields[
            field_name
        ] = field_value

    return merged_fields


def _add_completed_node(
    completed_nodes: list[str],
    node_name: str,
) -> list[str]:
    """
    Add the node without duplicating its name.
    """

    normalized_nodes = list(
        completed_nodes
    )

    if node_name not in normalized_nodes:
        normalized_nodes.append(
            node_name
        )

    return normalized_nodes


def update_from_chat_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Apply complaint corrections supplied through chat.

    Accepted fields are merged into extracted_fields. Existing
    classification, risk, validation, and final output values
    are cleared so downstream nodes can recalculate them.
    """

    node_name = "update_from_chat"

    if state.get("has_error"):
        return {
            "current_node": node_name,
        }

    chat_message = (
        state.get("chat_message")
        or ""
    ).strip()

    if not chat_message:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": (
                "A chat message is required "
                "to update complaint fields."
            ),
            "error_details": {
                "reason": (
                    "chat_message_missing"
                ),
            },
            "processing_status": "FAILED",
        }

    current_fields = state.get(
        "extracted_fields",
        {},
    )

    existing_fields = state.get(
        "existing_fields",
        {},
    )

    if not current_fields and not existing_fields:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": (
                "Existing complaint fields are required "
                "before applying a chat correction."
            ),
            "error_details": {
                "reason": (
                    "existing_complaint_fields_missing"
                ),
            },
            "processing_status": "FAILED",
        }

    try:
        complaint_fields = {
            **existing_fields,
            **current_fields,
        }

        complaint_json = json.dumps(
            complaint_fields,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        user_prompt = (
            build_chat_user_prompt(
                complaint_json=complaint_json,
                chat_message=chat_message,
            )
        )

        result = (
            groq_client.generate_completion(
                system_prompt=(
                    CHAT_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=1200,
            )
        )

        content = result.get(
            "content"
        )

        if not isinstance(content, str):
            raise ValueError(
                "Groq chat-correction response "
                "did not contain text content."
            )

        parsed_json = (
            _parse_json_response(
                content
            )
        )

        normalized_json = (
            _normalize_chat_output(
                parsed_json
            )
        )

        validated_output = (
            ChatCorrectionOutput
            .model_validate(
                normalized_json
            )
        )

        chat_correction_result = (
            validated_output.model_dump()
        )

        requested_updates = (
            _get_schema_updated_fields(
                chat_correction_result
            )
        )

        schema_rejected_updates = (
            _get_schema_rejected_updates(
                chat_correction_result
            )
        )

        (
            accepted_updates,
            unsupported_updates,
        ) = _filter_field_updates(
            requested_updates
        )

        rejected_updates = {
            **schema_rejected_updates,
            **unsupported_updates,
        }

        merged_fields = _merge_fields(
            current_fields=complaint_fields,
            updated_fields=accepted_updates,
        )

        chat_correction_result[
            "updated_fields"
        ] = accepted_updates

        chat_correction_result[
            "rejected_updates"
        ] = rejected_updates

        completed_nodes = (
            _add_completed_node(
                completed_nodes=state.get(
                    "completed_nodes",
                    [],
                ),
                node_name=node_name,
            )
        )

        prompt_tokens = (
            state.get(
                "prompt_tokens",
                0,
            )
            + (
                result.get(
                    "prompt_tokens"
                )
                or 0
            )
        )

        completion_tokens = (
            state.get(
                "completion_tokens",
                0,
            )
            + (
                result.get(
                    "completion_tokens"
                )
                or 0
            )
        )

        total_tokens = (
            state.get(
                "total_tokens",
                0,
            )
            + (
                result.get(
                    "total_tokens"
                )
                or 0
            )
        )

        assistant_message = (
            chat_correction_result.get(
                "assistant_message"
            )
            or (
                "Complaint information was updated "
                "successfully."
                if accepted_updates
                else (
                    "No complaint fields were updated "
                    "from the chat message."
                )
            )
        )

        return {
            "chat_correction_result": (
                chat_correction_result
            ),
            "updated_fields": (
                accepted_updates
            ),
            "rejected_updates": (
                rejected_updates
            ),
            "extracted_fields": (
                merged_fields
            ),

            # Clear downstream results because complaint data changed.
            "is_complete": False,
            "missing_fields": [],
            "invalid_fields": {},
            "validation_warnings": [],
            "classification_result": {},
            "risk_assessment_result": {},
            "final_output": {},

            "clarification_required": False,
            "clarification_question": None,
            "assistant_message": assistant_message,

            "model_name": result.get(
                "model"
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": (
                completion_tokens
            ),
            "total_tokens": total_tokens,

            "current_node": node_name,
            "completed_nodes": (
                completed_nodes
            ),
            "processing_status": (
                "PROCESSING"
            ),
            "has_error": False,
            "error_node": None,
            "error_message": None,
            "error_details": {},
        }

    except (
        json.JSONDecodeError,
        ValidationError,
        ValueError,
        RuntimeError,
        TypeError,
    ) as exc:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": str(exc),
            "error_details": {
                "exception_type": (
                    type(exc).__name__
                ),
            },
            "processing_status": "FAILED",
        }
import json
from typing import Any

from pydantic import ValidationError

from app.ai.clients.groq_client import groq_client
from app.ai.graph.state import ComplaintGraphState
from app.ai.prompts.classification_prompt import (
    CLASSIFICATION_SYSTEM_PROMPT,
    build_classification_user_prompt,
)
from app.ai.schemas.structured_outputs import (
    ComplaintClassificationOutput,
)
from app.ai.utils.prompt_payload import (
    compact_dictionary,
)
from app.ai.utils.prompt_payload import (
    build_classification_payload,
)

def _parse_json_response(
    content: str,
) -> dict[str, Any]:
    """
    Convert the Groq response into a JSON object.

    Handles plain JSON and JSON wrapped in Markdown code fences.
    """

    if not content or not content.strip():
        raise ValueError(
            "Groq classification response was empty."
        )

    cleaned_content = content.strip()

    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]

    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:]

    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]

    cleaned_content = cleaned_content.strip()

    parsed_data = json.loads(
        cleaned_content
    )

    if not isinstance(parsed_data, dict):
        raise ValueError(
            "Groq classification response must be a JSON object."
        )

    return parsed_data


def _normalize_confidence_value(
    value: Any,
) -> float | None:
    """
    Normalize common LLM confidence formats into a float
    between 0 and 1.

    Supported examples:
    - 0.91
    - "0.91"
    - "91%"
    - "High"
    - "Medium"
    - "Low"
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

    normalized_value = value.strip().lower()

    if not normalized_value:
        return None

    confidence_mapping = {
        "very low": 0.2,
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
        percentage_text = (
            normalized_value
            .removesuffix("%")
            .strip()
        )

        try:
            percentage_value = float(
                percentage_text
            )
        except ValueError:
            return None

        if 0 <= percentage_value <= 100:
            return percentage_value / 100

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


def _normalize_boolean_value(
    value: Any,
    default: bool = False,
) -> bool:
    """
    Normalize common LLM boolean representations.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 1:
            return True

        if value == 0:
            return False

    if isinstance(value, str):
        normalized_value = value.strip().lower()

        truthy_values = {
            "true",
            "yes",
            "y",
            "1",
        }

        falsy_values = {
            "false",
            "no",
            "n",
            "0",
        }

        if normalized_value in truthy_values:
            return True

        if normalized_value in falsy_values:
            return False

    return default


def _normalize_severity_value(
    value: Any,
) -> str | None:
    """
    Normalize common severity variations.

    The structured schema expects:
    MINOR, MAJOR, or CRITICAL.
    """

    if value is None:
        return None

    if not isinstance(value, str):
        return None

    normalized_value = value.strip().upper()

    severity_mapping = {
        "LOW": "MINOR",
        "MINOR": "MINOR",
        "MEDIUM": "MAJOR",
        "MODERATE": "MAJOR",
        "MAJOR": "MAJOR",
        "HIGH": "MAJOR",
        "SEVERE": "CRITICAL",
        "CRITICAL": "CRITICAL",
    }

    return severity_mapping.get(
        normalized_value
    )


def _normalize_classification_output(
    classification_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize common LLM output variations before Pydantic
    validation.
    """

    normalized_data = dict(
        classification_data
    )

    normalized_data[
        "classification_confidence"
    ] = _normalize_confidence_value(
        normalized_data.get(
            "classification_confidence"
        )
    )

    normalized_data[
        "suggested_severity"
    ] = _normalize_severity_value(
        normalized_data.get(
            "suggested_severity"
        )
    )

    normalized_data[
        "is_quality_complaint"
    ] = _normalize_boolean_value(
        normalized_data.get(
            "is_quality_complaint"
        ),
        default=False,
    )

    normalized_data[
        "is_adverse_event"
    ] = _normalize_boolean_value(
        normalized_data.get(
            "is_adverse_event"
        ),
        default=False,
    )

    normalized_data[
        "requires_immediate_attention"
    ] = _normalize_boolean_value(
        normalized_data.get(
            "requires_immediate_attention"
        ),
        default=False,
    )

    return normalized_data


def _apply_classification_consistency_rules(
    classification_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply deterministic business rules after classification.

    These checks prevent contradictory AI output from entering
    the graph state.
    """

    normalized_result = dict(
        classification_result
    )

    category = (
        normalized_result.get(
            "complaint_category"
        )
        or ""
    ).strip().lower()

    subcategory = (
        normalized_result.get(
            "complaint_subcategory"
        )
        or ""
    ).strip().lower()

    complaint_type = (
        normalized_result.get(
            "complaint_type"
        )
        or ""
    ).strip().lower()

    quality_categories = {
        "product quality",
        "quality complaint",
        "product defect",
        "quality defect",
    }

    if (
        category in quality_categories
        or "quality" in category
        or "product defect" in category
    ):
        normalized_result[
            "is_quality_complaint"
        ] = True

    quality_terms = {
        "discoloration",
        "contamination",
        "breakage",
        "leakage",
        "packaging defect",
        "packaging failure",
        "stability",
        "defect",
    }

    combined_classification_text = " ".join(
        [
            category,
            subcategory,
            complaint_type,
        ]
    )

    if any(
        quality_term
        in combined_classification_text
        for quality_term in quality_terms
    ):
        normalized_result[
            "is_quality_complaint"
        ] = True

    if normalized_result.get(
        "is_adverse_event"
    ):
        normalized_result[
            "requires_immediate_attention"
        ] = True

    if (
        normalized_result.get(
            "suggested_severity"
        )
        == "CRITICAL"
    ):
        normalized_result[
            "requires_immediate_attention"
        ] = True

    return normalized_result


def classify_complaint_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Classify the complaint using extracted complaint fields.

    This node determines complaint category, subcategory,
    complaint type, severity, and related classification
    information.
    """

    node_name = "classify_complaint"

    if state.get("has_error"):
        return {
            "current_node": node_name,
        }

    extracted_fields = state.get(
        "extracted_fields",
        {},
    )

    if not extracted_fields:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": (
                "Extracted complaint fields are required "
                "before classification."
            ),
            "error_details": {
                "reason": (
                    "extracted_fields_missing"
                ),
            },
            "processing_status": "FAILED",
        }

    try:
        classification_payload = (
            build_classification_payload(
                extracted_fields
            )
        )

        complaint_json = json.dumps(
            classification_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        user_prompt = (
            build_classification_user_prompt(
                complaint_json=complaint_json,
            )
        )

        result = groq_client.generate_completion(
            system_prompt=(
                CLASSIFICATION_SYSTEM_PROMPT
            ),
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=900,
        )

        response_content = result.get(
            "content"
        )

        if not isinstance(
            response_content,
            str,
        ):
            raise ValueError(
                "Groq classification response "
                "did not contain valid text content."
            )

        parsed_json = _parse_json_response(
            response_content
        )

        normalized_json = (
            _normalize_classification_output(
                parsed_json
            )
        )

        validated_output = (
            ComplaintClassificationOutput
            .model_validate(
                normalized_json
            )
        )

        classification_result = (
            validated_output.model_dump()
        )

        classification_result = (
            _apply_classification_consistency_rules(
                classification_result
            )
        )

        previous_completed_nodes = list(
            state.get(
                "completed_nodes",
                [],
            )
        )

        if node_name not in previous_completed_nodes:
            completed_nodes = [
                *previous_completed_nodes,
                node_name,
            ]
        else:
            completed_nodes = (
                previous_completed_nodes
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

        return {
            "classification_result": (
                classification_result
            ),
            "model_name": result.get(
                "model"
            ),
            "prompt_tokens": (
                prompt_tokens
            ),
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
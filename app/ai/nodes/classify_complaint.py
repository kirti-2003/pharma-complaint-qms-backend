import json
from typing import Any

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
    build_classification_payload,
)


def _apply_classification_consistency_rules(
    classification_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply deterministic business rules after AI classification.

    These rules prevent contradictory AI output from
    entering the LangGraph state.
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

    This node determines:
    - complaint category
    - complaint subcategory
    - complaint type
    - suggested severity
    - quality complaint status
    - adverse event status
    - immediate attention requirement
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

        result = (
            groq_client.generate_structured_output(
                system_prompt=(
                    CLASSIFICATION_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    ComplaintClassificationOutput
                ),
                temperature=0.1,
                max_tokens=2500,
            )
        )

        validated_output = result[
            "parsed"
        ]

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
            "total_tokens": (
                total_tokens
            ),
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
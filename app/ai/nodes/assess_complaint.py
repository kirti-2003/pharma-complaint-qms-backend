import json
from typing import Any

from app.ai.clients.groq_client import groq_client
from app.ai.graph.state import ComplaintGraphState
from app.ai.prompts.assessment_prompt import (
    ASSESSMENT_SYSTEM_PROMPT,
    build_assessment_user_prompt,
)
from app.ai.schemas.structured_outputs import (
    ComplaintAssessmentOutput,
)
from app.ai.utils.prompt_payload import (
    build_classification_payload,
)


def assess_complaint_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Perform combined complaint classification and risk assessment.

    This node uses one structured Groq call to produce:
    - complaint classification
    - preliminary risk assessment
    """

    node_name = "assess_complaint"

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
                "Extracted complaint fields are required."
            ),
            "error_details": {
                "reason": "extracted_fields_missing",
            },
            "processing_status": "FAILED",
        }

    try:
        assessment_payload = (
            build_classification_payload(
                extracted_fields
            )
        )

        complaint_json = json.dumps(
            assessment_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        user_prompt = (
            build_assessment_user_prompt(
                complaint_json
            )
        )

        result = (
            groq_client.generate_structured_output(
                system_prompt=(
                    ASSESSMENT_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    ComplaintAssessmentOutput
                ),
                temperature=0.0,
                max_tokens=4000,
            )
        )

        validated = result["parsed"]

        classification_result = (
            validated.classification.model_dump()
        )

        risk_assessment_result = (
            validated.risk_assessment.model_dump()
        )

        completed_nodes = list(
            state.get(
                "completed_nodes",
                [],
            )
        )

        if node_name not in completed_nodes:
            completed_nodes.append(
                node_name
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
            "risk_assessment_result": (
                risk_assessment_result
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
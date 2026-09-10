import json
from typing import Any

from app.ai.clients.groq_client import groq_client
from app.ai.graph.state import ComplaintGraphState
from app.ai.prompts.risk_prompt import (
    RISK_SYSTEM_PROMPT,
    build_risk_user_prompt,
)
from app.ai.schemas.structured_outputs import (
    RiskAssessmentOutput,
)
from app.ai.utils.prompt_payload import (
    build_risk_payload,
)


def assess_risk_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Generate an initial pharmaceutical complaint risk assessment.

    The node uses:
    - extracted complaint fields
    - complaint classification result

    The assessment is preliminary and does not represent
    a final quality or regulatory decision.
    """

    node_name = "assess_risk"

    if state.get("has_error"):
        return {
            "current_node": node_name,
        }

    extracted_fields = state.get(
        "extracted_fields",
        {},
    )

    classification_result = state.get(
        "classification_result",
        {},
    )

    if not extracted_fields:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": (
                "Extracted complaint fields are required "
                "before risk assessment."
            ),
            "error_details": {
                "reason": (
                    "extracted_fields_missing"
                ),
            },
            "processing_status": "FAILED",
        }

    if not classification_result:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": (
                "Complaint classification is required "
                "before risk assessment."
            ),
            "error_details": {
                "reason": (
                    "classification_result_missing"
                ),
            },
            "processing_status": "FAILED",
        }

    try:
        risk_payload = build_risk_payload(
            extracted_fields=(
                extracted_fields
            ),
            classification_result=(
                classification_result
            ),
        )

        complaint_json = json.dumps(
            risk_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        user_prompt = build_risk_user_prompt(
            complaint_json=complaint_json,
        )

        result = (
            groq_client.generate_structured_output(
                system_prompt=(
                    RISK_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    RiskAssessmentOutput
                ),
                temperature=0.1,
                max_tokens=3000,
            )
        )

        validated_output = result[
            "parsed"
        ]

        risk_assessment_result = (
            validated_output.model_dump()
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
import json
from typing import Any

from pydantic import ValidationError

from app.ai.clients.groq_client import groq_client
from app.ai.graph.state import ComplaintGraphState
from app.ai.prompts.risk_prompt import (
    RISK_SYSTEM_PROMPT,
    build_risk_user_prompt,
)
from app.ai.schemas.structured_outputs import (
    RiskAssessmentOutput,
)


def _parse_json_response(
    content: str,
) -> dict[str, Any]:
    """
    Convert the Groq risk-assessment response into a JSON object.
    """

    if not content or not content.strip():
        raise ValueError(
            "Groq risk response was empty."
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
            "Groq risk response must be a JSON object."
        )

    return parsed_data


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
                "reason": "extracted_fields_missing",
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
                "reason": "classification_result_missing",
            },
            "processing_status": "FAILED",
        }

    try:
        risk_payload = {
            "extracted_fields": extracted_fields,
            "classification_result": classification_result,
        }

        complaint_json = json.dumps(
            risk_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        user_prompt = build_risk_user_prompt(
            complaint_json=complaint_json,
        )

        result = groq_client.generate_completion(
            system_prompt=RISK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=1500,
        )

        parsed_json = _parse_json_response(
            result["content"]
        )

        validated_output = (
            RiskAssessmentOutput.model_validate(
                parsed_json
            )
        )

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
            state.get("prompt_tokens", 0)
            + (result.get("prompt_tokens") or 0)
        )

        completion_tokens = (
            state.get("completion_tokens", 0)
            + (result.get("completion_tokens") or 0)
        )

        total_tokens = (
            state.get("total_tokens", 0)
            + (result.get("total_tokens") or 0)
        )

        return {
            "risk_assessment_result": (
                risk_assessment_result
            ),
            "model_name": result.get("model"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "current_node": node_name,
            "completed_nodes": completed_nodes,
            "processing_status": "PROCESSING",
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
    ) as exc:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": str(exc),
            "error_details": {
                "exception_type": type(exc).__name__,
            },
            "processing_status": "FAILED",
        }
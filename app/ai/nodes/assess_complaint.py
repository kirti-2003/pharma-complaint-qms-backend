import json
from typing import Any

from pydantic import ValidationError

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


def _parse_json_response(
    content: str,
) -> dict[str, Any]:
    cleaned = content.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    parsed = json.loads(cleaned.strip())

    if not isinstance(parsed, dict):
        raise ValueError(
            "Assessment response must be a JSON object."
        )

    return parsed


def assess_complaint_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
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

        result = groq_client.generate_completion(
            system_prompt=ASSESSMENT_SYSTEM_PROMPT,
            user_prompt=build_assessment_user_prompt(
                complaint_json
            ),
            temperature=0.0,
            max_tokens=950,
        )

        parsed = _parse_json_response(
            result["content"]
        )

        validated = (
            ComplaintAssessmentOutput
            .model_validate(parsed)
        )

        completed_nodes = list(
            state.get("completed_nodes", [])
        )

        if node_name not in completed_nodes:
            completed_nodes.append(node_name)

        return {
            "classification_result": (
                validated.classification.model_dump()
            ),
            "risk_assessment_result": (
                validated.risk_assessment.model_dump()
            ),
            "model_name": result.get("model"),
            "prompt_tokens": (
                state.get("prompt_tokens", 0)
                + (result.get("prompt_tokens") or 0)
            ),
            "completion_tokens": (
                state.get("completion_tokens", 0)
                + (result.get("completion_tokens") or 0)
            ),
            "total_tokens": (
                state.get("total_tokens", 0)
                + (result.get("total_tokens") or 0)
            ),
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
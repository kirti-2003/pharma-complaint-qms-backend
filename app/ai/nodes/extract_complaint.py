import json
from typing import Any

from pydantic import ValidationError

from app.ai.clients.groq_client import groq_client
from app.ai.graph.state import ComplaintGraphState
from app.ai.prompts.extraction_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
)
from app.ai.schemas.structured_outputs import (
    ExtractedComplaintOutput,
)


def _parse_json_response(content: str) -> dict[str, Any]:
    """
    Convert the Groq text response into a Python dictionary.

    Markdown JSON fences are removed because an LLM may occasionally
    include them even when the prompt requests plain JSON.
    """

    cleaned_content = content.strip()

    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]

    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:]

    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]

    cleaned_content = cleaned_content.strip()

    parsed_data = json.loads(cleaned_content)

    if not isinstance(parsed_data, dict):
        raise ValueError(
            "Groq extraction response must be a JSON object."
        )

    return parsed_data


def extract_complaint_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Extract structured complaint fields from raw complaint text.

    This node:
    - reads raw_text from the graph state
    - builds the extraction prompt
    - calls Groq
    - validates the returned JSON using Pydantic
    - updates extracted fields and token usage

    Database operations are intentionally excluded from this node.
    """

    node_name = "extract_complaint"

    raw_text = (state.get("raw_text") or "").strip()

    if not raw_text:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": (
                "Complaint source text cannot be empty."
            ),
            "error_details": {
                "reason": "raw_text_missing",
            },
            "processing_status": "FAILED",
        }

    try:
        user_prompt = build_extraction_user_prompt(
            raw_text=raw_text,
        )

        result = groq_client.generate_completion(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=2000,
        )

        parsed_json = _parse_json_response(
            result["content"]
        )

        validated_output = (
            ExtractedComplaintOutput.model_validate(
                parsed_json
            )
        )

        extracted_fields = validated_output.model_dump()

        previous_completed_nodes = state.get(
            "completed_nodes",
            [],
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
            "extracted_fields": extracted_fields,
            "model_name": result.get("model"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "current_node": node_name,
            "completed_nodes": [
                *previous_completed_nodes,
                node_name,
            ],
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
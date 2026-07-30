import pytest

from app.ai.prompts.extraction_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
)


def test_extraction_system_prompt_contains_core_rules():
    assert "extract structured complaint information" in (
        EXTRACTION_SYSTEM_PROMPT.lower()
    )

    assert "do not guess" in EXTRACTION_SYSTEM_PROMPT.lower()
    assert "return valid json only" in EXTRACTION_SYSTEM_PROMPT.lower()
    assert "do not classify" in EXTRACTION_SYSTEM_PROMPT.lower()


def test_build_extraction_user_prompt():
    raw_text = (
        "A customer reported that Paracetamol 500 mg tablets from "
        "batch BT-2026-001 were broken inside the blister pack."
    )

    prompt = build_extraction_user_prompt(raw_text)

    assert raw_text in prompt
    assert '"product_name": null' in prompt
    assert '"batch_lot_number": null' in prompt
    assert '"complaint_description": null' in prompt
    assert '"supporting_evidence": []' in prompt


def test_build_extraction_user_prompt_strips_whitespace():
    prompt = build_extraction_user_prompt(
        "   Product carton was damaged during delivery.   "
    )

    assert "Product carton was damaged during delivery." in prompt
    assert "   Product carton was damaged during delivery.   " not in prompt


def test_build_extraction_user_prompt_rejects_empty_text():
    with pytest.raises(
        ValueError,
        match="Complaint source text cannot be empty",
    ):
        build_extraction_user_prompt("   ")
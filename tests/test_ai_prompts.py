import pytest

from app.ai.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
    CLASSIFICATION_SYSTEM_PROMPT,
    build_classification_user_prompt,
    RISK_SYSTEM_PROMPT,
    build_risk_user_prompt,
    CHAT_SYSTEM_PROMPT,
    build_chat_user_prompt,
)


# ==========================================================
# Extraction Prompt
# ==========================================================

def test_extraction_system_prompt_contains_core_rules():
    assert "extract structured complaint information" in (
        EXTRACTION_SYSTEM_PROMPT.lower()
    )

    assert "do not guess" in EXTRACTION_SYSTEM_PROMPT.lower()
    assert "return valid json only" in EXTRACTION_SYSTEM_PROMPT.lower()


def test_build_extraction_user_prompt():
    raw_text = (
        "Customer reported broken tablets in blister pack."
    )

    prompt = build_extraction_user_prompt(raw_text)

    assert raw_text in prompt
    assert '"product_name": null' in prompt
    assert '"complaint_description": null' in prompt


def test_build_extraction_prompt_rejects_empty():
    with pytest.raises(
        ValueError,
        match="Complaint source text cannot be empty",
    ):
        build_extraction_user_prompt("    ")


# ==========================================================
# Classification Prompt
# ==========================================================

def test_classification_system_prompt_contains_rules():
    prompt = CLASSIFICATION_SYSTEM_PROMPT.lower()

    assert "classification" in prompt
    assert "minor" in prompt
    assert "major" in prompt
    assert "critical" in prompt
    assert "return valid json only" in prompt


def test_build_classification_prompt():
    complaint = """
    {
        "product_name":"Paracetamol",
        "batch_lot_number":"BT-123"
    }
    """

    prompt = build_classification_user_prompt(
        complaint
    )

    assert "Paracetamol" in prompt
    assert '"complaint_category"' in prompt
    assert '"suggested_severity"' in prompt


def test_classification_prompt_rejects_empty():
    with pytest.raises(
        ValueError,
        match="Complaint JSON cannot be empty",
    ):
        build_classification_user_prompt(" ")


# ==========================================================
# Risk Prompt
# ==========================================================

def test_risk_system_prompt_contains_rules():
    prompt = RISK_SYSTEM_PROMPT.lower()

    assert "risk assessment" in prompt
    assert "return valid json only" in prompt
    assert "do not classify" in prompt


def test_build_risk_prompt():
    complaint = """
    {
        "severity":"CRITICAL"
    }
    """

    prompt = build_risk_user_prompt(
        complaint
    )

    assert "CRITICAL" in prompt
    assert '"risk_level"' in prompt
    assert '"risk_score"' in prompt


def test_risk_prompt_rejects_empty():
    with pytest.raises(
        ValueError,
        match="Complaint JSON cannot be empty",
    ):
        build_risk_user_prompt(" ")


# ==========================================================
# Chat Prompt
# ==========================================================

def test_chat_system_prompt_contains_rules():
    prompt = CHAT_SYSTEM_PROMPT.lower()

    assert "ai complaint assistant" in prompt
    assert "updated fields" in prompt
    assert "return valid json only" in prompt


def test_build_chat_prompt():
    complaint = """
    {
        "batch_lot_number":"BT-123"
    }
    """

    message = (
        "Batch number should be BT-456."
    )

    prompt = build_chat_user_prompt(
        complaint,
        message,
    )

    assert "BT-123" in prompt
    assert "BT-456" in prompt
    assert '"updated_fields"' in prompt


def test_chat_prompt_rejects_empty_complaint():
    with pytest.raises(
        ValueError,
        match="Complaint JSON cannot be empty",
    ):
        build_chat_user_prompt(
            "",
            "Hello",
        )


def test_chat_prompt_rejects_empty_message():
    with pytest.raises(
        ValueError,
        match="User message cannot be empty",
    ):
        build_chat_user_prompt(
            "{}",
            "",
        )
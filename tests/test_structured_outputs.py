from app.ai.schemas.structured_outputs import (
    ComplaintClassificationOutput,
    ExtractedComplaintOutput,
    RiskAssessmentOutput,
)


def test_extracted_complaint_output():
    result = ExtractedComplaintOutput(
        product_name="Paracetamol 500 mg",
        batch_lot_number="BT-2026-001",
        complaint_description="Tablets were found broken inside the strip.",
        adverse_event_reported=False,
    )

    assert result.product_name == "Paracetamol 500 mg"
    assert result.batch_lot_number == "BT-2026-001"
    assert result.adverse_event_reported is False


def test_classification_output():
    result = ComplaintClassificationOutput(
        complaint_category="Product Quality",
        complaint_subcategory="Broken Tablet",
        suggested_severity="MEDIUM",
        is_quality_complaint=True,
        classification_confidence=0.92,
    )

    assert result.is_quality_complaint is True
    assert result.classification_confidence == 0.92


def test_risk_assessment_output():
    result = RiskAssessmentOutput(
        risk_level="MEDIUM",
        requires_batch_investigation=True,
        suggested_next_action="Review batch manufacturing and packaging records.",
        risk_confidence=0.87,
    )

    assert result.risk_level == "MEDIUM"
    assert result.requires_batch_investigation is True
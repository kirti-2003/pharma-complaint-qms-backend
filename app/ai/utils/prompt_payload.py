from typing import Any


def is_meaningful_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)

    return True


def compact_dictionary(
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if is_meaningful_value(value)
    }

CLASSIFICATION_FIELD_NAMES = {
    "product_name",
    "product_strength_grade",
    "dosage_form",
    "batch_lot_number",
    "complaint_description",
    "observed_issue",
    "quantity_affected",
    "patient_involved",
    "adverse_event_reported",
    "patient_outcome",
    "storage_conditions",
}


RISK_FIELD_NAMES = {
    "product_name",
    "product_strength_grade",
    "dosage_form",
    "batch_lot_number",
    "complaint_description",
    "observed_issue",
    "quantity_affected",
    "patient_involved",
    "adverse_event_reported",
    "patient_outcome",
    "storage_conditions",
    "supporting_evidence",
}


def is_meaningful_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)

    return True


def select_meaningful_fields(
    data: dict[str, Any],
    allowed_fields: set[str],
) -> dict[str, Any]:
    return {
        field_name: field_value
        for field_name, field_value in data.items()
        if (
            field_name in allowed_fields
            and is_meaningful_value(field_value)
        )
    }


def build_classification_payload(
    extracted_fields: dict[str, Any],
) -> dict[str, Any]:
    return select_meaningful_fields(
        extracted_fields,
        CLASSIFICATION_FIELD_NAMES,
    )


def build_risk_payload(
    extracted_fields: dict[str, Any],
    classification_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "complaint": select_meaningful_fields(
            extracted_fields,
            RISK_FIELD_NAMES,
        ),
        "classification": {
            key: value
            for key, value in classification_result.items()
            if key
            in {
                "complaint_category",
                "complaint_subcategory",
                "suggested_severity",
                "is_quality_complaint",
                "is_adverse_event",
                "requires_immediate_attention",
            }
            and is_meaningful_value(value)
        },
    }
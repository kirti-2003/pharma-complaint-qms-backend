import re
from typing import Any

from pydantic import ValidationError

from app.ai.graph.state import ComplaintGraphState
from app.ai.schemas.structured_outputs import (
    FieldValidationOutput,
)


# These fields are required for every complaint.
BASE_REQUIRED_FIELDS: tuple[str, ...] = (
    "complaint_description",
    "product_name",
)


# Fields normally required when the complaint concerns
# a physical pharmaceutical product or batch.
PRODUCT_COMPLAINT_REQUIRED_FIELDS: tuple[str, ...] = (
    "batch_lot_number",
)


FIELD_DISPLAY_NAMES: dict[str, str] = {
    "complainant_name": "complainant name",
    "complainant_email": "complainant email",
    "complainant_phone": "complainant phone number",
    "customer_type": "customer type",
    "product_name": "product name",
    "product_strength_grade": "product strength or grade",
    "dosage_form": "dosage form",
    "batch_lot_number": "batch or lot number",
    "manufacturing_date": "manufacturing date",
    "expiry_date": "expiry date",
    "complaint_date": "complaint date",
    "incident_date": "incident date",
    "country": "country",
    "complaint_description": "complaint description",
    "observed_issue": "observed issue",
    "quantity_affected": "quantity affected",
    "patient_involved": "patient involvement information",
    "adverse_event_reported": "adverse-event information",
    "patient_outcome": "patient outcome",
    "storage_conditions": "storage conditions",
    "source_reference": "source reference",
}


EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9-]+"
    r"(?:\.[A-Za-z0-9-]+)+$"
)


PHONE_PATTERN = re.compile(
    r"^[0-9+\-().\s]{7,25}$"
)


PRODUCT_ISSUE_TERMS: tuple[str, ...] = (
    "discoloration",
    "discolored",
    "contamination",
    "contaminated",
    "broken",
    "breakage",
    "damaged",
    "leakage",
    "leaking",
    "foreign particle",
    "foreign material",
    "odor",
    "smell",
    "stability",
    "degradation",
    "packaging defect",
    "seal failure",
    "missing tablet",
    "missing capsule",
    "wrong strength",
    "wrong product",
)


def _is_missing_value(
    value: Any,
) -> bool:
    """
    Return True when a complaint field should be treated
    as missing.
    """

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, list | dict | tuple | set):
        return len(value) == 0

    return False


def _combine_complaint_fields(
    existing_fields: dict[str, Any],
    extracted_fields: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine existing complaint fields with newly extracted fields.

    Non-empty extracted values take priority over existing values.
    Empty AI values do not overwrite already available values.
    """

    combined_fields = dict(
        existing_fields
    )

    for field_name, extracted_value in extracted_fields.items():
        if not _is_missing_value(
            extracted_value
        ):
            combined_fields[
                field_name
            ] = extracted_value

        elif field_name not in combined_fields:
            combined_fields[
                field_name
            ] = extracted_value

    return combined_fields


def _is_product_related_complaint(
    complaint_fields: dict[str, Any],
) -> bool:
    """
    Determine whether the extracted complaint appears to concern
    a physical pharmaceutical product.

    This is used only for deterministic field validation.
    It is not a replacement for the classification node.
    """

    product_name = complaint_fields.get(
        "product_name"
    )

    dosage_form = complaint_fields.get(
        "dosage_form"
    )

    observed_issue = (
        complaint_fields.get(
            "observed_issue"
        )
        or ""
    )

    complaint_description = (
        complaint_fields.get(
            "complaint_description"
        )
        or ""
    )

    searchable_text = (
        f"{observed_issue} "
        f"{complaint_description}"
    ).strip().lower()

    if not _is_missing_value(product_name):
        return True

    if not _is_missing_value(dosage_form):
        return True

    return any(
        issue_term in searchable_text
        for issue_term in PRODUCT_ISSUE_TERMS
    )


def _get_required_fields(
    complaint_fields: dict[str, Any],
) -> list[str]:
    """
    Determine required fields based on the complaint data.
    """

    required_fields = list(
        BASE_REQUIRED_FIELDS
    )

    if _is_product_related_complaint(
        complaint_fields
    ):
        required_fields.extend(
            PRODUCT_COMPLAINT_REQUIRED_FIELDS
        )

    return list(
        dict.fromkeys(
            required_fields
        )
    )


def _find_missing_fields(
    complaint_fields: dict[str, Any],
    required_fields: list[str],
) -> list[str]:
    """
    Return required fields that have no usable value.
    """

    return [
        field_name
        for field_name in required_fields
        if _is_missing_value(
            complaint_fields.get(
                field_name
            )
        )
    ]


def _validate_email(
    email: Any,
) -> str | None:
    """
    Validate the complainant email when one is provided.
    """

    if _is_missing_value(email):
        return None

    if not isinstance(email, str):
        return (
            "Complainant email must be a string."
        )

    if not EMAIL_PATTERN.fullmatch(
        email.strip()
    ):
        return (
            "Complainant email is not in a valid format."
        )

    return None


def _validate_phone(
    phone: Any,
) -> str | None:
    """
    Validate the complainant phone number when one is provided.
    """

    if _is_missing_value(phone):
        return None

    if not isinstance(phone, str):
        return (
            "Complainant phone number must be a string."
        )

    if not PHONE_PATTERN.fullmatch(
        phone.strip()
    ):
        return (
            "Complainant phone number contains invalid characters "
            "or has an invalid length."
        )

    digit_count = sum(
        character.isdigit()
        for character in phone
    )

    if digit_count < 7:
        return (
            "Complainant phone number must contain at least "
            "7 digits."
        )

    return None


def _validate_quantity_affected(
    quantity_affected: Any,
) -> str | None:
    """
    Validate the affected quantity when it can be interpreted
    as a simple numeric value.

    Free-text quantities such as 'approximately 10 bottles'
    remain valid.
    """

    if _is_missing_value(
        quantity_affected
    ):
        return None

    if isinstance(
        quantity_affected,
        bool,
    ):
        return (
            "Quantity affected cannot be a boolean value."
        )

    if isinstance(
        quantity_affected,
        int | float,
    ):
        if quantity_affected < 0:
            return (
                "Quantity affected cannot be negative."
            )

        return None

    if not isinstance(
        quantity_affected,
        str,
    ):
        return (
            "Quantity affected must be text or a numeric value."
        )

    normalized_quantity = (
        quantity_affected.strip()
    )

    if not normalized_quantity:
        return None

    try:
        numeric_quantity = float(
            normalized_quantity
        )
    except ValueError:
        # Text values such as "12 capsules" are allowed.
        return None

    if numeric_quantity < 0:
        return (
            "Quantity affected cannot be negative."
        )

    return None


def _find_invalid_fields(
    complaint_fields: dict[str, Any],
) -> dict[str, str]:
    """
    Find invalid or contradictory complaint values.
    """

    invalid_fields: dict[str, str] = {}

    email_error = _validate_email(
        complaint_fields.get(
            "complainant_email"
        )
    )

    if email_error:
        invalid_fields[
            "complainant_email"
        ] = email_error

    phone_error = _validate_phone(
        complaint_fields.get(
            "complainant_phone"
        )
    )

    if phone_error:
        invalid_fields[
            "complainant_phone"
        ] = phone_error

    quantity_error = (
        _validate_quantity_affected(
            complaint_fields.get(
                "quantity_affected"
            )
        )
    )

    if quantity_error:
        invalid_fields[
            "quantity_affected"
        ] = quantity_error

    patient_involved = (
        complaint_fields.get(
            "patient_involved"
        )
    )

    adverse_event_reported = (
        complaint_fields.get(
            "adverse_event_reported"
        )
    )

    if (
        adverse_event_reported is True
        and patient_involved is False
    ):
        invalid_fields[
            "patient_involved"
        ] = (
            "Patient involvement cannot be false when an "
            "adverse event is reported."
        )

    manufacturing_date = (
        complaint_fields.get(
            "manufacturing_date"
        )
    )

    expiry_date = (
        complaint_fields.get(
            "expiry_date"
        )
    )

    if (
        isinstance(manufacturing_date, str)
        and isinstance(expiry_date, str)
        and manufacturing_date.strip()
        and expiry_date.strip()
        and manufacturing_date.strip().lower()
        == expiry_date.strip().lower()
    ):
        invalid_fields[
            "expiry_date"
        ] = (
            "Expiry date should not be identical to the "
            "manufacturing date."
        )

    return invalid_fields


def _build_validation_warnings(
    complaint_fields: dict[str, Any],
) -> list[str]:
    """
    Build non-blocking data-quality warnings.
    """

    warnings: list[str] = []

    complainant_email = (
        complaint_fields.get(
            "complainant_email"
        )
    )

    complainant_phone = (
        complaint_fields.get(
            "complainant_phone"
        )
    )

    if (
        _is_missing_value(
            complainant_email
        )
        and _is_missing_value(
            complainant_phone
        )
    ):
        warnings.append(
            "No complainant email or phone number was provided."
        )

    if _is_missing_value(
        complaint_fields.get(
            "complainant_name"
        )
    ):
        warnings.append(
            "Complainant name was not provided."
        )

    if _is_missing_value(
        complaint_fields.get(
            "quantity_affected"
        )
    ):
        warnings.append(
            "Affected quantity was not provided."
        )

    if _is_missing_value(
        complaint_fields.get(
            "incident_date"
        )
    ):
        warnings.append(
            "Incident date was not provided."
        )

    if _is_missing_value(
        complaint_fields.get(
            "country"
        )
    ):
        warnings.append(
            "Complaint country was not provided."
        )

    if _is_missing_value(
        complaint_fields.get(
            "storage_conditions"
        )
    ):
        warnings.append(
            "Storage or transportation conditions were not provided."
        )

    supporting_evidence = (
        complaint_fields.get(
            "supporting_evidence"
        )
    )

    if _is_missing_value(
        supporting_evidence
    ):
        warnings.append(
            "No supporting evidence was identified."
        )

    adverse_event_reported = (
        complaint_fields.get(
            "adverse_event_reported"
        )
    )

    patient_outcome = (
        complaint_fields.get(
            "patient_outcome"
        )
    )

    if (
        adverse_event_reported is True
        and _is_missing_value(
            patient_outcome
        )
    ):
        warnings.append(
            "An adverse event was reported, but the patient "
            "outcome was not provided."
        )

    return warnings


def _build_clarification_question(
    missing_fields: list[str],
    invalid_fields: dict[str, str],
) -> str | None:
    """
    Build one user-facing clarification request.
    """

    clarification_parts: list[str] = []

    if missing_fields:
        display_names = [
            FIELD_DISPLAY_NAMES.get(
                field_name,
                field_name.replace(
                    "_",
                    " ",
                ),
            )
            for field_name in missing_fields
        ]

        if len(display_names) == 1:
            missing_text = display_names[0]

        else:
            missing_text = (
                ", ".join(
                    display_names[:-1]
                )
                + f", and {display_names[-1]}"
            )

        clarification_parts.append(
            f"Please provide the missing {missing_text}."
        )

    if invalid_fields:
        invalid_display_names = [
            FIELD_DISPLAY_NAMES.get(
                field_name,
                field_name.replace(
                    "_",
                    " ",
                ),
            )
            for field_name in invalid_fields
        ]

        if len(invalid_display_names) == 1:
            invalid_text = (
                invalid_display_names[0]
            )

        else:
            invalid_text = (
                ", ".join(
                    invalid_display_names[:-1]
                )
                + (
                    f", and "
                    f"{invalid_display_names[-1]}"
                )
            )

        clarification_parts.append(
            f"Please correct the invalid {invalid_text}."
        )

    if not clarification_parts:
        return None

    return " ".join(
        clarification_parts
    )


def validate_fields_node(
    state: ComplaintGraphState,
) -> dict[str, Any]:
    """
    Validate extracted complaint fields.

    This node uses deterministic validation only. It does not
    call an LLM and therefore does not consume tokens.
    """

    node_name = "validate_fields"

    if state.get("has_error"):
        return {
            "current_node": node_name,
        }

    extracted_fields = state.get(
        "extracted_fields",
        {},
    )

    existing_fields = state.get(
        "existing_fields",
        {},
    )

    if not extracted_fields and not existing_fields:
        return {
            "current_node": node_name,
            "has_error": True,
            "error_node": node_name,
            "error_message": (
                "Complaint fields are required before validation."
            ),
            "error_details": {
                "reason": (
                    "complaint_fields_missing"
                ),
            },
            "processing_status": "FAILED",
        }

    try:
        complaint_fields = (
            _combine_complaint_fields(
                existing_fields=existing_fields,
                extracted_fields=extracted_fields,
            )
        )

        required_fields = (
            _get_required_fields(
                complaint_fields
            )
        )

        missing_fields = (
            _find_missing_fields(
                complaint_fields=complaint_fields,
                required_fields=required_fields,
            )
        )

        invalid_fields = (
            _find_invalid_fields(
                complaint_fields
            )
        )

        warnings = (
            _build_validation_warnings(
                complaint_fields
            )
        )

        is_complete = (
            not missing_fields
            and not invalid_fields
        )

        validation_output = (
            FieldValidationOutput(
                is_complete=is_complete,
                missing_fields=missing_fields,
                invalid_fields=invalid_fields,
                warnings=warnings,
            )
        )

        clarification_required = (
            not is_complete
        )

        clarification_question = (
            _build_clarification_question(
                missing_fields=missing_fields,
                invalid_fields=invalid_fields,
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

        processing_status = (
            "PROCESSING"
            if is_complete
            else "WAITING_FOR_USER"
        )

        return {
            "is_complete": (
                validation_output.is_complete
            ),
            "missing_fields": (
                validation_output.missing_fields
            ),
            "invalid_fields": (
                validation_output.invalid_fields
            ),
            "validation_warnings": (
                validation_output.warnings
            ),
            "clarification_required": (
                clarification_required
            ),
            "clarification_question": (
                clarification_question
            ),
            "assistant_message": (
                clarification_question
                if clarification_required
                else None
            ),
            "current_node": node_name,
            "completed_nodes": (
                completed_nodes
            ),
            "processing_status": (
                processing_status
            ),
            "has_error": False,
            "error_node": None,
            "error_message": None,
            "error_details": {},
        }

    except (
        ValidationError,
        ValueError,
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
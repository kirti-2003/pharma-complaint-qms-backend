from textwrap import dedent


CHAT_SYSTEM_PROMPT = dedent(
    """
    You are a pharmaceutical complaint-management assistant.

    Your job is to interpret a user's chat message and determine
    whether the user is correcting or adding complaint information.

    Use only the existing complaint data and the new chat message.

    Do not invent information.

    Return valid JSON only.

    Do not:
    - return Markdown
    - use code fences
    - include text outside JSON
    - update fields that the user did not mention

    Allowed complaint fields:

    - complainant_name
    - complainant_email
    - complainant_phone
    - customer_type
    - product_name
    - product_strength_grade
    - dosage_form
    - batch_lot_number
    - manufacturing_date
    - expiry_date
    - complaint_date
    - incident_date
    - country
    - complaint_description
    - observed_issue
    - quantity_affected
    - patient_involved
    - adverse_event_reported
    - patient_outcome
    - storage_conditions
    - supporting_evidence
    - source_reference
    - additional_information

    user_intent must be exactly one of:

    - UPDATE_FIELD
    - PROVIDE_INFORMATION
    - ASK_QUESTION
    - CONFIRM_INFORMATION
    - UNKNOWN

    updated_fields must contain only fields clearly supplied or
    corrected by the user.

    rejected_updates must contain information that cannot safely
    be mapped to a complaint field.

    When the user corrects an existing value, use the corrected
    value in updated_fields.
    """
).strip()


CHAT_USER_PROMPT_TEMPLATE = dedent(
    """
    Existing complaint data
    -----------------------
    {complaint_json}
    -----------------------

    User chat message
    -----------------------
    {chat_message}
    -----------------------

    Return exactly one JSON object:

    {{
      "user_intent": "UNKNOWN",
      "updated_fields": {{}},
      "rejected_updates": {{}},
      "clarification_required": false,
      "clarification_question": null,
      "assistant_message": null
    }}

    Return JSON only.
    """
).strip()


def build_chat_user_prompt(
    complaint_json: str,
    chat_message: str,
) -> str:
    cleaned_complaint_json = (
        complaint_json.strip()
    )

    cleaned_chat_message = (
        chat_message.strip()
    )

    if not cleaned_complaint_json:
        raise ValueError(
            "Complaint JSON cannot be empty."
        )

    if not cleaned_chat_message:
        raise ValueError(
            "Chat message cannot be empty."
        )

    return CHAT_USER_PROMPT_TEMPLATE.format(
        complaint_json=(
            cleaned_complaint_json
        ),
        chat_message=cleaned_chat_message,
    )
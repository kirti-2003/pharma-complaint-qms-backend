from textwrap import dedent


EXTRACTION_SYSTEM_PROMPT = dedent(
    """
    You are an AI assistant for a pharmaceutical customer complaint
    management system.

    Your task is to extract structured complaint information from the
    supplied text.

    The text may come from:
    - a customer complaint
    - an email
    - a PDF
    - an image converted to text
    - an internal complaint note
    - a distributor, hospital, pharmacy, or patient report

    Extraction rules:

    1. Extract only information explicitly stated in the source text.

    2. Do not guess, infer, assume, or invent missing information.

    3. If a value is not available, return null.

    4. Preserve important identifiers exactly as written, including:
       - batch numbers
       - lot numbers
       - product names
       - reference numbers
       - strengths
       - dates

    5. Do not classify the complaint or assess its risk in this step.

    6. Do not provide medical advice, regulatory conclusions, or a final
       quality decision.

    7. When the source text contains conflicting values:
       - use the latest clearly corrected value when explicitly stated
       - otherwise leave the field null
       - mention the conflict in additional_information

    8. For boolean fields:
       - return true only when the source clearly confirms the condition
       - return false only when the source clearly denies the condition
       - return null when the information is unclear or absent

    9. For supporting_evidence, include only evidence explicitly mentioned,
       such as:
       - product photographs
       - package photographs
       - invoices
       - complaint samples
       - medical records
       - labels
       - delivery records

    10. Return valid JSON only.

    11. Do not wrap the JSON in markdown code fences.

    12. The JSON keys must match the expected structured output schema.

    13. Do not include explanations before or after the JSON.

    14. Keep complaint_description factual and concise while preserving the
        original meaning.

    15. Use additional_information only for useful complaint details that do
        not map to a defined field.
    """
).strip()


EXTRACTION_USER_PROMPT_TEMPLATE = dedent(
    """
    Extract the pharmaceutical complaint information from the source text
    below.

    Source text:
    --------------------
    {raw_text}
    --------------------

    Return JSON using this exact structure:

    {{
      "complainant_name": null,
      "complainant_email": null,
      "complainant_phone": null,
      "customer_type": null,
      "product_name": null,
      "product_strength_grade": null,
      "dosage_form": null,
      "batch_lot_number": null,
      "manufacturing_date": null,
      "expiry_date": null,
      "complaint_date": null,
      "incident_date": null,
      "country": null,
      "complaint_description": null,
      "observed_issue": null,
      "quantity_affected": null,
      "patient_involved": null,
      "adverse_event_reported": null,
      "patient_outcome": null,
      "storage_conditions": null,
      "supporting_evidence": [],
      "source_reference": null,
      "additional_information": {{}}
    }}
    """
).strip()


def build_extraction_user_prompt(raw_text: str) -> str:
    """
    Build the user prompt used by the complaint extraction node.

    Args:
        raw_text: Complaint text extracted from manual input, email,
            PDF, image, or another supported source.

    Returns:
        A formatted extraction prompt for the Groq model.

    Raises:
        ValueError: If raw_text is empty or contains only whitespace.
    """

    cleaned_text = raw_text.strip()

    if not cleaned_text:
        raise ValueError("Complaint source text cannot be empty.")

    return EXTRACTION_USER_PROMPT_TEMPLATE.format(
        raw_text=cleaned_text,
    )
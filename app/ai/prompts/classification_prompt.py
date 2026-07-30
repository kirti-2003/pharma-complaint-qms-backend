from textwrap import dedent


CLASSIFICATION_SYSTEM_PROMPT = dedent(
    """
    You are a pharmaceutical Quality Management System
    complaint classification assistant.

    Classify the complaint using only the extracted complaint
    information supplied by the user.

    Do not invent facts that are not present in the input.

    Return one valid JSON object only.

    Do not:
    - return Markdown
    - wrap the response in code fences
    - include explanatory text outside the JSON
    - add fields outside the requested structure

    The JSON output must include exactly these fields:

    - complaint_category
    - complaint_subcategory
    - complaint_type
    - suggested_severity
    - is_quality_complaint
    - is_adverse_event
    - requires_immediate_attention
    - classification_confidence
    - classification_reasoning

    Complaint category guidance:

    Use a clear high-level category such as:

    - Product Quality
    - Packaging
    - Labeling
    - Delivery
    - Documentation
    - Service
    - Adverse Event

    Complaint type guidance:

    Use a clear complaint type such as:

    - API
    - Finished Dosage Form
    - Packaging
    - Medical Device
    - Service
    - Logistics

    Severity values:

    suggested_severity must be exactly one of:

    - MINOR
    - MAJOR
    - CRITICAL

    MINOR:

    - Administrative or documentation issue
    - Cosmetic issue with no meaningful product-quality impact
    - No patient-safety impact
    - Product remains usable and compliant
    - Investigation may not be necessary

    MAJOR:

    - Product defect
    - Discoloration
    - Breakage
    - Leakage
    - Packaging failure
    - Stability concern
    - Incorrect labeling
    - Possible contamination concern
    - Product quality may be affected
    - Investigation is required
    - No confirmed serious patient harm

    CRITICAL:

    - Confirmed or strongly suspected serious patient-safety hazard
    - Serious adverse event
    - Sterility failure
    - Significant contamination
    - Wrong product
    - Wrong strength
    - Potentially life-threatening defect
    - Immediate escalation is required

    Boolean field rules:

    - is_quality_complaint must be a JSON boolean:
      true or false

    - is_adverse_event must be a JSON boolean:
      true or false

    - requires_immediate_attention must be a JSON boolean:
      true or false

    Do not return boolean values as strings.

    Correct:
    "is_quality_complaint": true

    Incorrect:
    "is_quality_complaint": "true"

    Confidence rules:

    classification_confidence must be a JSON number
    between 0 and 1.

    Valid examples:

    - 0.65
    - 0.80
    - 0.95

    Invalid examples:

    - "High"
    - "Medium"
    - "Low"
    - "90%"
    - 90

    Consistency rules:

    - If complaint_category is Product Quality,
      is_quality_complaint must be true.

    - If the complaint describes a product defect,
      discoloration, breakage, leakage, contamination concern,
      packaging failure, or stability concern,
      is_quality_complaint should normally be true.

    - If an adverse event or patient injury is explicitly denied,
      is_adverse_event must be false.

    - If a possible adverse event is reported,
      is_adverse_event must be true.

    - If is_adverse_event is true,
      requires_immediate_attention should normally be true.

    - If suggested_severity is CRITICAL,
      requires_immediate_attention must be true.

    - Product discoloration in a sealed package should generally
      be classified as a product-quality complaint requiring
      investigation.

    - Product discoloration with no reported patient harm should
      generally be MAJOR rather than MINOR.

    - Do not leave classification_reasoning null when enough
      information exists to classify the complaint.

    - classification_reasoning should be concise and based only
      on the supplied complaint information.
    """
).strip()


CLASSIFICATION_USER_PROMPT_TEMPLATE = dedent(
    """
    Classify the following pharmaceutical complaint.

    Complaint Data
    -----------------------
    {complaint_json}
    -----------------------

    Return exactly one JSON object with this structure:

    {{
      "complaint_category": null,
      "complaint_subcategory": null,
      "complaint_type": null,
      "suggested_severity": null,
      "is_quality_complaint": false,
      "is_adverse_event": false,
      "requires_immediate_attention": false,
      "classification_confidence": 0.0,
      "classification_reasoning": null
    }}

    Field requirements:

    - suggested_severity must be exactly:
      MINOR, MAJOR, or CRITICAL

    - classification_confidence must be a number
      between 0 and 1

    - is_quality_complaint must be true or false

    - is_adverse_event must be true or false

    - requires_immediate_attention must be true or false

    - Return valid JSON only

    - Do not return Markdown

    - Do not use code fences

    - Do not include text before or after the JSON
    """
).strip()


def build_classification_user_prompt(
    complaint_json: str,
) -> str:
    """
    Build the user prompt used by the classification node.
    """

    cleaned_complaint_json = (
        complaint_json.strip()
    )

    if not cleaned_complaint_json:
        raise ValueError(
            "Complaint JSON cannot be empty."
        )

    return (
        CLASSIFICATION_USER_PROMPT_TEMPLATE
        .format(
            complaint_json=(
                cleaned_complaint_json
            ),
        )
    )
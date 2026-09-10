from textwrap import dedent


ASSESSMENT_SYSTEM_PROMPT = dedent(
    """
    You are a pharmaceutical Quality Management System assistant.

    Classify the supplied complaint and produce a preliminary
    risk assessment for human QA review.

    Use only the supplied complaint information.
    Do not invent missing facts.

    Classification requirements:
    - category and subcategory
    - complaint type
    - severity: MINOR, MAJOR, or CRITICAL
    - whether it is a quality complaint
    - whether it reports an adverse event
    - whether it requires immediate attention
    - concise reasoning
    - numeric classification confidence

    Risk requirements:
    - risk level: LOW, MEDIUM, HIGH, or CRITICAL
    - patient safety risk
    - product quality risk
    - regulatory risk
    - business risk
    - escalation requirements
    - sample collection requirement
    - batch investigation requirement
    - adverse-event review requirement
    - suggested next action
    - recommended actions
    - risk factors
    - concise reasoning
    - numeric risk confidence

    Confidence rules:
    - classification_confidence must be a valid JSON number
      between 0.0 and 1.0.
    - risk_confidence must be a valid JSON number
      between 0.0 and 1.0.
    - Use digits only for confidence values.
    - Valid examples: 0.25, 0.5, 0.85, 0.9, 1.0
    - Invalid examples: "0.9", "high", "nine",
      "0. nine", "90%"
    - Never spell numbers using words.
    - Never include units, labels, or percentages
      in confidence fields.

    Important consistency rules:
    - A reported adverse event should normally require immediate review.
    - Wrong product, wrong strength, sterility failure, confirmed
      contamination, or serious patient harm should normally be CRITICAL.
    - A meaningful product defect without serious reported harm should
      normally be MAJOR or HIGH depending on the evidence.
    - Do not treat the output as a final regulatory decision.

    Output rules:
    - Return every field required by the schema.
    - Use null for unavailable nullable scalar fields.
    - Use [] for unavailable list fields.
    - Return valid JSON only.
    - Do not return Markdown or code fences.
    - Do not return any explanation outside the JSON object.
    """
).strip()


ASSESSMENT_USER_PROMPT_TEMPLATE = dedent(
    """
    Assess this pharmaceutical complaint:

    {complaint_json}

    Return exactly one JSON object using this structure:

    {{
      "classification": {{
        "complaint_category": null,
        "complaint_subcategory": null,
        "complaint_type": null,
        "suggested_severity": null,
        "is_quality_complaint": false,
        "is_adverse_event": false,
        "requires_immediate_attention": false,
        "classification_confidence": 0.9,
        "classification_reasoning": null
      }},
      "risk_assessment": {{
        "risk_level": null,
        "patient_safety_risk": null,
        "product_quality_risk": null,
        "regulatory_risk": null,
        "business_risk": null,
        "requires_escalation": false,
        "requires_sample_collection": false,
        "requires_batch_investigation": false,
        "requires_adverse_event_review": false,
        "suggested_next_action": null,
        "recommended_actions": [],
        "risk_factors": [],
        "risk_confidence": 0.80,
        "risk_reasoning": null
      }}
    }}

    Replace example values with values supported by the complaint.
    """
).strip()


def build_assessment_user_prompt(
    complaint_json: str,
) -> str:
    cleaned = complaint_json.strip()

    if not cleaned:
        raise ValueError(
            "Complaint JSON cannot be empty."
        )

    return ASSESSMENT_USER_PROMPT_TEMPLATE.format(
        complaint_json=cleaned
    )
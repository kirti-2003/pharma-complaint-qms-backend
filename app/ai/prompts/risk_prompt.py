from textwrap import dedent


RISK_SYSTEM_PROMPT = dedent(
    """
    You are a pharmaceutical Quality Management System assistant
    performing an initial complaint risk assessment.

    This is a preliminary AI recommendation for QA review.
    It is not a final regulatory or quality decision.

    Assess:

    - patient safety risk
    - product quality risk
    - regulatory risk
    - business risk
    - need for escalation
    - need for sample collection
    - need for batch investigation
    - need for adverse-event review
    - suggested next action
    - recommended actions
    - risk factors
    - confidence
    - reasoning

    Risk rules:

    LOW:
    - Administrative or cosmetic issue
    - No evidence of product-quality impact
    - No patient safety impact
    - Investigation may not be required

    MEDIUM:
    - Possible quality impact
    - Limited affected quantity
    - No reported patient harm
    - Investigation or follow-up is recommended

    HIGH:
    - Significant product defect
    - Packaging integrity concern
    - Stability concern
    - Possible contamination
    - Multiple units or batches may be affected
    - Immediate QA investigation is recommended

    CRITICAL:
    - Serious patient safety risk
    - Confirmed contamination
    - Sterility failure
    - Wrong product or wrong strength
    - Serious adverse event or life-threatening concern

    Additional rules:

    - Discoloration of pharmaceutical capsules in a sealed bottle may indicate
      moisture ingress, stability failure, packaging seal failure, or another
      product-quality defect.
    - Such complaints should generally recommend sample collection and batch
      investigation even when no adverse event is reported.
    - Do not leave suggested_next_action empty when investigation is required.
    - Include concise risk_reasoning.
    - Do not invent facts.
    - Return valid JSON only.
    - Do not return markdown.
    """
).strip()


RISK_USER_PROMPT_TEMPLATE = dedent(
    """
    Assess the initial risk for the following pharmaceutical complaint.

    Complaint Information
    ----------------------
    {complaint_json}
    ----------------------

    Return JSON using exactly this structure:

    {{
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
      "risk_confidence": null,
      "risk_reasoning": null
    }}
    """
).strip()


def build_risk_user_prompt(
    complaint_json: str,
) -> str:

    cleaned = complaint_json.strip()

    if not cleaned:
        raise ValueError(
            "Complaint JSON cannot be empty."
        )

    return RISK_USER_PROMPT_TEMPLATE.format(
        complaint_json=cleaned,
    )
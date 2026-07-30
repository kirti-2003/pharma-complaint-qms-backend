from app.ai.prompts.extraction_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
    build_extraction_user_prompt,
)

from app.ai.prompts.classification_prompt import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT_TEMPLATE,
    build_classification_user_prompt,
)

from app.ai.prompts.risk_prompt import (
    RISK_SYSTEM_PROMPT,
    RISK_USER_PROMPT_TEMPLATE,
    build_risk_user_prompt,
)

from app.ai.prompts.chat_prompt import (
    CHAT_SYSTEM_PROMPT,
    CHAT_USER_PROMPT_TEMPLATE,
    build_chat_user_prompt,
)


__all__ = [
    # Extraction
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_USER_PROMPT_TEMPLATE",
    "build_extraction_user_prompt",

    # Classification
    "CLASSIFICATION_SYSTEM_PROMPT",
    "CLASSIFICATION_USER_PROMPT_TEMPLATE",
    "build_classification_user_prompt",

    # Risk
    "RISK_SYSTEM_PROMPT",
    "RISK_USER_PROMPT_TEMPLATE",
    "build_risk_user_prompt",

    # Chat
    "CHAT_SYSTEM_PROMPT",
    "CHAT_USER_PROMPT_TEMPLATE",
    "build_chat_user_prompt",
]
from functools import lru_cache

from app.services.ai_service import (
    AIService,
)


@lru_cache
def get_ai_service() -> AIService:
    """
    Provide a reusable AI service instance.
    """

    return AIService()
from typing import Any

from groq import Groq

from app.core.config import settings


class GroqClient:
    """
    Wrapper around the Groq API client.

    This class is responsible only for communicating with Groq.
    Complaint workflow and database logic remain outside this class.
    """

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing from environment settings."
            )

        self.client = Groq(
            api_key=settings.GROQ_API_KEY,
        )

        self.default_model = settings.GROQ_MODEL
        self.fallback_model = settings.GROQ_FALLBACK_MODEL

    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """
        Send a chat-completion request to Groq.

        Returns generated text and token-usage information.
        """

        selected_model = model or self.default_model

        try:
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = (
                response.choices[0].message.content or ""
            ).strip()

            usage = response.usage

            return {
                "content": content,
                "model": selected_model,
                "prompt_tokens": (
                    usage.prompt_tokens if usage else None
                ),
                "completion_tokens": (
                    usage.completion_tokens if usage else None
                ),
                "total_tokens": (
                    usage.total_tokens if usage else None
                ),
            }

        except Exception as exc:
            raise RuntimeError(
                f"Groq completion failed: {str(exc)}"
            ) from exc


groq_client = GroqClient()
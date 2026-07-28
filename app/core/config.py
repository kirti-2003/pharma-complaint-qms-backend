from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Values are read from the root-level .env file.
    """

    # Application settings
    app_name: str = "Pharma Complaint QMS"
    app_environment: str = "development"
    debug: bool = True

    # Database settings
    database_url: str

    # Frontend configuration
    frontend_url: str = "http://localhost:5173"

    # File upload configuration
    upload_directory: str = "uploads/complaints"
    max_upload_size_mb: int = 10

    # Groq configuration
    groq_api_key: str | None = None
    groq_model: str = "gemma2-9b-it"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached application settings instance.

    The .env file is read once and the same Settings object is reused
    throughout the application.
    """

    return Settings()


settings = get_settings()
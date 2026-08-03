"""Environment-backed application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for deployment-specific settings and secrets."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "https://ollama.com"
    ollama_api_key: str
    intent_model: str = "qwen2.5:7b-cloud"
    grounding_judge_model: str = "qwen2.5:7b-cloud"
    response_model: str = "gpt-oss:120b-cloud"
    confidence_threshold: float = 0.75
    sentiment_escalation_threshold: float = -0.4
    max_clarifications: int = 2
    whisper_model: str = "base.en"
    tts_provider: str = "coqui"
    elevenlabs_api_key: str | None = None
    vector_db_path: str = "./data/chroma"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    database_url: str = "sqlite:///./sessions.db"
    api_auth_secret: str
    allowed_origins: list[str] = ["http://localhost:3000"]
    rate_limit_per_minute: int = 60
    max_call_duration_seconds: int = 900


# Pydantic reads these required values from its configured environment source at runtime.
settings = Settings()  # type: ignore[call-arg]

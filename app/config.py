"""Environment-backed application configuration."""

import json
from typing import Any, Literal

from pydantic import field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class _OriginEnvSettingsSource(EnvSettingsSource):
    """Leave origin strings undecoded so the settings validator can parse CSV values."""

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        if field_name == "allowed_origins" and isinstance(value, str):
            return value
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class _OriginDotEnvSettingsSource(DotEnvSettingsSource):
    """Apply the same CSV handling to values loaded from `.env`."""

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        if field_name == "allowed_origins" and isinstance(value, str):
            return value
        return super().prepare_field_value(field_name, field, value, value_is_complex)


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
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: str = "int8"
    tts_provider: Literal["coqui", "elevenlabs"] = "coqui"
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model_id: str = "eleven_multilingual_v2"
    vector_db_path: str = "./data/chroma"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    database_url: str = "sqlite:///./sessions.db"
    api_auth_secret: str
    allowed_origins: list[str] = ["http://localhost:3000"]
    rate_limit_per_minute: int = 60
    max_call_duration_seconds: int = 900
    environment: str = "development"
    log_level: str = "INFO"
    debug_logging: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use sources that preserve the documented CSV form of `ALLOWED_ORIGINS`."""
        return (
            init_settings,
            _OriginEnvSettingsSource(settings_cls),
            _OriginDotEnvSettingsSource(settings_cls),
            file_secret_settings,
        )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: Any) -> list[str]:
        """Accept the documented comma-separated env value as well as a JSON array."""
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            raise ValueError("ALLOWED_ORIGINS must be a comma-separated string or list.")
        text = value.strip()
        if text.startswith("["):
            decoded = json.loads(text)
            if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
                return decoded
            raise ValueError("ALLOWED_ORIGINS JSON value must contain only strings.")
        origins = [origin.strip() for origin in text.split(",") if origin.strip()]
        if not origins:
            raise ValueError("ALLOWED_ORIGINS must include at least one origin.")
        return origins

    @model_validator(mode="after")
    def validate_tts_configuration(self) -> "Settings":
        """Fail application startup instead of silently changing a selected TTS provider."""
        if self.tts_provider == "elevenlabs" and not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs.")
        return self


# Pydantic reads these required values from its configured environment source at runtime.
settings = Settings()  # type: ignore[call-arg]

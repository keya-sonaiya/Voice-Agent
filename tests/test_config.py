from app.config import Settings


def test_allowed_origins_accepts_documented_comma_separated_value() -> None:
    settings = Settings(
        ollama_api_key="test-key",
        api_auth_secret="test-secret",
        allowed_origins="http://localhost:3000, https://support.example.com",
    )
    assert settings.allowed_origins == ["http://localhost:3000", "https://support.example.com"]


def test_allowed_origins_accepts_json_array() -> None:
    settings = Settings(
        ollama_api_key="test-key",
        api_auth_secret="test-secret",
        allowed_origins='["http://localhost:3000"]',
    )
    assert settings.allowed_origins == ["http://localhost:3000"]

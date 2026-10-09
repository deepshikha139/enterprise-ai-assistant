from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, found from this file's location so it works from any folder.
ENV_FILE = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    """App settings, read from environment variables or backend/.env."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8")

    # Required: no default, so startup fails with a clear error if it's missing.
    # SecretStr hides the value when printed or logged.
    anthropic_api_key: SecretStr
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 1024
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 30.0
    # The one website address allowed to call this API from a browser.
    cors_allowed_origin: str = "http://localhost:3000"


settings = Settings()

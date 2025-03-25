import os

from pydantic_settings import BaseSettings
from pydantic import SecretStr
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class AuthSettings(BaseSettings):
    """Authentication-related settings loaded from environment variables."""

    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: SecretStr = SecretStr(os.getenv("GOOGLE_CLIENT_SECRET"))
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/callback/google"

    SCOPES: dict = {
        "user": "Read information about the current user.",
        "requests": "Read and create audio processing requests.",
    }

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "allow"}


@lru_cache()
def get_auth_settings() -> AuthSettings:
    """
    Get cached authentication settings.

    Returns:
        AuthSettings: Authentication configuration
    """
    return AuthSettings()

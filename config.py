import os

from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    """Application settings."""

    # Core settings
    APP_NAME: str = "Audio-to-Text API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "An API for processing YouTube videos and converting audio to text"
    )
    SECRET_KEY: str = os.environ.get("SECRET_KEY")
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")
    DEBUG: bool = os.environ.get("DEBUG", "True").lower() in ("true", "1", "t")

    # API settings
    API_PORT: int = int(os.environ.get("API_PORT", 8000))
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")

    # WebSocket settings
    WS_PORT: int = int(os.environ.get("WS_PORT", 9090))
    WS_HOST: str = os.environ.get("WS_HOST", "127.0.0.1")

    # Database settings - check if we have all required database parameters
    DB_USER: str = os.environ.get("DB_USER", "")
    DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "")
    DB_HOST: str = os.environ.get("DB_HOST", "")
    DB_PORT: str = os.environ.get("DB_PORT", "")
    DB_NAME: str = os.environ.get("DB_NAME", "")
    DB_SSL: bool = os.environ.get("DB_SSL", "false").lower() in ("true", "1", "t")

    # Compute database URI based on environment and available parameters
    @property
    def DATABASE_URL(self) -> str:
        """Generate database connection URL"""
        # If all PostgreSQL parameters are available
        if all(
            [self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_PORT, self.DB_NAME]
        ):
            uri = f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            if self.DB_SSL:
                uri += "?sslmode=require"
            return uri

        # Otherwise use SQLite based on environment
        if self.ENVIRONMENT == "testing":
            return "sqlite:///:memory:"
        elif self.ENVIRONMENT == "production":
            print(
                "WARNING: Using SQLite in production due to missing database parameters!"
            )
            return "sqlite:///production.db"
        else:  # development
            return "sqlite:///dev.db"

    # OpenAI settings (if used)
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "allow"}


# Create a settings instance
settings = Settings()

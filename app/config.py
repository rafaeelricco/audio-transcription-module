import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def create_sqlite_uri(database):
    return f"sqlite:///{os.path.join(BASEDIR, database)}"


def get_postgres_uri():
    """Build PostgreSQL connection URI from environment variables"""
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT")
    db_name = os.environ.get("DB_NAME")

    if all([db_user, db_password, db_host, db_port, db_name]):
        uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        if os.environ.get("DB_SSL") == "true":
            uri += "?sslmode=require"
        return uri
    return None


class Settings(BaseSettings):
    """Base settings for the application"""

    # Application settings
    APP_NAME: str = "Audio-to-Text API"
    APP_VERSION: str = "1.0.0"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-key-please-change-in-production")

    # Environment settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

    # Database settings
    SQLALCHEMY_DATABASE_URI: str = get_postgres_uri() or (
        create_sqlite_uri("development.db")
        if ENVIRONMENT == "development"
        else (
            create_sqlite_uri("testing.db")
            if ENVIRONMENT == "testing"
            else create_sqlite_uri("production.db")
        )
    )

    # API settings
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")

    # WebSocket settings
    WEBSOCKET_PORT: int = int(os.getenv("WEBSOCKET_PORT", 9090))
    WEBSOCKET_HOST: str = os.getenv("WEBSOCKET_HOST", "127.0.0.1")

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings():
    """Return cached settings instance"""
    return Settings()

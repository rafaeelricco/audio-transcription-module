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
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")

    if all([db_user, db_password, db_host, db_port, db_name]):
        uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        if os.environ.get("DB_SSL") == "true":
            uri += "?sslmode=require"
        return uri
    return None


class Settings(BaseSettings):
    """Base settings for the application"""

    APP_NAME: str = "Audio-to-Text API"
    APP_VERSION: str = "1.0.0"

    API_HOST: str = os.getenv("API_HOST")
    WEBSOCKET_HOST: str = os.getenv("WEBSOCKET_HOST")

    ENVIRONMENT: str = os.getenv("ENV")
    DEBUG: bool = os.getenv("DEBUG").lower() in ("true", "1", "t")

    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: str = os.getenv("DB_PORT")
    DB_NAME: str = os.getenv("DB_NAME")
    DB_SSL: bool = os.getenv("DB_SSL").lower() in ("true", "1", "t")

    SQLALCHEMY_DATABASE_URI: str = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        if all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME])
        else get_postgres_uri()
        or (
            create_sqlite_uri("development.db")
            if ENVIRONMENT == "development"
            else (
                create_sqlite_uri("testing.db")
                if ENVIRONMENT == "testing"
                else create_sqlite_uri("production.db")
            )
        )
    )

    API_PORT: int = int(os.getenv("APP_PORT"))

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "allow"}


@lru_cache()
def get_settings():
    """Return cached settings instance"""
    return Settings()

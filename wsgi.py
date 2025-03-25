import sys
import logging
import uvicorn

from dotenv import load_dotenv
from pathlib import Path
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


def start_servers():
    """Start both the WebSocket server and FastAPI application"""
    uvicorn.run(
        "app.main:app",
        ws="websockets",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "error",
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "* %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": True,
                },
            },
        },
    )


if __name__ == "__main__":
    start_servers()

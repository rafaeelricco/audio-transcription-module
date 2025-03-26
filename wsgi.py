import sys
import logging
import gunicorn.app.base

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
    """Start both the WebSocket server and FastAPI application using Gunicorn"""

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        "bind": f"{settings.APP_HOST}:{settings.APP_PORT}",
        "workers": 4,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "reload": settings.DEBUG,
        "loglevel": "info" if settings.DEBUG else "error",
    }

    StandaloneApplication("app.main:app", options).run()


if __name__ == "__main__":
    start_servers()

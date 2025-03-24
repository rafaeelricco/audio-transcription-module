import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def create_sqlite_uri(database):
    return f"sqlite:///{os.path.join(BASEDIR, database)}"


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    SQLALCHEMY_RECORD_QUERIES = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = create_sqlite_uri("development.db")


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = create_sqlite_uri("testing.db")


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = create_sqlite_uri("production.db")


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

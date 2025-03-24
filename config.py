import os


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-please-change-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True

    # Check if we have all required database parameters
    if all(
        [
            os.environ.get("DB_USER"),
            os.environ.get("DB_PASSWORD"),
            os.environ.get("DB_HOST"),
            os.environ.get("DB_PORT"),
            os.environ.get("DB_NAME"),
        ]
    ):
        # Database configuration using PostgreSQL
        SQLALCHEMY_DATABASE_URI = (
            f"postgresql://{os.environ.get('DB_USER')}:"
            f"{os.environ.get('DB_PASSWORD')}@"
            f"{os.environ.get('DB_HOST')}:"
            f"{os.environ.get('DB_PORT')}/"
            f"{os.environ.get('DB_NAME')}"
        )

        if os.environ.get("DB_SSL") == "true":
            SQLALCHEMY_DATABASE_URI += "?sslmode=require"
    else:
        # Use SQLite for development if PostgreSQL params are missing
        SQLALCHEMY_DATABASE_URI = "sqlite:///dev.db"


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


class ProductionConfig(Config):
    """Production configuration."""

    # Check if we have all required database parameters
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT")
    db_name = os.environ.get("DB_NAME")

    if all([db_user, db_password, db_host, db_port, db_name]):
        # Database configuration using PostgreSQL
        SQLALCHEMY_DATABASE_URI = (
            f"postgresql://{db_user}:"
            f"{db_password}@"
            f"{db_host}:"
            f"{db_port}/"
            f"{db_name}"
        )

        if os.environ.get("DB_SSL") == "true":
            SQLALCHEMY_DATABASE_URI += "?sslmode=require"
    else:
        # Fail safe for production if db params are missing
        SQLALCHEMY_DATABASE_URI = "sqlite:///production.db"
        print("WARNING: Using SQLite in production due to missing database parameters!")


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

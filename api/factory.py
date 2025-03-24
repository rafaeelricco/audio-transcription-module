from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_mail import Mail
from config import config

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
cors = CORS(resources={r"/*/*": {"origins": "*"}})
mail = Mail()


login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "auth.login"


def create_app(config_name):
    app = Flask(__name__)

    app.config["MAIL_SERVER"] = "smtp.mailtrap.io"
    app.config["MAIL_PORT"] = 465
    app.config["MAIL_USE_TLS"] = True
    app.config["MAIL_USE_SSL"] = False
    app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
    app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
    app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    cors.init_app(app)
    mail.init_app(app)

    from app.mail import mail as mail_blueprint
    from app.auth import auth as auth_blueprint

    app.register_blueprint(mail_blueprint)
    app.register_blueprint(auth_blueprint, url_prefix="/auth")

    return app

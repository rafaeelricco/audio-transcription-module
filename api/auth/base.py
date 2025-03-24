from datetime import timedelta

from flask import request, jsonify, make_response
from flask_login import logout_user, login_user

from app.auth import auth
from app.factory import bcrypt, db
from app.models.base import User
from app.mail.base import send_email_confirmation
import re

# Route to register a new user
@auth.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data:
        return make_response(jsonify({"message": "No input data provided"}), 400)
    if not data.get("username"):
        return make_response(jsonify({"message": "Username is required"}), 400)
    if not data.get("email"):
        return make_response(jsonify({"message": "Email is required"}), 400)
    if not data.get("password"):
        return make_response(jsonify({"message": "Password is required"}), 400)

    """ Regex to check valid email """
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    """ Check if the email is valid, compare with regex """
    if not re.match(email_regex, data.get("email")):
        return make_response(jsonify({"message": "The email is not valid"}), 400)

    """ Check if the username is valid, compare with regex """
    user = User.query.filter_by(username=data["username"]).first()
    if user is not None:
        return make_response(jsonify({"message": "User already exists"}), 400)
    else:
        send_email_confirmation(data["email"], data["username"])
        new_user = User(
            username=data["username"],
            email=data["email"],
            password=bcrypt.generate_password_hash(data["password"]).decode("utf-8"),
        )
        db.session.add(new_user)
        db.session.commit()
        return make_response(jsonify({"message": "User created successfully"}), 201)


# Route to login a user
@auth.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data["email"]).first()
    if user is None:
        return make_response(jsonify({"message": "User does not exist"}), 404)

    if not bcrypt.check_password_hash(user.password, data["password"]):
        return make_response(
            jsonify({"message": "Password or email is incorrect"}), 401
        )
    else:
        login_user(user, remember=True, duration=timedelta(days=7))
        return make_response(
            jsonify(
                {"message": "User logged in successfully, token is valid for 7 days"}
            ),
            200,
        )


# Route to logout a user
@auth.route("/logout", methods=["GET", "POST"])
def logout():
    logout_user()
    return make_response(jsonify({"message": "Logged out successfully"}), 200)

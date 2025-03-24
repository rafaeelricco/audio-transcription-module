import os
from flask import make_response, jsonify, request, render_template, url_for
from flask_mail import Message, Mail
from itsdangerous import URLSafeTimedSerializer

from app.factory import db
from app.mail import mail
from app.models.base import User
from app.factory import login_manager


secret_key = URLSafeTimedSerializer(os.getenv("SECRET_KEY"))

# This callback is used to reload the user object from the user ID stored in the session.
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# function to send email confirmation
def send_email_confirmation(email, username):
    data = request.get_json()
    email = data["email"]
    username = data["username"]
    mail = Mail()
    msg = Message(
        "Ativação de conta | Flask Login",
        recipients=[email],
    )
    link = url_for(
        "mail.confirm_email",
        token=secret_key.dumps(email, salt="email-confirm"),
        _external=True,
    )

    msg.html = render_template("email_confirmation.html", link=link, username=username)

    mail.send(msg)
    return make_response(jsonify({"message": "Email sent, verify your email"}), 200)


# Route to confirm email
@mail.route("/confirm/<token>", methods=["GET"])
def confirm_email(token):
    email = secret_key.loads(token, salt="email-confirm", max_age=3600)
    user = User.query.filter_by(email=email).first_or_404()
    if user.email_confirmed:
        return make_response(jsonify({"message": "Email already confirmed"}), 400)
    else:
        user.email_confirmed = True
        db.session.add(user)
        db.session.commit()
        return make_response(jsonify({"message": "Email confirmed successfully"}), 200)

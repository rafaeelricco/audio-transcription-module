from flask_login import UserMixin

from app.factory import db


# Table of users
class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    email_confirmed = db.Column(db.Boolean, default=False)
    password = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return "<User %r>" % self.username

    def get_id(self):
        return self.id

    def get_username(self):
        return self.username

    def get_email(self):
        return self.email

    def get_email_confirmed(self):
        return self.email_confirmed

import uuid
from flask_login import UserMixin

from app.factory import db


class User(db.Model, UserMixin):
    """User model representing a person who submits processing requests."""

    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)

    # Relationship to processing requests
    requests = db.relationship("ProcessingRequest", backref="user", lazy="dynamic")

    def __repr__(self):
        return f"<User {self.name} ({self.email})>"

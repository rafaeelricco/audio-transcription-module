import uuid
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from flask_login import UserMixin

from app.factory import db


class User(db.Model, UserMixin):
    """User model representing a person who submits processing requests."""

    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)

    requests = db.relationship("ProcessingRequest", backref="user", lazy="dynamic")

    def __repr__(self):
        return f"<User {self.name} ({self.email})>"


class ProcessingRequest(db.Model):
    """Model for audio processing requests with YouTube URLs."""

    __tablename__ = "processing_requests"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    urls = db.Column(db.JSON, nullable=False)
    result = db.Column(db.Text, nullable=True)

    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)

    def __repr__(self):
        return f"<ProcessingRequest {self.id} - {len(self.urls)} URLs>"

    @property
    def serialize(self):
        """Return object data in easily serializable format"""
        return {
            "id": self.id,
            "urls": self.urls,
            "user": {
                "id": self.user.id,
                "name": self.user.name,
                "email": self.user.email,
            },
            "result": self.result,
        }

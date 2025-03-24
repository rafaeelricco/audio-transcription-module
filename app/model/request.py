import uuid
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from app.factory import db
from app.model.user import User


class ProcessingRequest(db.Model):
    """Model for audio processing requests with YouTube URLs."""

    __tablename__ = "processing_requests"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    urls = db.Column(db.JSON, nullable=False)
    result = db.Column(db.Text, nullable=True)

    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)

    def __repr__(self):
        """
        Return a string representation of the ProcessingRequest object.

        Format: <ProcessingRequest <id> - <n> URLs>
        """
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

import uuid
from sqlalchemy import Column, String, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.db.database import Base


class ProcessingRequest(Base):
    """Model for audio processing requests with YouTube URLs."""

    __tablename__ = "processing_requests"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String(512), nullable=False)
    status = Column(String(32), default='pending', nullable=False)
    result = Column(Text, nullable=True)

    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="requests")

    def __repr__(self):
        """
        Return a string representation of the ProcessingRequest object.

        Format: <ProcessingRequest <id> - <url>>
        """
        return f"<ProcessingRequest {self.id} - {self.url}>"

    @property
    def serialize(self):
        """Return object data in easily serializable format"""
        return {
            "id": self.id,
            "url": self.url,
            "user": {
                "id": self.user.id,
                "name": self.user.name,
                "email": self.user.email,
            },
            "result": self.result,
            "status": self.status,
        }

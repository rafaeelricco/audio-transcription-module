import uuid
from sqlalchemy import Column, String, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base


class ProcessingRequest(Base):
    """Model for audio processing requests with YouTube URLs."""

    __tablename__ = "processing_requests"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    urls = Column(JSON, nullable=False)
    result = Column(Text, nullable=True)

    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="requests")

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

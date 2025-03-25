import uuid
from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """User model representing a person who submits processing requests."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(120), nullable=False)
    email = Column(String(120), nullable=False, unique=True)

    # Relationship to processing requests
    requests = relationship("ProcessingRequest", back_populates="user")

    def __repr__(self):
        return f"<User {self.name} ({self.email})>"

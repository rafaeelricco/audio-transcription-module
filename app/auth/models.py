"""Models for authentication."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    """Schema for OAuth token response."""
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    """Schema for token payload data."""
    email: Optional[EmailStr] = None
    scopes: list[str] = []
    exp: Optional[datetime] = None
    
class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    name: str = Field(..., min_length=2, max_length=120)
    
class UserResponse(BaseModel):
    """Schema for user data that's safe to return in API responses."""
    id: str
    name: str
    email: str
    
    class Config:
        orm_mode = True
        
class GoogleUserInfo(BaseModel):
    """Schema for Google user info response."""
    email: EmailStr
    name: str
    picture: Optional[str] = None
    sub: str  # Google's unique identifier for the user

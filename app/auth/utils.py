"""Authentication utilities for the application."""

from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from sqlalchemy.orm import Session

from app.auth.config import get_auth_settings
from app.auth.models import TokenData, UserResponse
from app.database import get_db
from app.model.user import User

# Get authentication settings
auth_settings = get_auth_settings()

# Define OAuth2 scheme with scopes
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes=auth_settings.SCOPES
)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token for a user.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional token expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    # Set token expiration
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=auth_settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    # Encode and return token
    encoded_jwt = jwt.encode(
        to_encode, 
        auth_settings.SECRET_KEY, 
        algorithm=auth_settings.ALGORITHM
    )
    
    return encoded_jwt

async def get_current_user(
    security_scopes: SecurityScopes, 
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Validate the access token and return the current user.
    
    Args:
        security_scopes: Security scopes required for this endpoint
        token: JWT token
        db: Database session
        
    Returns:
        User: Current user object
        
    Raises:
        HTTPException: If token is invalid or user doesn't have required permissions
    """
    # Prepare authentication error message
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        # Decode token
        payload = jwt.decode(
            token, 
            auth_settings.SECRET_KEY, 
            algorithms=[auth_settings.ALGORITHM]
        )
        
        # Extract email from token
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
            
        # Extract scopes from token
        token_scopes = payload.get("scopes", [])
        
        # Create token data object
        token_data = TokenData(
            email=email,
            scopes=token_scopes,
            exp=datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
        )
        
    except JWTError:
        raise credentials_exception
        
    # Get user from database
    user = db.query(User).filter(User.email == token_data.email).first()
    
    if user is None:
        raise credentials_exception
        
    # Verify user has all required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
            
    return user

def get_user_response(user: User) -> UserResponse:
    """
    Convert User model to UserResponse schema.
    
    Args:
        user: User model instance
        
    Returns:
        UserResponse: User data safe for API responses
    """
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email
    )

"""Authentication router for Google OAuth."""

import uuid
import json

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Request,
    Response,
    Security,
)
from sqlalchemy.orm import Session
from authlib.integrations.starlette_client import OAuth, OAuthError
from httpx import AsyncClient
from app.config import get_settings
from app.model.auth import UserResponse, GoogleUserInfo
from app.auth.utils import create_access_token, get_current_user, get_user_response
from app.db.database import get_db
from app.model.user import User

auth_settings = get_settings()

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)

oauth = OAuth()
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=auth_settings.GOOGLE_CLIENT_ID,
    client_secret=auth_settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
    client_kwargs={
        "scope": "openid email profile",
        "redirect_uri": auth_settings.GOOGLE_REDIRECT_URI,
    },
)


@router.get("/login/google")
async def login_via_google(request: Request):
    """
    Start Google OAuth flow by redirecting to Google login page.

    Args:
        request: FastAPI request object

    Returns:
        RedirectResponse: Redirect to Google authentication
    """
    redirect_uri = request.url_for("callback_google")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback/google")
async def callback_google(request: Request, db: Session = Depends(get_db)):
    """
    Handle Google OAuth callback after user has authenticated.

    Args:
        request: FastAPI request object
        db: Database session

    Returns:
        Token: JWT access token

    Raises:
        HTTPException: If OAuth flow fails
    """
    try:
        token = await oauth.google.authorize_access_token(request)

        async with AsyncClient() as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {token['access_token']}"},
            )
            user_data = resp.json()

        user_info = GoogleUserInfo(
            email=user_data.get("email"),
            name=user_data.get("name"),
            picture=user_data.get("picture"),
            sub=user_data.get("sub"),
        )

        user = db.query(User).filter(User.email == user_info.email).first()

        if user is None:
            user = User(
                id=str(uuid.uuid4()), email=user_info.email, name=user_info.name
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        access_token = create_access_token(
            data={
                "sub": user.email,
                "scopes": ["user", "requests"],
            }
        )

        # Save the access token to the user's record in the database
        user.access_token = access_token
        db.commit()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Successful</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; line-height: 1.6; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .token-box {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; margin: 20px 0; text-align: left; word-break: break-all; }}
                .success {{ color: #28a745; }}
                button {{ background-color: #007bff; color: white; border: none; padding: 10px 15px; 
                         border-radius: 4px; cursor: pointer; margin-top: 20px; }}
                button:hover {{ background-color: #0069d9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="success">Authentication Successful!</h1>
                <p>You've successfully authenticated with Google.</p>
                <p>Your access token is:</p>
                <div class="token-box">{access_token}</div>
                <p>This token has been stored in your browser and will be used for API requests.</p>
                <button onclick="window.close()">Close this window</button>
            </div>
            <script>
                // Store the token in localStorage
                localStorage.setItem('auth_token', '{access_token}');
                // You can redirect to your main application here if needed
                // window.location.href = '/';
            </script>
        </body>
        </html>
        """

        return Response(content=html_content, media_type="text/html")

    except OAuthError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth error: {error.error}",
        )


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Security(get_current_user, scopes=["user"])
):
    """
    Get information about the currently authenticated user.

    Args:
        current_user: Currently authenticated user (from token)

    Returns:
        UserResponse: Current user information
    """
    return get_user_response(current_user)


@router.get("/verify", response_model=dict)
async def verify_token(current_user: User = Security(get_current_user, scopes=[])):
    """
    Verify that the current token is valid.

    Args:
        current_user: Currently authenticated user (from token)

    Returns:
        dict: Token verification status
    """
    return {"status": "valid", "user": get_user_response(current_user)}

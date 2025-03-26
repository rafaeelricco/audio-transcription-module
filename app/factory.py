from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from app.config import get_settings
from dotenv import load_dotenv
from app.db.database import init_db

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

settings = get_settings()

app = FastAPI(
    title="Audio-to-Text API",
    description="An API for processing YouTube videos and converting audio to text",
    version="1.0.0",
)

app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/", tags=["root"])
async def root():
    """Root endpoint that provides API information and redirects to docs."""
    return {
        "message": "Welcome to the Audio-to-Text API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {"api": "/api", "auth": "/auth"},
    }


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_redirect():
    """Redirect to the Swagger UI documentation."""
    return RedirectResponse(url="/docs")


from app.api.routes import router as api_router
from app.auth.routes import router as auth_router
# from app.ws.routes import router as ws_router

app.include_router(api_router)
app.include_router(auth_router)
# app.include_router(ws_router)
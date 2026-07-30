from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import settings
from app.core.database import dispose_engine, engine
from app.core.logging import configure_logging, get_logger
from app.routes import (
    ai_routes,
    complaint_routes,
    attachment_routes,
    health_routes,
)

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Handle application startup and shutdown.

    Startup:
    - Verify PostgreSQL connection

    Shutdown:
    - Dispose the SQLAlchemy engine
    """

    try:
        with engine.connect() as connection:
            database_name = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()

        logger.info(
            "Database connection successful. Connected to: %s",
            database_name,
        )

    except Exception:
        logger.exception("Database connection failed during startup.")
        raise

    yield

    logger.info("Application shutdown started.")
    dispose_engine()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "AI-powered customer complaint management system "
        "for pharmaceutical manufacturing."
    ),
    debug=settings.debug,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router( health_routes.router, prefix="/api", tags=["Health"],)
app.include_router(ai_routes.router, prefix="/api",)
app.include_router( complaint_routes.router, prefix="/api",)
app.include_router( attachment_routes.router, prefix="/api",)


@app.get("/")
def read_root() -> dict[str, str]:
    """
    Return basic API information.
    """

    return {
        "message": "Welcome to the Pharma Complaint QMS API",
        "status": "running",
        "documentation": "/docs",
    }
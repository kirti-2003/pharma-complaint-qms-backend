from fastapi import APIRouter
from sqlalchemy import text

from app.dependencies.database import DatabaseSession


router = APIRouter()


@router.get("/health")
def health_check(db: DatabaseSession) -> dict[str, str]:
    """
    Check whether the API and PostgreSQL database are available.
    """

    db.execute(text("SELECT 1"))

    return {
        "status": "healthy",
        "api": "running",
        "database": "connected",
    }
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings
from app.core.logging import get_logger


logger = get_logger(__name__)


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    Every database model in the application will inherit from this class.
    """

    pass


engine: Engine = create_engine(
    settings.database_url.strip(),
    echo=settings.debug,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    Provide one database session for each FastAPI request.

    The session is automatically closed after the request finishes.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def dispose_engine() -> None:
    """
    Close all database connections held by the SQLAlchemy engine.

    This function will be called when the FastAPI application shuts down.
    """

    engine.dispose()
    logger.info("Database engine disposed successfully.")
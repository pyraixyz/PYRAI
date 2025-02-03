from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from pyrai.api.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    echo=settings.DB_ECHO,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db() -> None:
    """Initialize database."""
    # Import all models here to ensure they are registered
    from pyrai.models import user, model, task  # noqa: F401
    
    Base.metadata.create_all(bind=engine) 
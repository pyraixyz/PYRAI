from typing import Annotated, Generator
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from redis import Redis

from pyrai.api.auth import get_current_user
from pyrai.api.config import settings
from pyrai.db.session import SessionLocal
from pyrai.models.user import User
from pyrai.core.cache import get_redis_client

def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis() -> Generator[Redis, None, None]:
    """Get Redis client."""
    redis = get_redis_client()
    try:
        yield redis
    finally:
        redis.close()

def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user

def get_current_superuser(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

# Common dependencies
DB = Annotated[Session, Depends(get_db)]
Redis = Annotated[Redis, Depends(get_redis)]
CurrentUser = Annotated[User, Depends(get_current_active_user)]
Superuser = Annotated[User, Depends(get_current_superuser)] 
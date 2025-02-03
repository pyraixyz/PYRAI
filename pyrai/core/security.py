import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from passlib.context import CryptContext
from jose import JWTError, jwt

from pyrai.api.config import settings
from pyrai.core.logging import logger

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_secret_key(length: int = 32) -> str:
    """Generate a secure secret key."""
    return secrets.token_urlsafe(length)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(datetime.UTC) + expires_delta
    else:
        expire = datetime.now(datetime.UTC) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(
            claims=to_encode,
            key=settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        raise

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.error(f"Token verification error: {e}")
        return None

def generate_api_key() -> str:
    """Generate API key."""
    return f"pk_{secrets.token_urlsafe(32)}"

def hash_api_key(api_key: str) -> str:
    """Hash API key for storage."""
    return pwd_context.hash(api_key)

def verify_api_key(plain_api_key: str, hashed_api_key: str) -> bool:
    """Verify API key against its hash."""
    try:
        return pwd_context.verify(plain_api_key, hashed_api_key)
    except Exception as e:
        logger.error(f"API key verification error: {e}")
        return False 
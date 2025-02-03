from typing import Any
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from pyrai.core.config import config
from pyrai.core.security import create_access_token, verify_password
from pyrai.models.user import User
from pyrai.api.schemas.auth import Token, TokenData
from pyrai.api.auth import get_current_user
from pyrai.db.session import AsyncSession, get_session

router = APIRouter()

@router.post("/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session)
) -> Any:
    """Login user and return access token."""
    # Authenticate user
    user = await session.scalar(
        select(User).where(User.email == form_data.username)
    )
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=config.security.access_token_expire_minutes)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user)
) -> Any:
    """Refresh access token."""
    access_token = create_access_token(
        data={"sub": str(current_user.id)},
        expires_delta=timedelta(minutes=config.security.access_token_expire_minutes)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    } 
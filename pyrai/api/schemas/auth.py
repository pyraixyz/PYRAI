from typing import Optional
from pydantic import BaseModel, EmailStr

class Token(BaseModel):
    """Token schema."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data schema."""
    user_id: Optional[int] = None

class UserAuth(BaseModel):
    """User authentication schema."""
    email: EmailStr
    password: str

class UserCreate(UserAuth):
    """User creation schema."""
    full_name: str
    is_active: bool = True
    is_superuser: bool = False

class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None 
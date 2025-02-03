from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

class User(BaseModel):
    """API user details."""
    username: str
    permissions: List[str]
    disabled: bool = False

class Token(BaseModel):
    """Authentication token."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    permissions: List[str] = []

class AuthManager:
    """Manages API authentication."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30
    ):
        """Initialize auth manager.
        
        Args:
            secret_key: JWT secret key
            algorithm: JWT algorithm
            access_token_expire_minutes: Token expiration time
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.users: Dict[str, User] = {}
        
    def create_user(
        self,
        username: str,
        password: str,
        permissions: List[str]
    ):
        """Create new user."""
        hashed_password = self.pwd_context.hash(password)
        self.users[username] = {
            "username": username,
            "hashed_password": hashed_password,
            "permissions": permissions,
            "disabled": False
        }
        
    def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[User]:
        """Authenticate user credentials."""
        user = self.users.get(username)
        if not user:
            return None
            
        if not self.pwd_context.verify(password, user["hashed_password"]):
            return None
            
        if user["disabled"]:
            return None
            
        return User(
            username=username,
            permissions=user["permissions"]
        )
        
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )
            
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
    async def get_current_user(
        self,
        token: str = Depends(oauth2_scheme)
    ) -> User:
        """Get current authenticated user."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
                
            token_data = TokenData(
                username=username,
                permissions=payload.get("permissions", [])
            )
        except JWTError:
            raise credentials_exception
            
        user = self.users.get(token_data.username)
        if user is None:
            raise credentials_exception
            
        return User(**user)
        
    def check_permission(
        self,
        user: User,
        required_permission: str
    ) -> bool:
        """Check if user has required permission."""
        return required_permission in user.permissions 
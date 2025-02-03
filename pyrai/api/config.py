from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class APISettings(BaseSettings):
    # API Configuration
    API_TITLE: str = "PyRai API"
    API_DESCRIPTION: str = "Python library for AI development"
    API_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False
    
    # Redis
    REDIS_URL: str
    REDIS_DB: int = 0
    REDIS_PREFIX: str = "pyrai:"
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    
    # External Services
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # Feature Flags
    ENABLE_DOCS: bool = True
    ENABLE_REDOC: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_PROFILER: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

settings = APISettings() 
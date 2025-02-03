from typing import Any, Dict, Optional, Union
from pathlib import Path
import os
import yaml

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from pyrai.core.constants import DEFAULT_CONFIG

class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(..., description="Database URL")
    pool_size: int = Field(5, description="Connection pool size")
    max_overflow: int = Field(10, description="Maximum overflow connections")
    echo: bool = Field(False, description="Enable SQL echo")

class RedisConfig(BaseModel):
    """Redis configuration."""
    url: str = Field(..., description="Redis URL")
    db: int = Field(0, description="Redis database number")
    prefix: str = Field("pyrai:", description="Key prefix")
    socket_timeout: int = Field(5, description="Socket timeout")
    socket_connect_timeout: int = Field(5, description="Socket connect timeout")

class SecurityConfig(BaseModel):
    """Security configuration."""
    secret_key: str = Field(..., description="Secret key for JWT")
    algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(30, description="Token expiration time")
    password_min_length: int = Field(8, description="Minimum password length")
    max_login_attempts: int = Field(5, description="Maximum login attempts")
    lockout_time: int = Field(300, description="Account lockout time in seconds")

class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    prometheus_enabled: bool = Field(True, description="Enable Prometheus metrics")
    sentry_dsn: Optional[str] = Field(None, description="Sentry DSN")
    log_level: str = Field("INFO", description="Logging level")
    metrics_interval: int = Field(10, description="Metrics collection interval")

class Config(BaseSettings):
    """Application configuration."""
    # Environment
    env: str = Field("development", description="Environment name")
    debug: bool = Field(False, description="Debug mode")
    testing: bool = Field(False, description="Testing mode")

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))

    # Components
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    monitoring: MonitoringConfig

    # API
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    workers: int = Field(4, description="Number of workers")
    cors_origins: list[str] = Field(["*"], description="CORS origins")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
    )

    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_data = yaml.safe_load(f)

        # Merge with default config
        config_data = {**DEFAULT_CONFIG, **config_data}
        
        # Convert paths to absolute
        if "data_dir" in config_data:
            config_data["data_dir"] = str(Path(config_data["data_dir"]).absolute())
        if "logs_dir" in config_data:
            config_data["logs_dir"] = str(Path(config_data["logs_dir"]).absolute())

        return cls(**config_data)

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_data = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

def load_config() -> Config:
    """Load application configuration."""
    config_path = os.getenv("PYRAI_CONFIG")
    if config_path:
        return Config.load_from_file(config_path)
    return Config()

config = load_config() 
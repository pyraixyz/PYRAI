"""
Core configuration management for PYRAI.
Handles all configuration settings for the core engine components.
"""

from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field


class CoreConfig(BaseSettings):
    """
    Core configuration settings for PYRAI engine.
    Handles all the basic settings needed for the core components to function.
    """
    
    # Model Manager Configuration
    model_storage_path: str = Field(
        default="./models",
        description="Base path for model storage"
    )
    max_model_size: int = Field(
        default=1024 * 1024 * 1024,  # 1GB
        description="Maximum model size in bytes"
    )
    
    # Resource Scheduler Configuration
    min_memory_required: int = Field(
        default=1024 * 1024 * 1024,  # 1GB
        description="Minimum memory required for operation"
    )
    max_concurrent_tasks: int = Field(
        default=4,
        description="Maximum number of concurrent tasks"
    )
    
    # Training Coordinator Configuration
    default_batch_size: int = Field(
        default=32,
        description="Default batch size for training"
    )
    checkpoint_interval: int = Field(
        default=1000,
        description="Steps between checkpoints"
    )
    
    # Storage Manager Configuration
    storage_backend: str = Field(
        default="local",
        description="Storage backend type (local, s3, ipfs)"
    )
    cache_size: int = Field(
        default=1024 * 1024 * 100,  # 100MB
        description="Cache size in bytes"
    )

    class Config:
        """Pydantic config class"""
        env_prefix = "PYRAI_"
        case_sensitive = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return self.dict()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CoreConfig":
        """Create config from dictionary"""
        return cls(**config_dict)

    def update(self, **kwargs: Any) -> None:
        """Update config values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Global config instance
config = CoreConfig() 
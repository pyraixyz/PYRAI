from typing import Any, Dict, Optional
import os
import yaml
from copy import deepcopy
from dataclasses import dataclass

class Config:
    """Configuration management class."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = self._validate_config(deepcopy(config_dict))
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration.
        
        Args:
            updates: Configuration updates
        """
        validated = self._validate_config(updates)
        self._config = self._deep_update(self._config, validated)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return deepcopy(self._config)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration instance
        """
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Save path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self._config, f)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration values."""
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        # Validate training config
        if "training" in config:
            train_config = config["training"]
            if "batch_size" in train_config:
                train_config["batch_size"] = int(train_config["batch_size"])
            if "epochs" in train_config:
                train_config["epochs"] = int(train_config["epochs"])
                
        # Validate model config
        if "model" in config:
            model_config = config["model"]
            if "device" in model_config:
                device = model_config["device"]
                if device not in ["cuda", "cpu"]:
                    raise ValueError("Device must be 'cuda' or 'cpu'")
                    
        # Validate deployment config
        if "deployment" in config:
            deploy_config = config["deployment"]
            if "resources" in deploy_config:
                resources = deploy_config["resources"]
                if "cpu" in resources:
                    self._validate_resource_value(resources["cpu"])
                if "memory" in resources:
                    self._validate_resource_value(resources["memory"])
                    
        return config
    
    def _validate_resource_value(self, value: str) -> None:
        """Validate resource value format."""
        if not isinstance(value, str):
            raise ValueError("Resource value must be a string")
            
        # CPU format: "1" or "500m"
        if value.endswith('m'):
            try:
                int(value[:-1])
            except ValueError:
                raise ValueError(f"Invalid CPU value: {value}")
        else:
            try:
                int(value)
            except ValueError:
                raise ValueError(f"Invalid CPU value: {value}")
                
    def _deep_update(
        self,
        base: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep update dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base:
                base[key] = self._deep_update(base[key], value)
            else:
                base[key] = value
        return base 

@dataclass
class EngineConfig:
    """Engine configuration."""
    backend: str = "pytorch"
    device: str = "cuda"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EngineConfig":
        return cls(**config)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device
        }
        
    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config) 
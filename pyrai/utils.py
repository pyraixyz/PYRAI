import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    if path.suffix == '.yml' or path.suffix == '.yaml':
        with open(path) as f:
            return yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

def save_config(
    config: Dict[str, Any],
    path: Union[str, Path],
    format: str = 'yaml'
) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Save path
        format: File format ('yaml' or 'json')
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    
    if format == 'yaml':
        with open(path, 'w') as f:
            yaml.safe_dump(config, f, indent=2)
    elif format == 'json':
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

def get_device_info() -> Dict[str, Any]:
    """Get system device information.
    
    Returns:
        Device information dictionary
    """
    import torch
    import psutil
    
    info = {
        "cpu": {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent()
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        }
    }
    
    if torch.cuda.is_available():
        info["cuda"]["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_cached": torch.cuda.memory_reserved(i)
            }
            for i in range(torch.cuda.device_count())
        ]
        
    return info

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f}PB"

def parse_resource_value(value: str) -> int:
    """Parse resource value string.
    
    Args:
        value: Resource value string (e.g. "1Gi", "500Mi")
        
    Returns:
        Value in bytes
    """
    units = {
        'k': 1000,
        'm': 1000**2,
        'g': 1000**3,
        't': 1000**4,
        'ki': 1024,
        'mi': 1024**2,
        'gi': 1024**3,
        'ti': 1024**4
    }
    
    value = value.lower()
    for unit, multiplier in units.items():
        if value.endswith(unit):
            try:
                number = float(value[:-len(unit)])
                return int(number * multiplier)
            except ValueError:
                raise ValueError(f"Invalid resource value: {value}")
                
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Invalid resource value: {value}")

def setup_seed(seed: int) -> None:
    """Setup random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
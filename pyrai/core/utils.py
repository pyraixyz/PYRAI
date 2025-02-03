import os
import json
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime
import yaml

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext == ".json":
            return json.load(f)
        elif ext in {".yml", ".yaml"}:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    ext = os.path.splitext(path)[1].lower()
    with open(path, "w") as f:
        if ext == ".json":
            json.dump(config, f, indent=2)
        elif ext in {".yml", ".yaml"}:
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

def calculate_file_hash(path: str, algorithm: str = "sha256") -> str:
    """Calculate file hash."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    hash_obj = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def format_size(size: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f}h"
    days = hours / 24
    return f"{days:.2f}d"

def parse_datetime(value: str) -> datetime:
    """Parse datetime string."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse datetime: {value}") 
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json
import shutil
import hashlib
from pathlib import Path

class ModelVersion:
    """Model version metadata."""
    
    def __init__(self,
                 version_id: str,
                 model_id: str,
                 description: str,
                 created_at: datetime,
                 metrics: Dict[str, float],
                 parameters: Dict[str, Any],
                 tags: List[str] = None):
        self.version_id = version_id
        self.model_id = model_id
        self.description = description
        self.created_at = created_at
        self.metrics = metrics
        self.parameters = parameters
        self.tags = tags or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert version metadata to dictionary."""
        return {
            'version_id': self.version_id,
            'model_id': self.model_id,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'metrics': self.metrics,
            'parameters': self.parameters,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create version from dictionary."""
        return cls(
            version_id=data['version_id'],
            model_id=data['model_id'],
            description=data['description'],
            created_at=datetime.fromisoformat(data['created_at']),
            metrics=data['metrics'],
            parameters=data['parameters'],
            tags=data['tags']
        )

class VersionManager:
    """Manages model versions."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def save_version(self,
                    model: Any,
                    model_id: str,
                    description: str,
                    metrics: Dict[str, float],
                    parameters: Dict[str, Any],
                    tags: List[str] = None) -> ModelVersion:
        """Save a new model version."""
        # Generate version ID
        timestamp = datetime.utcnow()
        version_id = f"{model_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create version metadata
        version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            description=description,
            created_at=timestamp,
            metrics=metrics,
            parameters=parameters,
            tags=tags
        )
        
        # Create version directory
        version_dir = self.storage_path / model_id / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Save model artifacts
        self._save_model(model, version_dir / 'model')
        
        return version
    
    def load_version(self, model_id: str, version_id: str) -> tuple[Any, ModelVersion]:
        """Load a model version."""
        version_dir = self.storage_path / model_id / version_id
        
        # Load metadata
        with open(version_dir / 'metadata.json', 'r') as f:
            version = ModelVersion.from_dict(json.load(f))
        
        # Load model artifacts
        model = self._load_model(version_dir / 'model')
        
        return model, version
    
    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model."""
        model_dir = self.storage_path / model_id
        if not model_dir.exists():
            return []
            
        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir():
                with open(version_dir / 'metadata.json', 'r') as f:
                    versions.append(ModelVersion.from_dict(json.load(f)))
                    
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def delete_version(self, model_id: str, version_id: str) -> None:
        """Delete a model version."""
        version_dir = self.storage_path / model_id / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)
    
    def _save_model(self, model: Any, path: Path) -> None:
        """Save model artifacts."""
        if hasattr(model, 'save'):
            model.save(str(path))
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(model, f)
    
    def _load_model(self, path: Path) -> Any:
        """Load model artifacts."""
        if path.is_dir():
            # Try loading as TensorFlow model
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(str(path))
            except:
                pass
                
            # Try loading as PyTorch model
            try:
                import torch
                return torch.load(str(path))
            except:
                pass
        else:
            # Try loading as pickle
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f) 
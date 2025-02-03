"""Model state management module."""

from typing import Dict, Any, Optional
import json
import os
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger()

class ModelStatus(Enum):
    """Model status enumeration."""
    CREATED = "created"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"

class ModelManager:
    """Model state management."""
    
    def __init__(self, storage_path: str = "models"):
        self.storage_path = storage_path
        self.models: Dict[str, Dict[str, Any]] = {}
        os.makedirs(storage_path, exist_ok=True)
        
    def register_model(
        self,
        model_id: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register new model."""
        model_info = {
            "id": model_id,
            "type": model_type,
            "config": config,
            "status": ModelStatus.CREATED.value,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        self.models[model_id] = model_info
        self._save_model_info(model_id, model_info)
        return model_info
        
    def update_model_status(
        self,
        model_id: str,
        status: ModelStatus,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update model status and metadata."""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")
            
        self.models[model_id].update({
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        })
        
        if metadata:
            self.models[model_id]["metadata"] = metadata
            
        self._save_model_info(model_id, self.models[model_id])
        
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information."""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found")
        return self.models[model_id]
        
    def _save_model_info(self, model_id: str, info: Dict[str, Any]):
        """Save model information to storage."""
        file_path = os.path.join(self.storage_path, f"{model_id}.json")
        with open(file_path, "w") as f:
            json.dump(info, f, indent=2)
            
    def load_models(self):
        """Load all model information from storage."""
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                model_id = filename[:-5]
                file_path = os.path.join(self.storage_path, filename)
                with open(file_path, "r") as f:
                    self.models[model_id] = json.load(f)

model_manager = ModelManager() 
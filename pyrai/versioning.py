from typing import Any, Dict, List, Optional
import os
import json
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class Version:
    """Model version information."""
    version_id: str
    model_id: str
    description: str
    created_at: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    path: str

class VersionManager:
    """Model version management system."""
    
    def __init__(self, storage_path: str):
        """Initialize version manager.
        
        Args:
            storage_path: Path for storing model versions
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def save_version(
        self,
        model: Any,
        model_id: str,
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Version:
        """Save model version.
        
        Args:
            model: Model to save
            model_id: Model identifier
            description: Version description
            metrics: Model metrics
            parameters: Model parameters
            
        Returns:
            Version information
        """
        # Create version ID
        version_id = self._generate_version_id()
        
        # Create version directory
        version_path = os.path.join(
            self.storage_path,
            model_id,
            version_id
        )
        os.makedirs(version_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_path, "model.pt")
        self._save_model(model, model_path)
        
        # Create version info
        version = Version(
            version_id=version_id,
            model_id=model_id,
            description=description,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
            parameters=parameters or {},
            path=version_path
        )
        
        # Save version info
        self._save_version_info(version)
        
        return version
    
    def load_version(
        self,
        model_id: str,
        version_id: str
    ) -> Version:
        """Load model version.
        
        Args:
            model_id: Model identifier
            version_id: Version identifier
            
        Returns:
            Version information
        """
        version_path = os.path.join(
            self.storage_path,
            model_id,
            version_id
        )
        
        if not os.path.exists(version_path):
            raise ValueError(f"Version not found: {version_id}")
            
        info_path = os.path.join(version_path, "version.json")
        with open(info_path) as f:
            info = json.load(f)
            
        return Version(**info)
    
    def list_versions(
        self,
        model_id: str
    ) -> List[Version]:
        """List model versions.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of versions
        """
        model_path = os.path.join(self.storage_path, model_id)
        if not os.path.exists(model_path):
            return []
            
        versions = []
        for version_id in os.listdir(model_path):
            try:
                version = self.load_version(model_id, version_id)
                versions.append(version)
            except Exception:
                continue
                
        return sorted(
            versions,
            key=lambda v: v.created_at,
            reverse=True
        )
    
    def delete_version(
        self,
        model_id: str,
        version_id: str
    ) -> None:
        """Delete model version.
        
        Args:
            model_id: Model identifier
            version_id: Version identifier
        """
        version_path = os.path.join(
            self.storage_path,
            model_id,
            version_id
        )
        
        if os.path.exists(version_path):
            shutil.rmtree(version_path)
    
    def _generate_version_id(self) -> str:
        """Generate unique version identifier."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _save_model(self, model: Any, path: str) -> None:
        """Save model to disk."""
        # TODO: Use appropriate backend
        import torch
        torch.save(model.state_dict(), path)
    
    def _save_version_info(self, version: Version) -> None:
        """Save version information."""
        info_path = os.path.join(version.path, "version.json")
        with open(info_path, 'w') as f:
            json.dump(asdict(version), f, indent=2) 
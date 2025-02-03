from typing import Dict, Any, Optional
from pydantic import BaseModel

class DeploymentConfig(BaseModel):
    """Model deployment configuration."""
    
    model_id: str
    version_id: Optional[str]
    name: str
    description: Optional[str]
    resources: Dict[str, Any]
    scaling: Dict[str, Any]
    monitoring: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "model_123",
                "version_id": "20240101_120000",
                "name": "my-model-deployment",
                "description": "Production deployment of my model",
                "resources": {
                    "cpu": "1",
                    "memory": "2Gi",
                    "gpu": "0"
                },
                "scaling": {
                    "min_replicas": 1,
                    "max_replicas": 3,
                    "target_cpu_utilization": 80
                },
                "monitoring": {
                    "enable_metrics": True,
                    "enable_logging": True,
                    "log_level": "INFO"
                }
            }
        } 
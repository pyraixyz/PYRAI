from typing import Dict, Any, Optional, List
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from ..core.base import ModelBackend
from ..core.versioning import VersionManager

class PredictionRequest(BaseModel):
    """Model prediction request."""
    inputs: Dict[str, Any]
    model_id: str
    version_id: Optional[str] = None

class PredictionResponse(BaseModel):
    """Model prediction response."""
    predictions: Dict[str, Any]
    model_id: str
    version_id: str

class ModelServer:
    """Model serving server."""
    
    def __init__(self,
                 version_manager: VersionManager,
                 backend: ModelBackend,
                 host: str = "0.0.0.0",
                 port: int = 8000):
        self.version_manager = version_manager
        self.backend = backend
        self.host = host
        self.port = port
        self.app = FastAPI(title="PYRAI Model Server")
        self.loaded_models: Dict[str, Any] = {}
        
        # Register routes
        self.app.post("/predict", response_model=PredictionResponse)(self.predict)
        self.app.get("/models/{model_id}/versions")(self.list_versions)
        self.app.get("/health")(self.health_check)
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make model predictions."""
        try:
            # Get model version
            version_id = request.version_id
            if not version_id:
                # Use latest version if not specified
                versions = self.version_manager.list_versions(request.model_id)
                if not versions:
                    raise HTTPException(404, "Model not found")
                version_id = versions[0].version_id
            
            # Load model if not already loaded
            model_key = f"{request.model_id}_{version_id}"
            if model_key not in self.loaded_models:
                model, version = self.version_manager.load_version(
                    request.model_id, version_id
                )
                self.loaded_models[model_key] = model
            
            # Convert inputs to appropriate format
            inputs = self._prepare_inputs(request.inputs)
            
            # Make prediction
            predictions = self.backend.predict(
                self.loaded_models[model_key],
                inputs
            )
            
            return PredictionResponse(
                predictions=self._prepare_outputs(predictions),
                model_id=request.model_id,
                version_id=version_id
            )
            
        except Exception as e:
            raise HTTPException(500, str(e))
    
    async def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """List model versions."""
        versions = self.version_manager.list_versions(model_id)
        return [v.to_dict() for v in versions]
    
    async def health_check(self) -> Dict[str, str]:
        """Server health check."""
        return {"status": "healthy"}
    
    def start(self) -> None:
        """Start the server."""
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Any:
        """Prepare inputs for model prediction."""
        # Convert inputs to appropriate format (numpy array, tensor, etc.)
        # Implementation depends on the specific backend
        return np.array(inputs['data'])
    
    def _prepare_outputs(self, outputs: Any) -> Dict[str, Any]:
        """Prepare outputs for response."""
        # Convert outputs to JSON-serializable format
        if isinstance(outputs, np.ndarray):
            return {'predictions': outputs.tolist()}
        return {'predictions': outputs} 
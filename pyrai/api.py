from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from pyrai import Engine
from pyrai.backends import PyTorchBackend

app = FastAPI(title="PyRAI API")

class PredictRequest(BaseModel):
    inputs: Dict[str, Any]
    model_id: str
    version_id: Optional[str] = None

class DeployRequest(BaseModel):
    model_id: str
    version: str
    config: Optional[Dict[str, Any]] = None

class TrainingRequest(BaseModel):
    model_path: str
    data_path: str
    config: Optional[Dict[str, Any]] = None

@app.post("/predict")
async def predict(request: PredictRequest):
    """Make predictions."""
    try:
        engine = Engine()
        predictions = await engine.predict(
            inputs=request.inputs,
            model_id=request.model_id,
            version_id=request.version_id
        )
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deploy")
async def deploy(request: DeployRequest):
    """Deploy model."""
    try:
        engine = Engine()
        deployment = engine.deploy(
            model_id=request.model_id,
            version=request.version,
            config=request.config
        )
        return {
            "status": "success",
            "deployment": {
                "name": deployment.name,
                "endpoint": deployment.endpoint
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train(request: TrainingRequest):
    """Train model."""
    try:
        engine = Engine(backend=PyTorchBackend())
        history = await engine.train(
            model_path=request.model_path,
            data_path=request.data_path,
            config=request.config
        )
        return {
            "status": "success",
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/status")
async def get_status(model_id: str, version: Optional[str] = None):
    """Get model status."""
    try:
        engine = Engine()
        status = engine.get_status(
            model_id=model_id,
            version=version
        )
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/metrics")
async def get_metrics(
    model_id: str,
    version: Optional[str] = None,
    metrics: Optional[List[str]] = None
):
    """Get model metrics."""
    try:
        engine = Engine()
        metrics_data = engine.monitor.get_metrics(
            model_id=model_id,
            version=version,
            metric_names=metrics
        )
        return metrics_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 
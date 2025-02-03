from typing import Any, Dict, List, Optional, Union, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .monitor import RuntimeStats
from .auto_responder import ResponseAction
from .response_evaluator import ResponseEvaluation
from .predictor import PredictionResult
from .learner import ResponseLearner
from .ab_tester import ABTester, TestConfig, TestResult

class StatsUpdate(BaseModel):
    """Performance statistics update."""
    throughput: float
    latency: float
    gpu_memory: List[float]
    gpu_utilization: List[float]
    communication_overhead: float
    data_loading_time: float

class ActionRequest(BaseModel):
    """Response action request."""
    stats: StatsUpdate
    predictions: Dict[str, float]
    alerts: List[str]

class ActionResponse(BaseModel):
    """Response action response."""
    action: str
    params: Dict[str, Any]
    confidence: float
    requires_approval: bool

class TestRequest(BaseModel):
    """A/B test request."""
    baseline_action: str
    variant_action: str
    phase_duration: int
    min_improvement: float
    confidence_level: float

class OptimizationAPI:
    """REST API for performance optimization."""
    
    def __init__(
        self,
        learner: ResponseLearner,
        tester: Optional[ABTester] = None
    ):
        """Initialize API.
        
        Args:
            learner: Response learner
            tester: Optional A/B tester
        """
        self.learner = learner
        self.tester = tester
        self.app = FastAPI()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes."""
        @self.app.post("/action")
        async def get_action(request: ActionRequest) -> ActionResponse:
            """Get recommended action."""
            # Convert request to RuntimeStats
            stats = RuntimeStats(
                throughput=request.stats.throughput,
                latency=request.stats.latency,
                gpu_memory=request.stats.gpu_memory,
                gpu_utilization=request.stats.gpu_utilization,
                communication_overhead=request.stats.communication_overhead,
                data_loading_time=request.stats.data_loading_time
            )
            
            # Get best action
            action = self.learner.get_best_action(stats)
            if not action:
                if self.learner.should_explore():
                    # Random exploration
                    action = np.random.choice(list(ResponseAction))
                else:
                    raise HTTPException(
                        status_code=404,
                        detail="No suitable action found"
                    )
                    
            # Get success probability
            confidence = self.learner.predict_success(stats, action)
            
            return ActionResponse(
                action=action.value,
                params={},  # Add action parameters
                confidence=confidence or 0.0,
                requires_approval=confidence < 0.9 if confidence else True
            )
            
        @self.app.post("/test")
        async def start_test(request: TestRequest) -> Dict[str, Any]:
            """Start A/B test."""
            try:
                baseline = ResponseAction(request.baseline_action)
                variant = ResponseAction(request.variant_action)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid action names"
                )
                
            config = TestConfig(
                baseline_action=baseline,
                variant_action=variant,
                phase_duration=request.phase_duration,
                min_improvement=request.min_improvement,
                confidence_level=request.confidence_level
            )
            
            self.tester = ABTester(config)
            
            return {
                "status": "started",
                "config": request.dict()
            }
            
        @self.app.get("/test/status")
        async def get_test_status() -> Dict[str, Any]:
            """Get A/B test status."""
            if not self.tester:
                raise HTTPException(
                    status_code=404,
                    detail="No active test"
                )
                
            return {
                "phase": self.tester.phase.value,
                "steps": self.tester.steps,
                "current_action": self.tester.get_current_action().value
            }
            
        @self.app.post("/test/update")
        async def update_test(stats: StatsUpdate) -> Optional[Dict[str, Any]]:
            """Update A/B test."""
            if not self.tester:
                raise HTTPException(
                    status_code=404,
                    detail="No active test"
                )
                
            runtime_stats = RuntimeStats(
                throughput=stats.throughput,
                latency=stats.latency,
                gpu_memory=stats.gpu_memory,
                gpu_utilization=stats.gpu_utilization,
                communication_overhead=stats.communication_overhead,
                data_loading_time=stats.data_loading_time
            )
            
            result = self.tester.update(runtime_stats)
            if result:
                return {
                    "status": "completed",
                    "improvement": result.improvement,
                    "confidence": result.confidence,
                    "significant": result.significant,
                    "winner": result.winner.value if result.winner else None
                }
                
            return {"status": "in_progress"}
            
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run API server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port) 
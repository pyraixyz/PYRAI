from typing import Any, Dict, List, Optional, Union, Tuple
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import mlflow
from .distillation import KnowledgeDistillation, DistillationConfig
from .distillation_monitor import DistillationMetrics
from .distillation_tuner import TuningConfig

class ExperimentStatus(Enum):
    """Experiment status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str
    distillation_config: DistillationConfig
    tuning_config: Optional[TuningConfig] = None
    tags: Dict[str, str] = None
    save_dir: str = "experiments"
    tracking_uri: Optional[str] = None

@dataclass
class ExperimentResult:
    """Experiment results."""
    config: ExperimentConfig
    status: ExperimentStatus
    metrics: Dict[str, List[float]]
    best_model_path: Optional[str]
    duration: float
    error: Optional[str]

class ExperimentManager:
    """Manages distillation experiments."""
    
    def __init__(
        self,
        config: ExperimentConfig
    ):
        """Initialize experiment manager.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)
            
    def run_experiment(
        self,
        distillation: KnowledgeDistillation,
        train_data: Any,
        val_data: Optional[Any] = None,
        **kwargs
    ) -> ExperimentResult:
        """Run distillation experiment.
        
        Args:
            distillation: Knowledge distillation instance
            train_data: Training data
            val_data: Optional validation data
            **kwargs: Additional training arguments
            
        Returns:
            Experiment results
        """
        start_time = time.time()
        result = ExperimentResult(
            config=self.config,
            status=ExperimentStatus.RUNNING,
            metrics={},
            best_model_path=None,
            duration=0.0,
            error=None
        )
        
        try:
            # Start MLflow run
            with mlflow.start_run(
                experiment_name=self.config.name,
                tags=self.config.tags
            ) as run:
                # Log configs
                mlflow.log_params({
                    "distillation_config": asdict(self.config.distillation_config),
                    "tuning_config": asdict(self.config.tuning_config)
                    if self.config.tuning_config else None
                })
                
                # Training loop with metric tracking
                for epoch in range(kwargs.get("epochs", 10)):
                    metrics = self._train_epoch(
                        distillation,
                        train_data,
                        val_data,
                        **kwargs
                    )
                    
                    # Update metrics history
                    for name, value in metrics.items():
                        if name not in result.metrics:
                            result.metrics[name] = []
                        result.metrics[name].append(value)
                        
                    # Log metrics
                    mlflow.log_metrics(metrics, step=epoch)
                    
                # Save best model
                model_path = self.save_dir / f"{self.config.name}_best.pt"
                torch.save(distillation.student.state_dict(), model_path)
                mlflow.log_artifact(str(model_path))
                result.best_model_path = str(model_path)
                
                # Save experiment results
                result.status = ExperimentStatus.COMPLETED
                result.duration = time.time() - start_time
                self._save_result(result)
                
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error = str(e)
            result.duration = time.time() - start_time
            self._save_result(result)
            raise
            
        return result
        
    def _train_epoch(
        self,
        distillation: KnowledgeDistillation,
        train_data: Any,
        val_data: Optional[Any],
        **kwargs
    ) -> Dict[str, float]:
        """Train single epoch."""
        # Training
        train_metrics = []
        for batch in train_data:
            metrics = distillation.train_step(
                batch,
                kwargs.get("optimizer")
            )
            train_metrics.append(metrics)
            
        # Calculate average metrics
        avg_metrics = {}
        for key in train_metrics[0].keys():
            avg_metrics[f"train_{key}"] = np.mean([
                m[key] for m in train_metrics
            ])
            
        # Validation
        if val_data:
            val_metrics = distillation.evaluate(val_data)
            avg_metrics.update({
                f"val_{k}": v for k, v in val_metrics.items()
            })
            
        return avg_metrics
        
    def _save_result(self, result: ExperimentResult):
        """Save experiment results."""
        result_path = self.save_dir / f"{self.config.name}_result.json"
        
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
            
    def load_result(self, name: str) -> Optional[ExperimentResult]:
        """Load experiment results."""
        result_path = self.save_dir / f"{name}_result.json"
        
        if not result_path.exists():
            return None
            
        with open(result_path) as f:
            data = json.load(f)
            return ExperimentResult(**data)
            
    def get_all_results(self) -> List[ExperimentResult]:
        """Get all experiment results."""
        results = []
        for path in self.save_dir.glob("*_result.json"):
            with open(path) as f:
                data = json.load(f)
                results.append(ExperimentResult(**data))
        return results
        
    def compare_experiments(
        self,
        names: List[str]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Compare multiple experiments."""
        comparison = {}
        
        for name in names:
            result = self.load_result(name)
            if result and result.status == ExperimentStatus.COMPLETED:
                comparison[name] = result.metrics
                
        return comparison 
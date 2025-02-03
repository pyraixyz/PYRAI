from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import optuna
from .experiment_manager import ExperimentManager, ExperimentConfig, ExperimentStatus
from .distillation import DistillationConfig
from .distillation_tuner import TuningConfig

class SearchSpace(Enum):
    """Hyperparameter search spaces."""
    TEMPERATURE = "temperature"
    ALPHA = "alpha"
    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    LAYERS = "layers"

@dataclass
class SearchConfig:
    """Search configuration."""
    name: str
    spaces: List[SearchSpace]
    n_trials: int = 50
    timeout: Optional[float] = None
    n_jobs: int = 1
    search_ranges: Dict[str, Tuple[float, float]] = None
    pruner_type: str = "median"
    sampler_type: str = "tpe"

class HyperparameterSearch:
    """Manages hyperparameter search."""
    
    def __init__(
        self,
        config: SearchConfig,
        experiment_manager: ExperimentManager,
        objective_metric: str = "val_accuracy_gap"
    ):
        """Initialize search.
        
        Args:
            config: Search configuration
            experiment_manager: Experiment manager
            objective_metric: Metric to optimize
        """
        self.config = config
        self.experiment_manager = experiment_manager
        self.objective_metric = objective_metric
        self.study = None
        self._setup_study()
        
    def _setup_study(self):
        """Setup Optuna study."""
        # Create pruner
        if self.config.pruner_type == "median":
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = optuna.pruners.SuccessiveHalvingPruner()
            
        # Create sampler
        if self.config.sampler_type == "tpe":
            sampler = optuna.samplers.TPESampler()
        else:
            sampler = optuna.samplers.RandomSampler()
            
        self.study = optuna.create_study(
            study_name=self.config.name,
            direction="maximize",
            pruner=pruner,
            sampler=sampler
        )
        
    def search(
        self,
        train_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Run hyperparameter search.
        
        Args:
            train_fn: Training function that returns objective value
            
        Returns:
            Best parameters
        """
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            return train_fn(params)
            
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        return self.study.best_params
        
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters for trial."""
        params = {}
        ranges = self.config.search_ranges or {}
        
        for space in self.config.spaces:
            if space == SearchSpace.TEMPERATURE:
                params["temperature"] = trial.suggest_float(
                    "temperature",
                    *ranges.get("temperature", (0.5, 5.0)),
                    log=True
                )
            elif space == SearchSpace.ALPHA:
                params["alpha"] = trial.suggest_float(
                    "alpha",
                    *ranges.get("alpha", (0.1, 0.9))
                )
            elif space == SearchSpace.LEARNING_RATE:
                params["learning_rate"] = trial.suggest_float(
                    "learning_rate",
                    *ranges.get("learning_rate", (1e-4, 1e-2)),
                    log=True
                )
            elif space == SearchSpace.BATCH_SIZE:
                params["batch_size"] = trial.suggest_int(
                    "batch_size",
                    *ranges.get("batch_size", (16, 256)),
                    log=True
                )
            else:  # LAYERS
                n_layers = trial.suggest_int(
                    "n_layers",
                    *ranges.get("n_layers", (1, 5))
                )
                params["layers"] = [
                    trial.suggest_int(
                        f"layer_{i}",
                        *ranges.get("layer_size", (64, 512)),
                        log=True
                    )
                    for i in range(n_layers)
                ]
                
        return params
        
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history."""
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "duration": trial.duration.total_seconds()
                })
        return history
        
    def plot_optimization_history(self) -> Any:
        """Plot optimization history."""
        return optuna.visualization.plot_optimization_history(self.study)
        
    def plot_param_importances(self) -> Any:
        """Plot parameter importances."""
        return optuna.visualization.plot_param_importances(self.study)
        
    def plot_parallel_coordinate(self) -> Any:
        """Plot parallel coordinate plot."""
        return optuna.visualization.plot_parallel_coordinate(self.study) 
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import tensorflow as tf
import optuna
from dataclasses import dataclass
from .predictor import PerformancePredictor, PerformancePrediction
from .strategy import StrategySelector, ModelProfile
from .parallel import ParallelStrategy

@dataclass
class TuningResult:
    """Auto-tuning results."""
    best_strategy: ParallelStrategy
    best_config: Dict[str, Any]
    predicted_performance: PerformancePrediction
    tuning_history: List[Dict[str, Any]]

class AutoTuner:
    """Automatic parallel strategy tuner."""
    
    def __init__(
        self,
        predictor: PerformancePredictor,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        metric: str = "throughput"
    ):
        """Initialize auto-tuner.
        
        Args:
            predictor: Performance predictor
            n_trials: Number of trials
            timeout: Timeout in seconds
            metric: Optimization metric
        """
        self.predictor = predictor
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        
    def tune(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        batch_size: int,
        input_shape: Union[tuple, List[tuple]],
        constraints: Optional[Dict[str, float]] = None
    ) -> TuningResult:
        """Tune parallel strategy and configuration.
        
        Args:
            model: Model to parallelize
            batch_size: Training batch size
            input_shape: Input shape(s)
            constraints: Optional performance constraints
            
        Returns:
            Tuning results
        """
        # Initialize strategy selector
        selector = StrategySelector(
            num_gpus=self.predictor.num_gpus,
            memory_threshold=0.8,
            communication_threshold=0.5
        )
        
        # Profile model
        model_profile = selector._profile_model(model, batch_size, input_shape)
        
        # Create study
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            # Sample strategy
            strategy = trial.suggest_categorical(
                "strategy",
                [s.value for s in ParallelStrategy]
            )
            strategy = ParallelStrategy(strategy)
            
            # Sample configuration based on strategy
            config = self._sample_config(trial, strategy)
            
            # Predict performance
            prediction = self.predictor.predict_performance(
                model_profile,
                strategy,
                config,
                batch_size
            )
            
            # Check constraints
            if constraints:
                if not self._check_constraints(prediction, constraints):
                    return float('-inf')
                    
            # Return optimization metric
            return getattr(prediction, self.metric)
            
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        # Get best results
        best_strategy = ParallelStrategy(study.best_params["strategy"])
        best_config = self._get_config(study.best_params, best_strategy)
        best_prediction = self.predictor.predict_performance(
            model_profile,
            best_strategy,
            best_config,
            batch_size
        )
        
        return TuningResult(
            best_strategy=best_strategy,
            best_config=best_config,
            predicted_performance=best_prediction,
            tuning_history=study.trials_dataframe().to_dict('records')
        )
        
    def _sample_config(
        self,
        trial: optuna.Trial,
        strategy: ParallelStrategy
    ) -> Dict[str, Any]:
        """Sample strategy configuration."""
        if strategy == ParallelStrategy.PIPELINE:
            return {
                "num_stages": trial.suggest_int(
                    "num_stages",
                    2,
                    self.predictor.num_gpus
                ),
                "chunk_size": trial.suggest_int(
                    "chunk_size",
                    1,
                    32
                )
            }
        elif strategy == ParallelStrategy.TENSOR:
            return {
                "split_size": trial.suggest_int(
                    "split_size",
                    128,
                    1024,
                    step=128
                )
            }
        else:  # Hybrid
            return {
                "pipeline_stages": trial.suggest_int(
                    "pipeline_stages",
                    2,
                    self.predictor.num_gpus // 2
                ),
                "tensor_splits": trial.suggest_int(
                    "tensor_splits",
                    2,
                    4
                ),
                "chunk_size": trial.suggest_int(
                    "chunk_size",
                    1,
                    32
                )
            }
            
    def _check_constraints(
        self,
        prediction: PerformancePrediction,
        constraints: Dict[str, float]
    ) -> bool:
        """Check if prediction satisfies constraints."""
        for metric, limit in constraints.items():
            if getattr(prediction, metric) > limit:
                return False
        return True 
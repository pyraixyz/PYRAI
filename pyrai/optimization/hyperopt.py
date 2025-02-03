from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import torch
import tensorflow as tf
import numpy as np
import optuna
from dataclasses import dataclass
from .monitor import RuntimeStats
from .predictor import MLPredictor

@dataclass
class OptimizationResult:
    """Hyperparameter optimization result."""
    best_params: Dict[str, Any]
    best_value: float
    history: List[Dict[str, Any]]
    importance: Dict[str, float]

class HyperParameterOptimizer:
    """Bayesian hyperparameter optimizer."""
    
    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ):
        """Initialize optimizer.
        
        Args:
            objective_fn: Objective function to minimize
            n_trials: Number of trials
            timeout: Optimization timeout in seconds
            n_jobs: Number of parallel jobs
        """
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        
    def optimize(
        self,
        param_space: Dict[str, Any],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> OptimizationResult:
        """Run hyperparameter optimization.
        
        Args:
            param_space: Parameter search space
            constraints: Optional parameter constraints
            
        Returns:
            Optimization results
        """
        def objective(trial):
            # Sample parameters
            params = {}
            for name, space in param_space.items():
                if isinstance(space, tuple):
                    if isinstance(space[0], int):
                        params[name] = trial.suggest_int(name, space[0], space[1])
                    else:
                        if space[0] > 0 and space[1] > 0:
                            params[name] = trial.suggest_loguniform(name, space[0], space[1])
                        else:
                            params[name] = trial.suggest_uniform(name, space[0], space[1])
                elif isinstance(space, list):
                    params[name] = trial.suggest_categorical(name, space)
                    
            # Check constraints
            if constraints:
                for param, (min_val, max_val) in constraints.items():
                    if params[param] < min_val or params[param] > max_val:
                        return float('inf')
                        
            return self.objective_fn(params)
            
        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler()
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs
        )
        
        # Get results
        history = study.trials_dataframe().to_dict('records')
        importance = optuna.importance.get_param_importances(study)
        
        return OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            history=history,
            importance=importance
        )
        
    @classmethod
    def create_performance_optimizer(
        cls,
        monitor: 'EnhancedPerformanceMonitor',
        target_metric: str = "throughput",
        **kwargs
    ) -> 'HyperParameterOptimizer':
        """Create optimizer for performance tuning."""
        predictor = MLPredictor(target_metric)
        
        def objective(params):
            # Apply parameters
            monitor._apply_parameters(params)
            
            # Collect performance metrics
            stats = monitor.get_stats()
            prediction = predictor.update(stats)
            
            if prediction:
                return -prediction.predicted_value  # Negative for minimization
            return float('inf')
            
        return cls(objective, **kwargs) 
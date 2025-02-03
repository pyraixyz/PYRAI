from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass
import optuna

@dataclass
class OptimizationSpace:
    """Optimization hyperparameter space."""
    name: str
    type: str
    range: List[Any]
    
class AutoOptimizer:
    """Automated optimization search."""
    
    def __init__(
        self,
        model: Any,
        eval_fn: Callable,
        search_space: List[OptimizationSpace],
        n_trials: int = 100,
        timeout: Optional[int] = None
    ):
        """Initialize auto optimizer.
        
        Args:
            model: Model to optimize
            eval_fn: Evaluation function
            search_space: Search space definition
            n_trials: Number of trials
            timeout: Search timeout in seconds
        """
        self.model = model
        self.eval_fn = eval_fn
        self.search_space = search_space
        self.n_trials = n_trials
        self.timeout = timeout
        
    def optimize(self) -> Dict[str, Any]:
        """Run optimization search.
        
        Returns:
            Best configuration found
        """
        study = optuna.create_study(direction="maximize")
        
        def objective(trial):
            config = {}
            for param in self.search_space:
                if param.type == "categorical":
                    config[param.name] = trial.suggest_categorical(
                        param.name,
                        param.range
                    )
                elif param.type == "int":
                    config[param.name] = trial.suggest_int(
                        param.name,
                        param.range[0],
                        param.range[1]
                    )
                elif param.type == "float":
                    config[param.name] = trial.suggest_float(
                        param.name,
                        param.range[0],
                        param.range[1]
                    )
            return self.eval_fn(self.model, config)
            
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        return study.best_params 
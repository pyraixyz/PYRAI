from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import optuna
from .hyperparameter_search import SearchConfig, SearchSpace
from .experiment_manager import ExperimentManager

class SearchAlgorithm(Enum):
    """Advanced search algorithms."""
    BOHB = "bohb"  # Bayesian Optimization and HyperBand
    PBT = "pbt"    # Population Based Training
    ASHA = "asha"  # Asynchronous Successive Halving
    GENETIC = "genetic"

@dataclass
class AdvancedSearchConfig(SearchConfig):
    """Advanced search configuration."""
    algorithm: SearchAlgorithm
    population_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    survival_rate: float = 0.5
    resource_attr: str = "epoch"
    min_resource: int = 1
    max_resource: int = 10

class AdvancedSearch:
    """Advanced hyperparameter search."""
    
    def __init__(
        self,
        config: AdvancedSearchConfig,
        experiment_manager: ExperimentManager,
        objective_metric: str = "val_accuracy_gap"
    ):
        """Initialize advanced search.
        
        Args:
            config: Search configuration
            experiment_manager: Experiment manager
            objective_metric: Metric to optimize
        """
        self.config = config
        self.experiment_manager = experiment_manager
        self.objective_metric = objective_metric
        self.population = []
        self.history = []
        
    def search(
        self,
        train_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Run advanced search.
        
        Args:
            train_fn: Training function
            
        Returns:
            Best parameters
        """
        if self.config.algorithm == SearchAlgorithm.BOHB:
            return self._run_bohb(train_fn)
        elif self.config.algorithm == SearchAlgorithm.PBT:
            return self._run_pbt(train_fn)
        elif self.config.algorithm == SearchAlgorithm.ASHA:
            return self._run_asha(train_fn)
        else:  # GENETIC
            return self._run_genetic(train_fn)
            
    def _run_bohb(
        self,
        train_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Run BOHB algorithm."""
        # Create BOHB study
        study = optuna.create_study(
            study_name=self.config.name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=self.config.min_resource,
                max_resource=self.config.max_resource,
                reduction_factor=3
            )
        )
        
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            return train_fn(params)
            
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        return study.best_params
        
    def _run_pbt(
        self,
        train_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Run Population Based Training."""
        # Initialize population
        self.population = [
            self._generate_random_params()
            for _ in range(self.config.population_size)
        ]
        
        best_params = None
        best_score = float('-inf')
        
        for step in range(self.config.n_trials):
            # Evaluate population
            scores = []
            for params in self.population:
                score = train_fn(params)
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            # Sort population by score
            sorted_indices = np.argsort(scores)[::-1]
            self.population = [
                self.population[i] for i in sorted_indices
            ]
            
            # Exploit and explore
            cutoff = int(len(self.population) * self.config.survival_rate)
            for i in range(cutoff, len(self.population)):
                # Copy parameters from better performing member
                source_idx = np.random.randint(cutoff)
                self.population[i] = self.population[source_idx].copy()
                
                # Perturb parameters
                self._perturb_params(self.population[i])
                
        return best_params
        
    def _run_asha(
        self,
        train_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Run Asynchronous Successive Halving."""
        study = optuna.create_study(
            study_name=self.config.name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.SuccessiveHalvingPruner(
                min_resource=self.config.min_resource,
                reduction_factor=4,
                min_early_stopping_rate=0
            )
        )
        
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            return train_fn(params)
            
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        return study.best_params
        
    def _run_genetic(
        self,
        train_fn: Callable[[Dict[str, Any]], float]
    ) -> Dict[str, Any]:
        """Run Genetic Algorithm."""
        # Initialize population
        self.population = [
            self._generate_random_params()
            for _ in range(self.config.population_size)
        ]
        
        best_params = None
        best_score = float('-inf')
        
        for generation in range(self.config.n_trials):
            # Evaluate population
            scores = []
            for params in self.population:
                score = train_fn(params)
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            # Selection
            parents = self._select_parents(scores)
            
            # Create next generation
            next_population = []
            
            while len(next_population) < len(self.population):
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    parent1, parent2 = np.random.choice(parents, size=2)
                    child = self._crossover(parent1, parent2)
                else:
                    child = np.random.choice(parents).copy()
                    
                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    self._mutate(child)
                    
                next_population.append(child)
                
            self.population = next_population
            
        return best_params
        
    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameters."""
        params = {}
        ranges = self.config.search_ranges or {}
        
        for space in self.config.spaces:
            if space == SearchSpace.TEMPERATURE:
                params["temperature"] = np.random.uniform(
                    *ranges.get("temperature", (0.5, 5.0))
                )
            elif space == SearchSpace.ALPHA:
                params["alpha"] = np.random.uniform(
                    *ranges.get("alpha", (0.1, 0.9))
                )
            # Add other parameter types...
            
        return params
        
    def _perturb_params(self, params: Dict[str, Any]):
        """Perturb parameters for PBT."""
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Perturb by factor of 0.8 or 1.2
                factor = np.random.choice([0.8, 1.2])
                params[key] = value * factor
                
    def _select_parents(
        self,
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Select parents using tournament selection."""
        tournament_size = 3
        parents = []
        
        while len(parents) < self.config.population_size // 2:
            # Select random candidates
            candidates = np.random.choice(
                len(scores),
                size=tournament_size,
                replace=False
            )
            
            # Find winner
            winner_idx = candidates[np.argmax([
                scores[i] for i in candidates
            ])]
            
            parents.append(self.population[winner_idx])
            
        return parents
        
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform crossover between parents."""
        child = {}
        
        for key in parent1:
            # Randomly select from either parent
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
                
        return child
        
    def _mutate(self, params: Dict[str, Any]):
        """Mutate parameters."""
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Add random noise
                noise = np.random.normal(0, 0.1 * value)
                params[key] = value + noise 
from typing import Dict, Any, List, Optional
import random
from datetime import datetime
import numpy as np
from pydantic import BaseModel
from ..core.versioning import VersionManager

class ExperimentConfig(BaseModel):
    """A/B test experiment configuration."""
    
    experiment_id: str
    model_id: str
    variants: List[Dict[str, Any]]
    traffic_split: List[float]
    metrics: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None

class ExperimentResults(BaseModel):
    """A/B test experiment results."""
    
    experiment_id: str
    variant_metrics: Dict[str, Dict[str, float]]
    sample_sizes: Dict[str, int]
    p_values: Dict[str, float]
    winner: Optional[str] = None

class ABTestingManager:
    """Manages A/B testing experiments."""
    
    def __init__(self, version_manager: VersionManager):
        self.version_manager = version_manager
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_experiment(self, config: ExperimentConfig) -> None:
        """Create a new A/B test experiment."""
        if sum(config.traffic_split) != 1.0:
            raise ValueError("Traffic split must sum to 1.0")
            
        if len(config.variants) != len(config.traffic_split):
            raise ValueError("Number of variants must match traffic split")
            
        self.experiments[config.experiment_id] = config
        self.results[config.experiment_id] = []
    
    def get_variant(self, experiment_id: str) -> Dict[str, Any]:
        """Get a variant based on traffic split."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        # Select variant based on traffic split
        variant = random.choices(
            experiment.variants,
            weights=experiment.traffic_split,
            k=1
        )[0]
        
        return variant
    
    def record_result(self,
                     experiment_id: str,
                     variant_id: str,
                     metrics: Dict[str, float]) -> None:
        """Record experiment result."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        self.results[experiment_id].append({
            'variant_id': variant_id,
            'metrics': metrics,
            'timestamp': datetime.utcnow()
        })
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResults:
        """Analyze experiment results."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        results = self.results[experiment_id]
        if not results:
            raise ValueError("No results recorded for experiment")
            
        # Calculate metrics for each variant
        variant_metrics = {}
        sample_sizes = {}
        for variant in experiment.variants:
            variant_id = variant['id']
            variant_results = [
                r for r in results
                if r['variant_id'] == variant_id
            ]
            sample_sizes[variant_id] = len(variant_results)
            
            # Calculate mean metrics
            metrics = {}
            for metric in experiment.metrics:
                values = [r['metrics'][metric] for r in variant_results]
                metrics[metric] = np.mean(values)
            variant_metrics[variant_id] = metrics
        
        # Perform statistical tests
        p_values = self._calculate_p_values(results, experiment.metrics)
        
        # Determine winner
        winner = self._determine_winner(
            variant_metrics,
            p_values,
            experiment.metrics[0]  # Use first metric as primary
        )
        
        return ExperimentResults(
            experiment_id=experiment_id,
            variant_metrics=variant_metrics,
            sample_sizes=sample_sizes,
            p_values=p_values,
            winner=winner
        )
    
    def _calculate_p_values(self,
                          results: List[Dict[str, Any]],
                          metrics: List[str]) -> Dict[str, float]:
        """Calculate p-values for metrics."""
        from scipy import stats
        
        p_values = {}
        for metric in metrics:
            # Group results by variant
            variant_values = {}
            for result in results:
                variant_id = result['variant_id']
                if variant_id not in variant_values:
                    variant_values[variant_id] = []
                variant_values[variant_id].append(result['metrics'][metric])
            
            # Perform t-test between variants
            variants = list(variant_values.keys())
            if len(variants) == 2:
                t_stat, p_value = stats.ttest_ind(
                    variant_values[variants[0]],
                    variant_values[variants[1]]
                )
                p_values[metric] = p_value
                
        return p_values
    
    def _determine_winner(self,
                        variant_metrics: Dict[str, Dict[str, float]],
                        p_values: Dict[str, float],
                        primary_metric: str,
                        significance_level: float = 0.05) -> Optional[str]:
        """Determine winning variant."""
        if primary_metric not in p_values:
            return None
            
        if p_values[primary_metric] > significance_level:
            return None  # No statistically significant winner
            
        # Return variant with best metric
        return max(
            variant_metrics.keys(),
            key=lambda v: variant_metrics[v][primary_metric]
        ) 
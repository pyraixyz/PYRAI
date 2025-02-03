from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from .auto_responder import ResponseAction
from .response_evaluator import ResponseEvaluation, ResponseOutcome
from .monitor import RuntimeStats

@dataclass
class LearningExample:
    """Learning example details."""
    state: Dict[str, float]
    action: ResponseAction
    outcome: ResponseOutcome
    improvement: float

class ResponseLearner:
    """Learns from response outcomes."""
    
    def __init__(
        self,
        min_examples: int = 50,
        success_threshold: float = 0.7,
        exploration_rate: float = 0.1
    ):
        """Initialize response learner.
        
        Args:
            min_examples: Minimum examples for training
            success_threshold: Success rate threshold
            exploration_rate: Random exploration rate
        """
        self.min_examples = min_examples
        self.success_threshold = success_threshold
        self.exploration_rate = exploration_rate
        self.examples: List[LearningExample] = []
        self.model = RandomForestClassifier(n_estimators=100)
        self.trained = False
        
    def add_example(
        self,
        stats: RuntimeStats,
        action: ResponseAction,
        evaluation: ResponseEvaluation
    ):
        """Add learning example.
        
        Args:
            stats: Performance statistics
            action: Response action
            evaluation: Response evaluation
        """
        state = self._extract_state(stats)
        
        example = LearningExample(
            state=state,
            action=action,
            outcome=evaluation.outcome,
            improvement=evaluation.improvement
        )
        
        self.examples.append(example)
        
        # Retrain if enough examples
        if len(self.examples) >= self.min_examples:
            self._train_model()
            
    def predict_success(
        self,
        stats: RuntimeStats,
        action: ResponseAction
    ) -> Optional[float]:
        """Predict success probability.
        
        Args:
            stats: Current statistics
            action: Proposed action
            
        Returns:
            Success probability if model trained
        """
        if not self.trained:
            return None
            
        state = self._extract_state(stats)
        features = self._create_features(state, action)
        
        # Get success probability
        proba = self.model.predict_proba([features])[0]
        return proba[1]  # Probability of success
        
    def should_explore(self) -> bool:
        """Determine if should explore."""
        return np.random.random() < self.exploration_rate
        
    def _extract_state(self, stats: RuntimeStats) -> Dict[str, float]:
        """Extract state features."""
        return {
            "throughput": stats.throughput,
            "latency": stats.latency,
            "gpu_memory": np.mean(stats.gpu_memory),
            "gpu_utilization": np.mean(stats.gpu_utilization),
            "communication_overhead": stats.communication_overhead,
            "data_loading_time": stats.data_loading_time
        }
        
    def _create_features(
        self,
        state: Dict[str, float],
        action: ResponseAction
    ) -> List[float]:
        """Create model features."""
        # Combine state and action
        features = list(state.values())
        features.extend(self._encode_action(action))
        return features
        
    def _encode_action(self, action: ResponseAction) -> List[float]:
        """One-hot encode action."""
        encoding = [0.0] * len(ResponseAction)
        encoding[list(ResponseAction).index(action)] = 1.0
        return encoding
        
    def _train_model(self):
        """Train prediction model."""
        X = []  # Features
        y = []  # Labels (success/failure)
        
        for example in self.examples:
            features = self._create_features(
                example.state,
                example.action
            )
            X.append(features)
            
            # Success if outcome is SUCCESS or PARTIAL
            success = example.outcome in [
                ResponseOutcome.SUCCESS,
                ResponseOutcome.PARTIAL
            ]
            y.append(1 if success else 0)
            
        # Train model
        self.model.fit(X, y)
        self.trained = True
        
    def get_success_rate(self, action: ResponseAction) -> float:
        """Get historical success rate."""
        relevant = [
            ex for ex in self.examples
            if ex.action == action
        ]
        
        if not relevant:
            return 0.0
            
        successes = sum(
            1 for ex in relevant
            if ex.outcome in [ResponseOutcome.SUCCESS, ResponseOutcome.PARTIAL]
        )
        
        return successes / len(relevant)
        
    def get_best_action(
        self,
        stats: RuntimeStats
    ) -> Optional[ResponseAction]:
        """Get best action for current state."""
        if not self.trained:
            return None
            
        best_action = None
        best_prob = -1.0
        
        for action in ResponseAction:
            prob = self.predict_success(stats, action)
            if prob and prob > best_prob:
                best_prob = prob
                best_action = action
                
        if best_prob >= self.success_threshold:
            return best_action
            
        return None 
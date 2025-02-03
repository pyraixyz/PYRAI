from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .auto_responder import ResponseAction
from .response_evaluator import ResponseEvaluation, ResponseOutcome
from .monitor import RuntimeStats

class TestPhase(Enum):
    """A/B test phases."""
    BASELINE = "baseline"
    VARIANT = "variant"
    ANALYSIS = "analysis"

@dataclass
class TestConfig:
    """A/B test configuration."""
    baseline_action: ResponseAction
    variant_action: ResponseAction
    phase_duration: int  # Steps per phase
    min_improvement: float
    confidence_level: float

@dataclass
class TestResult:
    """A/B test results."""
    baseline_metrics: Dict[str, float]
    variant_metrics: Dict[str, float]
    improvement: float
    confidence: float
    significant: bool
    winner: Optional[ResponseAction]

class ABTester:
    """Manages A/B testing of response actions."""
    
    def __init__(
        self,
        config: TestConfig,
        warmup_steps: int = 50
    ):
        """Initialize A/B tester.
        
        Args:
            config: Test configuration
            warmup_steps: Steps to warm up before testing
        """
        self.config = config
        self.warmup_steps = warmup_steps
        self.phase = TestPhase.BASELINE
        self.steps = 0
        self.baseline_stats: List[RuntimeStats] = []
        self.variant_stats: List[RuntimeStats] = []
        
    def update(
        self,
        stats: RuntimeStats
    ) -> Optional[TestResult]:
        """Update test with new statistics.
        
        Args:
            stats: Current statistics
            
        Returns:
            Test results if complete
        """
        self.steps += 1
        
        # Skip warmup period
        if self.steps <= self.warmup_steps:
            return None
            
        # Collect statistics
        if self.phase == TestPhase.BASELINE:
            self.baseline_stats.append(stats)
            if len(self.baseline_stats) >= self.config.phase_duration:
                self.phase = TestPhase.VARIANT
                
        elif self.phase == TestPhase.VARIANT:
            self.variant_stats.append(stats)
            if len(self.variant_stats) >= self.config.phase_duration:
                self.phase = TestPhase.ANALYSIS
                return self._analyze_results()
                
        return None
        
    def get_current_action(self) -> ResponseAction:
        """Get current test action."""
        if self.phase == TestPhase.BASELINE:
            return self.config.baseline_action
        else:
            return self.config.variant_action
            
    def _analyze_results(self) -> TestResult:
        """Analyze test results."""
        # Calculate metrics
        baseline_metrics = self._calculate_metrics(self.baseline_stats)
        variant_metrics = self._calculate_metrics(self.variant_stats)
        
        # Calculate improvement
        improvement = self._calculate_improvement(
            baseline_metrics,
            variant_metrics
        )
        
        # Perform statistical test
        confidence = self._calculate_confidence(
            baseline_metrics,
            variant_metrics
        )
        
        # Determine winner
        significant = confidence >= self.config.confidence_level
        winner = None
        
        if significant and improvement >= self.config.min_improvement:
            winner = self.config.variant_action
        elif significant:
            winner = self.config.baseline_action
            
        return TestResult(
            baseline_metrics=baseline_metrics,
            variant_metrics=variant_metrics,
            improvement=improvement,
            confidence=confidence,
            significant=significant,
            winner=winner
        )
        
    def _calculate_metrics(
        self,
        stats_list: List[RuntimeStats]
    ) -> Dict[str, float]:
        """Calculate average metrics."""
        return {
            "throughput": np.mean([s.throughput for s in stats_list]),
            "latency": np.mean([s.latency for s in stats_list]),
            "gpu_utilization": np.mean([
                np.mean(s.gpu_utilization) for s in stats_list
            ]),
            "memory_usage": np.mean([
                np.mean(s.gpu_memory) for s in stats_list
            ]),
            "communication_overhead": np.mean([
                s.communication_overhead for s in stats_list
            ])
        }
        
    def _calculate_improvement(
        self,
        baseline: Dict[str, float],
        variant: Dict[str, float]
    ) -> float:
        """Calculate overall improvement."""
        improvements = []
        
        # Throughput (higher is better)
        if baseline["throughput"] > 0:
            improvements.append(
                variant["throughput"] / baseline["throughput"] - 1
            )
            
        # Latency (lower is better)
        if variant["latency"] > 0:
            improvements.append(
                baseline["latency"] / variant["latency"] - 1
            )
            
        # GPU utilization (higher is better)
        improvements.append(
            variant["gpu_utilization"] / baseline["gpu_utilization"] - 1
            if baseline["gpu_utilization"] > 0 else 0
        )
        
        # Memory usage (lower is better)
        if variant["memory_usage"] > 0:
            improvements.append(
                baseline["memory_usage"] / variant["memory_usage"] - 1
            )
            
        # Communication overhead (lower is better)
        if variant["communication_overhead"] > 0:
            improvements.append(
                baseline["communication_overhead"] /
                variant["communication_overhead"] - 1
            )
            
        return np.mean(improvements)
        
    def _calculate_confidence(
        self,
        baseline: Dict[str, float],
        variant: Dict[str, float]
    ) -> float:
        """Calculate statistical confidence."""
        # Perform t-test on throughput
        baseline_throughput = [s.throughput for s in self.baseline_stats]
        variant_throughput = [s.throughput for s in self.variant_stats]
        
        t_stat, p_value = scipy.stats.ttest_ind(
            baseline_throughput,
            variant_throughput
        )
        
        return 1 - p_value 
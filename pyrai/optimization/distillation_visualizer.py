from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .distillation_monitor import DistillationMetrics
from .distillation_tuner import DistillationTuner

class DistillationVisualizer:
    """Visualizes distillation process and tuning."""
    
    def __init__(
        self,
        history_window: int = 100
    ):
        """Initialize visualizer.
        
        Args:
            history_window: Window size for visualization
        """
        self.history_window = history_window
        self.metrics_history: List[DistillationMetrics] = []
        self.steps: List[int] = []
        
    def update(
        self,
        metrics: DistillationMetrics,
        step: int
    ):
        """Update visualization data."""
        self.metrics_history.append(metrics)
        self.steps.append(step)
        
        # Trim history
        if len(self.metrics_history) > self.history_window:
            self.metrics_history = self.metrics_history[-self.history_window:]
            self.steps = self.steps[-self.history_window:]
            
    def plot_losses(self) -> go.Figure:
        """Plot loss curves."""
        fig = go.Figure()
        
        # Add loss curves
        fig.add_trace(go.Scatter(
            x=self.steps,
            y=[m.student_loss for m in self.metrics_history],
            name="Student Loss",
            mode="lines"
        ))
        
        fig.add_trace(go.Scatter(
            x=self.steps,
            y=[m.teacher_loss for m in self.metrics_history],
            name="Teacher Loss",
            mode="lines"
        ))
        
        fig.add_trace(go.Scatter(
            x=self.steps,
            y=[m.distillation_loss for m in self.metrics_history],
            name="Distillation Loss",
            mode="lines"
        ))
        
        fig.update_layout(
            title="Training Losses",
            xaxis_title="Step",
            yaxis_title="Loss",
            height=400
        )
        
        return fig
        
    def plot_metrics(self) -> go.Figure:
        """Plot other metrics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Accuracy Gap",
                "Loss Ratio",
                "Temperature",
                "Alpha"
            ]
        )
        
        # Accuracy gap
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=[m.accuracy_gap for m in self.metrics_history],
                name="Accuracy Gap"
            ),
            row=1, col=1
        )
        
        # Loss ratio
        loss_ratios = [
            m.student_loss / m.teacher_loss
            if m.teacher_loss > 0 else 0
            for m in self.metrics_history
        ]
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=loss_ratios,
                name="Loss Ratio"
            ),
            row=1, col=2
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=[m.temperature for m in self.metrics_history],
                name="Temperature"
            ),
            row=2, col=1
        )
        
        # Alpha
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=[m.alpha for m in self.metrics_history],
                name="Alpha"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False
        )
        
        return fig
        
    def plot_tuning(self, tuner: DistillationTuner) -> go.Figure:
        """Plot tuning progress."""
        if not tuner.history:
            return go.Figure()
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Score History",
                "Parameter Space",
                "Temperature vs Score",
                "Alpha vs Score"
            ]
        )
        
        # Score history
        scores = [h["score"] for h in tuner.history]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores))),
                y=scores,
                name="Score"
            ),
            row=1, col=1
        )
        
        # Parameter space
        fig.add_trace(
            go.Scatter(
                x=[h["temperature"] for h in tuner.history],
                y=[h["alpha"] for h in tuner.history],
                mode="markers+text",
                text=[f"{h['score']:.2f}" for h in tuner.history],
                textposition="top center",
                name="Trials"
            ),
            row=1, col=2
        )
        
        # Temperature vs score
        fig.add_trace(
            go.Scatter(
                x=[h["temperature"] for h in tuner.history],
                y=[h["score"] for h in tuner.history],
                mode="markers",
                name="Temperature"
            ),
            row=2, col=1
        )
        
        # Alpha vs score
        fig.add_trace(
            go.Scatter(
                x=[h["alpha"] for h in tuner.history],
                y=[h["score"] for h in tuner.history],
                mode="markers",
                name="Alpha"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False
        )
        
        return fig 
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from .predictor import PredictionResult
from .monitor import RuntimeStats

class PredictionVisualizer:
    """Visualizes performance predictions."""
    
    def __init__(
        self,
        history_window: int = 100,
        prediction_horizon: int = 10
    ):
        """Initialize prediction visualizer.
        
        Args:
            history_window: Window size for visualization
            prediction_horizon: Steps to predict ahead
        """
        self.history_window = history_window
        self.prediction_horizon = prediction_horizon
        self.history: List[Dict[str, float]] = []
        self.prediction_history: List[Dict[str, PredictionResult]] = []
        
    def update(
        self,
        stats: RuntimeStats,
        predictions: Dict[str, PredictionResult]
    ):
        """Update visualization data.
        
        Args:
            stats: Current statistics
            predictions: Current predictions
        """
        # Update actual values
        metrics = {
            "throughput": stats.throughput,
            "latency": stats.latency,
            "memory": np.mean(stats.gpu_memory),
            "compute": np.mean(stats.gpu_utilization)
        }
        self.history.append(metrics)
        if len(self.history) > self.history_window:
            self.history.pop(0)
            
        # Update predictions
        self.prediction_history.append(predictions)
        if len(self.prediction_history) > self.history_window:
            self.prediction_history.pop(0)
            
    def create_prediction_dashboard(self) -> go.Figure:
        """Create interactive prediction dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Throughput Prediction',
                'Latency Prediction',
                'Memory Usage Prediction',
                'Compute Utilization Prediction'
            )
        )
        
        # Add traces for each metric
        self._add_prediction_trace(fig, "throughput", 1, 1)
        self._add_prediction_trace(fig, "latency", 1, 2)
        self._add_prediction_trace(fig, "memory", 2, 1)
        self._add_prediction_trace(fig, "compute", 2, 2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Performance Predictions",
            title_x=0.5
        )
        
        return fig
        
    def _add_prediction_trace(
        self,
        fig: go.Figure,
        metric: str,
        row: int,
        col: int
    ):
        """Add prediction trace to dashboard."""
        # Actual values
        actual = [h[metric] for h in self.history]
        times = list(range(len(actual)))
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=actual,
                name=f'Actual {metric}',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        
        # Predicted values
        predictions = [
            p[metric].value if metric in p else None
            for p in self.prediction_history
        ]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=predictions,
                name=f'Predicted {metric}',
                line=dict(color='red', dash='dash')
            ),
            row=row, col=col
        )
        
        # Confidence intervals
        if predictions:
            confidence = [
                p[metric].confidence if metric in p else None
                for p in self.prediction_history
            ]
            
            upper = [
                p + c * p
                for p, c in zip(predictions, confidence)
                if p is not None and c is not None
            ]
            
            lower = [
                p - c * p
                for p, c in zip(predictions, confidence)
                if p is not None and c is not None
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=upper,
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.1)'),
                    name='Upper bound'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=lower,
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.1)'),
                    name='Lower bound'
                ),
                row=row, col=col
            )
            
        fig.update_xaxes(title_text="Steps", row=row, col=col)
        fig.update_yaxes(title_text=metric.capitalize(), row=row, col=col)
        
    def create_factor_analysis(self, predictions: Dict[str, PredictionResult]) -> go.Figure:
        """Create factor importance visualization."""
        metrics = list(predictions.keys())
        factors = {}
        
        # Collect factor importance for each metric
        for metric, pred in predictions.items():
            for factor, importance in pred.factors.items():
                if factor not in factors:
                    factors[factor] = {}
                factors[factor][metric] = importance
                
        # Create heatmap
        factor_names = list(factors.keys())
        values = np.array([
            [factors[f].get(m, 0) for m in metrics]
            for f in factor_names
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=metrics,
            y=factor_names,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Factor Importance Analysis',
            xaxis_title='Metrics',
            yaxis_title='Factors',
            height=400
        )
        
        return fig
        
    def create_trend_analysis(self) -> go.Figure:
        """Create trend analysis visualization."""
        if not self.prediction_history:
            return None
            
        metrics = list(self.prediction_history[-1].keys())
        trends = {m: [] for m in metrics}
        
        # Collect trends
        for predictions in self.prediction_history:
            for metric in metrics:
                if metric in predictions:
                    trends[metric].append(predictions[metric].trend)
                    
        # Create stacked bar chart
        fig = go.Figure()
        
        for trend in ["increasing", "stable", "decreasing"]:
            counts = [
                sum(1 for t in trends[m] if t == trend)
                for m in metrics
            ]
            
            fig.add_trace(go.Bar(
                name=trend,
                x=metrics,
                y=counts
            ))
            
        fig.update_layout(
            barmode='stack',
            title='Metric Trend Analysis',
            xaxis_title='Metrics',
            yaxis_title='Count',
            height=400
        )
        
        return fig 
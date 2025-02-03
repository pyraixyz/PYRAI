from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass
from .profiler import ProfileResult, ProfilerMetric
from .monitor import RuntimeStats
from .distillation_analyzer import DistillationAnalyzer
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

class InteractiveVisualizer:
    """Interactive performance visualization."""
    
    def __init__(
        self,
        history_window: int = 100,
        update_interval: int = 10,
        analyzer: DistillationAnalyzer,
        port: int = 8050
    ):
        """Initialize interactive visualizer.
        
        Args:
            history_window: Window size for visualization
            update_interval: Plot update interval
            analyzer: Distillation analyzer
            port: Dashboard port
        """
        self.history_window = history_window
        self.update_interval = update_interval
        self.history: List[RuntimeStats] = []
        self.fig = None
        self.analyzer = analyzer
        self.port = port
        self.app = dash.Dash(__name__)
        self.analysis_results = None
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("Distillation Analysis Dashboard"),
            
            # Control panel
            html.Div([
                html.H2("Controls"),
                
                # Analysis type selector
                html.Label("Analysis Type:"),
                dcc.Dropdown(
                    id="analysis-type",
                    options=[
                        {"label": t.value.capitalize(), "value": t.value}
                        for t in self.analyzer.config.types
                    ],
                    value=self.analyzer.config.types[0].value
                ),
                
                # Layer selector
                html.Label("Layer:"),
                dcc.Dropdown(
                    id="layer-selector",
                    options=[],  # Will be populated dynamically
                    value=None
                ),
                
                # Plot type selector
                html.Label("Plot Type:"),
                dcc.Dropdown(
                    id="plot-type",
                    options=[],  # Will be populated dynamically
                    value=None
                )
            ], style={"width": "30%", "float": "left", "padding": "20px"}),
            
            # Main visualization area
            html.Div([
                dcc.Graph(id="main-plot"),
                html.Div(id="plot-description")
            ], style={"width": "70%", "float": "right"}),
            
            # Statistics panel
            html.Div([
                html.H2("Statistics"),
                html.Div(id="statistics-panel")
            ], style={"clear": "both", "padding": "20px"})
        ])
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output("layer-selector", "options"),
             Output("plot-type", "options")],
            [Input("analysis-type", "value")]
        )
        def update_selectors(analysis_type):
            """Update layer and plot selectors."""
            if not self.analysis_results:
                return [], []
                
            result = self.analysis_results.get(analysis_type, {})
            
            # Get available layers
            layers = []
            if "statistics" in result:
                if isinstance(result["statistics"], dict):
                    layers = list(result["statistics"].keys())
                    
            # Get available plots
            plots = []
            if analysis_type == "loss":
                plots = ["trends", "correlations"]
            elif analysis_type == "feature":
                plots = ["similarity", "distribution"]
            elif analysis_type == "attention":
                plots = ["patterns", "similarity"]
            elif analysis_type == "gradient":
                plots = ["flow", "distribution"]
            else:  # representation
                plots = ["space", "clustering"]
                
            return (
                [{"label": l, "value": l} for l in layers],
                [{"label": p.capitalize(), "value": p} for p in plots]
            )
            
        @self.app.callback(
            [Output("main-plot", "figure"),
             Output("plot-description", "children"),
             Output("statistics-panel", "children")],
            [Input("analysis-type", "value"),
             Input("layer-selector", "value"),
             Input("plot-type", "value")]
        )
        def update_visualization(analysis_type, layer, plot_type):
            """Update visualization."""
            if not self.analysis_results or not analysis_type or not plot_type:
                return go.Figure(), "", ""
                
            result = self.analysis_results[analysis_type]
            
            # Get plot
            plot_key = f"{analysis_type}_{plot_type}"
            fig = self.analyzer.plot_analysis(
                {analysis_type: result}
            ).get(plot_key, go.Figure())
            
            # Get description
            description = self._get_plot_description(
                analysis_type,
                plot_type,
                result
            )
            
            # Get statistics
            statistics = self._get_statistics(
                analysis_type,
                layer,
                result
            )
            
            return fig, description, statistics
            
    def _get_plot_description(
        self,
        analysis_type: str,
        plot_type: str,
        result: Dict[str, Any]
    ) -> str:
        """Get plot description."""
        descriptions = {
            "loss_trends": "Training loss trends over time",
            "loss_correlations": "Correlations between different loss components",
            "feature_similarity": "Similarity metrics between student and teacher features",
            "feature_distribution": "Distribution of feature statistics",
            "attention_patterns": "Analysis of attention patterns",
            "attention_similarity": "Similarity between student and teacher attention",
            "gradient_flow": "Gradient flow through network layers",
            "gradient_distribution": "Distribution of gradient statistics",
            "representation_space": "Learned representation space visualization",
            "representation_clustering": "Clustering analysis of representations"
        }
        
        return descriptions.get(f"{analysis_type}_{plot_type}", "")
        
    def _get_statistics(
        self,
        analysis_type: str,
        layer: Optional[str],
        result: Dict[str, Any]
    ) -> str:
        """Get statistics summary."""
        stats = result.get("statistics", {})
        
        if not stats:
            return "No statistics available"
            
        if layer:
            stats = stats.get(layer, {})
            
        # Format statistics
        summary = []
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                summary.append(f"{k}: {v:.4f}")
            elif isinstance(v, dict):
                summary.append(f"{k}:")
                for sk, sv in v.items():
                    summary.append(f"  {sk}: {sv:.4f}")
                    
        return html.Pre("\n".join(summary))
        
    def visualize(self, results: Dict[str, Any]):
        """Run interactive visualization.
        
        Args:
            results: Analysis results
        """
        self.analysis_results = results
        self.app.run_server(port=self.port)
        
    def create_dashboard(self, stats: RuntimeStats, profile_results: List[ProfileResult]):
        """Create interactive performance dashboard.
        
        Args:
            stats: Current runtime statistics
            profile_results: Profiling results
        """
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'GPU Memory Usage', 'GPU Compute Utilization',
                'Communication Overhead', 'Training Throughput',
                'Performance Bottlenecks', 'Resource Distribution'
            )
        )
        
        # Memory usage
        self._add_memory_trace()
        
        # Compute utilization
        self._add_compute_trace()
        
        # Communication overhead
        self._add_communication_trace()
        
        # Throughput
        self._add_throughput_trace()
        
        # Bottlenecks
        self._add_bottleneck_trace(profile_results)
        
        # Resource distribution
        self._add_resource_trace(stats)
        
        # Update layout
        self.fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Performance Dashboard",
            title_x=0.5,
            hovermode='x'
        )
        
        return self.fig
        
    def _add_memory_trace(self):
        """Add memory usage visualization."""
        memory_utils = [
            np.mean(stats.gpu_memory)
            for stats in self.history
        ]
        times = list(range(len(memory_utils)))
        
        self.fig.add_trace(
            go.Scatter(
                x=times,
                y=memory_utils,
                name='GPU Memory',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add warning threshold
        self.fig.add_trace(
            go.Scatter(
                x=times,
                y=[0.9] * len(times),
                name='Warning Threshold',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        self.fig.update_xaxes(title_text="Steps", row=1, col=1)
        self.fig.update_yaxes(title_text="Memory Utilization", row=1, col=1)
        
    def _add_compute_trace(self):
        """Add compute utilization visualization."""
        gpu_utils = [
            np.mean(stats.gpu_utilization)
            for stats in self.history
        ]
        times = list(range(len(gpu_utils)))
        
        self.fig.add_trace(
            go.Scatter(
                x=times,
                y=gpu_utils,
                name='GPU Utilization',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        self.fig.update_xaxes(title_text="Steps", row=1, col=2)
        self.fig.update_yaxes(title_text="Compute Utilization", row=1, col=2)
        
    def _add_communication_trace(self):
        """Add communication overhead visualization."""
        comm_overheads = [
            stats.communication_overhead
            for stats in self.history
        ]
        times = list(range(len(comm_overheads)))
        
        self.fig.add_trace(
            go.Scatter(
                x=times,
                y=comm_overheads,
                name='Communication',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        self.fig.update_xaxes(title_text="Steps", row=2, col=1)
        self.fig.update_yaxes(title_text="Overhead (s)", row=2, col=1)
        
    def _add_throughput_trace(self):
        """Add throughput visualization."""
        throughputs = [
            stats.throughput
            for stats in self.history
        ]
        times = list(range(len(throughputs)))
        
        self.fig.add_trace(
            go.Scatter(
                x=times,
                y=throughputs,
                name='Throughput',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        self.fig.update_xaxes(title_text="Steps", row=2, col=2)
        self.fig.update_yaxes(title_text="Samples/second", row=2, col=2)
        
    def _add_bottleneck_trace(self, profile_results: List[ProfileResult]):
        """Add bottleneck visualization."""
        metrics = [result.metric.value for result in profile_results]
        scores = [result.bottleneck_score for result in profile_results]
        
        self.fig.add_trace(
            go.Bar(
                x=metrics,
                y=scores,
                name='Bottleneck Scores',
                marker_color='red'
            ),
            row=3, col=1
        )
        
        self.fig.update_xaxes(title_text="Metric", row=3, col=1)
        self.fig.update_yaxes(title_text="Bottleneck Score", row=3, col=1)
        
    def _add_resource_trace(self, stats: RuntimeStats):
        """Add resource distribution visualization."""
        resources = ['Memory', 'Compute', 'Network', 'I/O']
        utilizations = [
            np.mean(stats.gpu_memory),
            np.mean(stats.gpu_utilization),
            stats.communication_overhead,
            stats.data_loading_time
        ]
        
        self.fig.add_trace(
            go.Pie(
                labels=resources,
                values=utilizations,
                name='Resource Distribution'
            ),
            row=3, col=2
        )
        
    def create_layer_heatmap(self, layer_stats: Dict[str, Dict[str, float]]):
        """Create interactive layer performance heatmap.
        
        Args:
            layer_stats: Layer-wise statistics
        """
        layers = list(layer_stats.keys())
        metrics = ['latency', 'flops', 'parameters']
        
        # Normalize values for heatmap
        values = []
        for metric in metrics:
            metric_values = [layer_stats[layer][metric] for layer in layers]
            max_val = max(metric_values)
            values.append([v/max_val for v in metric_values])
            
        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=layers,
            y=metrics,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Layer Performance Heatmap',
            xaxis_title='Layers',
            yaxis_title='Metrics',
            height=400
        )
        
        return fig
        
    def create_timeline(self, stats: RuntimeStats):
        """Create interactive timeline visualization.
        
        Args:
            stats: Runtime statistics
        """
        # Create timeline data
        events = []
        times = []
        durations = []
        
        # Add computation events
        events.append('Computation')
        times.append(0)
        durations.append(stats.compute_time)
        
        # Add communication events
        events.append('Communication')
        times.append(stats.compute_time)
        durations.append(stats.communication_overhead)
        
        # Add I/O events
        events.append('I/O')
        times.append(stats.compute_time + stats.communication_overhead)
        durations.append(stats.data_loading_time)
        
        fig = px.timeline(
            dict(
                Task=events,
                Start=times,
                Finish=[t + d for t, d in zip(times, durations)]
            ),
            x_start="Start",
            x_end="Finish",
            y="Task",
            title="Training Timeline"
        )
        
        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Time (s)"
        )
        
        return fig 
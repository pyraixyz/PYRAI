from typing import Any, Dict, List, Optional, Union, Tuple
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
from dataclasses import asdict
from .interactive_controller import InteractiveController, ControlDecision
from .prediction_visualizer import PredictionVisualizer
from .response_evaluator import ResponseEvaluator
from .monitor import RuntimeStats

class WebInterface:
    """Web-based performance control interface."""
    
    def __init__(
        self,
        controller: InteractiveController,
        visualizer: PredictionVisualizer,
        evaluator: ResponseEvaluator,
        update_interval: int = 1000  # ms
    ):
        """Initialize web interface.
        
        Args:
            controller: Interactive controller
            visualizer: Prediction visualizer
            evaluator: Response evaluator
            update_interval: UI update interval
        """
        self.controller = controller
        self.visualizer = visualizer
        self.evaluator = evaluator
        self.update_interval = update_interval
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("Performance Control Dashboard"),
            
            # Control mode selector
            html.Div([
                html.Label("Control Mode:"),
                dcc.Dropdown(
                    id="mode-selector",
                    options=[
                        {"label": mode.value, "value": mode.value}
                        for mode in ControlMode
                    ],
                    value=self.controller.mode.value
                )
            ]),
            
            # Performance metrics
            html.Div([
                html.H2("Performance Metrics"),
                dcc.Graph(id="metrics-graph")
            ]),
            
            # Predictions
            html.Div([
                html.H2("Performance Predictions"),
                dcc.Graph(id="predictions-graph")
            ]),
            
            # Pending decisions
            html.Div([
                html.H2("Pending Decisions"),
                html.Div(id="decisions-list")
            ]),
            
            # Response evaluations
            html.Div([
                html.H2("Response Evaluations"),
                dcc.Graph(id="evaluations-graph")
            ]),
            
            # Update interval
            dcc.Interval(
                id="interval-component",
                interval=self.update_interval,
                n_intervals=0
            )
        ])
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        # Mode selector callback
        @self.app.callback(
            Output("mode-selector", "value"),
            Input("mode-selector", "value")
        )
        def update_mode(value):
            if value:
                self.controller.mode = ControlMode(value)
            return value
            
        # Metrics update callback
        @self.app.callback(
            Output("metrics-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_metrics(_):
            return self._create_metrics_figure()
            
        # Predictions update callback
        @self.app.callback(
            Output("predictions-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_predictions(_):
            return self.visualizer.create_prediction_dashboard()
            
        # Decisions update callback
        @self.app.callback(
            Output("decisions-list", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_decisions(_):
            return self._create_decisions_list()
            
        # Evaluations update callback
        @self.app.callback(
            Output("evaluations-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_evaluations(_):
            return self._create_evaluations_figure()
            
        # Decision approval callback
        @self.app.callback(
            Output("decisions-list", "children"),
            Input({"type": "approve-button", "index": dash.ALL}, "n_clicks"),
            State("decisions-list", "children")
        )
        def handle_approval(clicks, current_list):
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_list
                
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            decision_index = json.loads(button_id)["index"]
            
            decisions = self.controller.get_pending_decisions()
            if 0 <= decision_index < len(decisions):
                self.controller.approve_decision(decisions[decision_index])
                
            return self._create_decisions_list()
            
    def _create_metrics_figure(self) -> go.Figure:
        """Create metrics visualization."""
        if not hasattr(self, "latest_stats"):
            return go.Figure()
            
        stats = self.latest_stats
        
        fig = go.Figure()
        
        # Add metric bars
        metrics = {
            "Throughput": stats.throughput,
            "GPU Utilization": np.mean(stats.gpu_utilization),
            "Memory Usage": np.mean(stats.gpu_memory),
            "Communication": stats.communication_overhead
        }
        
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values())
        ))
        
        fig.update_layout(
            title="Current Performance Metrics",
            yaxis_title="Value",
            height=400
        )
        
        return fig
        
    def _create_decisions_list(self) -> List[html.Div]:
        """Create pending decisions list."""
        decisions = self.controller.get_pending_decisions()
        
        return [
            html.Div([
                html.H3(f"Decision {i+1}"),
                html.P(f"Action: {decision.action.value}"),
                html.P(f"Reason: {decision.reason}"),
                html.P(f"Confidence: {decision.confidence:.2f}"),
                html.Button(
                    "Approve",
                    id={"type": "approve-button", "index": i},
                    n_clicks=0
                )
            ])
            for i, decision in enumerate(decisions)
        ]
        
    def _create_evaluations_figure(self) -> go.Figure:
        """Create evaluations visualization."""
        evaluations = self.evaluator.evaluations
        
        if not evaluations:
            return go.Figure()
            
        fig = go.Figure()
        
        # Add evaluation results
        actions = [eval.action.value for eval in evaluations]
        improvements = [eval.improvement for eval in evaluations]
        colors = [
            "green" if eval.outcome == ResponseOutcome.SUCCESS else
            "yellow" if eval.outcome == ResponseOutcome.PARTIAL else
            "red"
            for eval in evaluations
        ]
        
        fig.add_trace(go.Bar(
            x=actions,
            y=improvements,
            marker_color=colors
        ))
        
        fig.update_layout(
            title="Response Evaluation Results",
            xaxis_title="Action",
            yaxis_title="Improvement",
            height=400
        )
        
        return fig
        
    def run(self, host: str = "localhost", port: int = 8050):
        """Run web interface."""
        self.app.run_server(host=host, port=port) 
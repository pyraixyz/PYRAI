from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from .distillation_tuner import DistillationTuner, TuningConfig, TuningMode
from .distillation_monitor import DistillationMetrics
from .distillation_visualizer import DistillationVisualizer

class InteractiveTuner:
    """Interactive distillation parameter tuner."""
    
    def __init__(
        self,
        tuner: DistillationTuner,
        visualizer: DistillationVisualizer,
        update_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        port: int = 8050
    ):
        """Initialize interactive tuner.
        
        Args:
            tuner: Distillation tuner
            visualizer: Distillation visualizer
            update_callback: Callback for parameter updates
            port: Dashboard port
        """
        self.tuner = tuner
        self.visualizer = visualizer
        self.update_callback = update_callback
        self.port = port
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("Distillation Parameter Tuning"),
            
            # Control panel
            html.Div([
                html.H2("Controls"),
                
                # Mode selector
                html.Label("Tuning Mode:"),
                dcc.Dropdown(
                    id="mode-selector",
                    options=[
                        {"label": mode.value, "value": mode.value}
                        for mode in TuningMode
                    ],
                    value=self.tuner.config.mode.value
                ),
                
                # Manual controls
                html.Div([
                    html.Label("Temperature:"),
                    dcc.Slider(
                        id="temperature-slider",
                        min=self.tuner.config.temperature_range[0],
                        max=self.tuner.config.temperature_range[1],
                        step=0.1,
                        value=2.0
                    ),
                    
                    html.Label("Alpha:"),
                    dcc.Slider(
                        id="alpha-slider",
                        min=self.tuner.config.alpha_range[0],
                        max=self.tuner.config.alpha_range[1],
                        step=0.05,
                        value=0.5
                    ),
                    
                    html.Button(
                        "Apply Parameters",
                        id="apply-button"
                    )
                ], id="manual-controls"),
                
                # Auto-tuning controls
                html.Div([
                    html.Label("Max Trials:"),
                    dcc.Input(
                        id="max-trials-input",
                        type="number",
                        value=self.tuner.config.max_trials
                    ),
                    
                    html.Label("Patience:"),
                    dcc.Input(
                        id="patience-input",
                        type="number",
                        value=self.tuner.config.patience
                    ),
                    
                    html.Button(
                        "Start Auto-tuning",
                        id="auto-tune-button"
                    )
                ], id="auto-controls")
            ]),
            
            # Visualization panel
            html.Div([
                html.H2("Visualization"),
                
                # Metrics plots
                dcc.Graph(id="losses-plot"),
                dcc.Graph(id="metrics-plot"),
                dcc.Graph(id="tuning-plot"),
                
                # Summary
                html.Div(id="tuning-summary")
            ]),
            
            # Update interval
            dcc.Interval(
                id="interval-component",
                interval=2000,  # 2 seconds
                n_intervals=0
            )
        ])
        
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output("manual-controls", "style"),
             Output("auto-controls", "style")],
            Input("mode-selector", "value")
        )
        def update_controls(mode):
            """Update control visibility."""
            if mode == TuningMode.MANUAL.value:
                return {"display": "block"}, {"display": "none"}
            else:
                return {"display": "none"}, {"display": "block"}
                
        @self.app.callback(
            Output("tuning-summary", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_summary(_):
            """Update tuning summary."""
            summary = self.tuner.get_tuning_summary()
            if not summary:
                return "No tuning data available"
                
            return html.Div([
                html.P(f"Trials: {summary['trials']}"),
                html.P(f"Best Score: {summary['best_score']:.3f}"),
                html.P(f"Score Trend: {summary['score_trend']:.3f}"),
                html.P(f"Steps without Improvement: {summary['no_improvement_steps']}")
            ])
            
        @self.app.callback(
            [Output("losses-plot", "figure"),
             Output("metrics-plot", "figure"),
             Output("tuning-plot", "figure")],
            Input("interval-component", "n_intervals")
        )
        def update_plots(_):
            """Update visualization plots."""
            return (
                self.visualizer.plot_losses(),
                self.visualizer.plot_metrics(),
                self.visualizer.plot_tuning(self.tuner)
            )
            
        @self.app.callback(
            Output("apply-button", "n_clicks"),
            [Input("apply-button", "n_clicks"),
             State("temperature-slider", "value"),
             State("alpha-slider", "value")]
        )
        def apply_manual_params(n_clicks, temperature, alpha):
            """Apply manual parameter settings."""
            if n_clicks is None:
                return None
                
            params = {
                "temperature": temperature,
                "alpha": alpha
            }
            
            if self.update_callback:
                self.update_callback(params)
                
            return None
            
        @self.app.callback(
            Output("auto-tune-button", "n_clicks"),
            [Input("auto-tune-button", "n_clicks"),
             State("max-trials-input", "value"),
             State("patience-input", "value")]
        )
        def start_auto_tuning(n_clicks, max_trials, patience):
            """Start automatic tuning."""
            if n_clicks is None:
                return None
                
            # Update tuner config
            self.tuner.config.max_trials = max_trials
            self.tuner.config.patience = patience
            
            # Reset tuner state
            self.tuner.trial = 0
            self.tuner.no_improvement = 0
            self.tuner.history = []
            
            return None
            
    def run(self):
        """Run interactive tuner."""
        self.app.run_server(port=self.port) 
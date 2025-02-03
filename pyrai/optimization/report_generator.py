from typing import Any, Dict, List, Optional, Union, Tuple
import os
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from .experiment_manager import ExperimentManager, ExperimentResult
from .hyperparameter_search import HyperparameterSearch

class ReportGenerator:
    """Generates experiment reports."""
    
    def __init__(
        self,
        experiment_manager: ExperimentManager,
        search: Optional[HyperparameterSearch] = None,
        report_dir: str = "reports"
    ):
        """Initialize report generator.
        
        Args:
            experiment_manager: Experiment manager
            search: Optional hyperparameter search
            report_dir: Report directory
        """
        self.experiment_manager = experiment_manager
        self.search = search
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(
        self,
        experiment_names: Optional[List[str]] = None
    ) -> str:
        """Generate HTML report.
        
        Args:
            experiment_names: Optional list of experiments to include
            
        Returns:
            Path to generated report
        """
        # Get experiments
        if experiment_names:
            results = [
                self.experiment_manager.load_result(name)
                for name in experiment_names
            ]
            results = [r for r in results if r]
        else:
            results = self.experiment_manager.get_all_results()
            
        # Create report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"report_{timestamp}.html"
        
        with open(report_path, "w") as f:
            f.write(self._generate_html(results))
            
        return str(report_path)
        
    def _generate_html(self, results: List[ExperimentResult]) -> str:
        """Generate HTML content."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Distillation Experiments Report</title>",
            self._get_style(),
            "</head>",
            "<body>",
            "<h1>Distillation Experiments Report</h1>",
            "<hr>",
            self._generate_summary_section(results),
            "<hr>",
            self._generate_comparison_section(results),
            "<hr>"
        ]
        
        # Add hyperparameter search section if available
        if self.search:
            html.extend([
                self._generate_search_section(),
                "<hr>"
            ])
            
        # Add individual experiment sections
        for result in results:
            html.extend([
                self._generate_experiment_section(result),
                "<hr>"
            ])
            
        html.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html)
        
    def _get_style(self) -> str:
        """Get CSS style."""
        return """
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
            }
            .section {
                margin: 20px 0;
            }
            .metric-card {
                border: 1px solid #ddd;
                padding: 10px;
                margin: 10px;
                display: inline-block;
            }
            .plot {
                margin: 20px 0;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f5f5f5;
            }
        </style>
        """
        
    def _generate_summary_section(
        self,
        results: List[ExperimentResult]
    ) -> str:
        """Generate summary section."""
        total = len(results)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(1 for r in results if r.status == "failed")
        
        return f"""
        <div class="section">
            <h2>Summary</h2>
            <div class="metric-card">
                <h3>Total Experiments</h3>
                <p>{total}</p>
            </div>
            <div class="metric-card">
                <h3>Completed</h3>
                <p>{completed}</p>
            </div>
            <div class="metric-card">
                <h3>Failed</h3>
                <p>{failed}</p>
            </div>
        </div>
        """
        
    def _generate_comparison_section(
        self,
        results: List[ExperimentResult]
    ) -> str:
        """Generate comparison section."""
        # Create comparison plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Accuracy Gap",
                "Training Loss",
                "Training Time",
                "Resource Usage"
            ]
        )
        
        for result in results:
            name = result.config.name
            metrics = result.metrics
            
            # Accuracy gap
            if "val_accuracy_gap" in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(metrics["val_accuracy_gap"]))),
                        y=metrics["val_accuracy_gap"],
                        name=f"{name} (Val)"
                    ),
                    row=1, col=1
                )
                
            # Training loss
            if "train_loss" in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(metrics["train_loss"]))),
                        y=metrics["train_loss"],
                        name=name
                    ),
                    row=1, col=2
                )
                
            # Add other plots...
            
        fig.update_layout(height=800)
        
        return f"""
        <div class="section">
            <h2>Experiment Comparison</h2>
            <div class="plot">
                {fig.to_html(full_html=False)}
            </div>
        </div>
        """
        
    def _generate_search_section(self) -> str:
        """Generate hyperparameter search section."""
        history = self.search.get_search_history()
        
        # Create history plot
        fig = go.Figure()
        
        x = [h["number"] for h in history]
        y = [h["value"] for h in history]
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name="Objective Value"
        ))
        
        fig.update_layout(
            title="Hyperparameter Search History",
            xaxis_title="Trial Number",
            yaxis_title="Objective Value"
        )
        
        return f"""
        <div class="section">
            <h2>Hyperparameter Search</h2>
            <div class="plot">
                {fig.to_html(full_html=False)}
            </div>
            <div class="plot">
                {self.search.plot_param_importances().to_html(full_html=False)}
            </div>
            <div class="plot">
                {self.search.plot_parallel_coordinate().to_html(full_html=False)}
            </div>
        </div>
        """
        
    def _generate_experiment_section(
        self,
        result: ExperimentResult
    ) -> str:
        """Generate individual experiment section."""
        return f"""
        <div class="section">
            <h2>Experiment: {result.config.name}</h2>
            <p><strong>Description:</strong> {result.config.description}</p>
            <p><strong>Status:</strong> {result.status}</p>
            <p><strong>Duration:</strong> {result.duration:.2f}s</p>
            {self._generate_metrics_table(result.metrics)}
            {self._generate_metrics_plots(result.metrics)}
        </div>
        """
        
    def _generate_metrics_table(
        self,
        metrics: Dict[str, List[float]]
    ) -> str:
        """Generate metrics table."""
        if not metrics:
            return ""
            
        rows = []
        for name, values in metrics.items():
            if values:
                rows.append(f"""
                <tr>
                    <td>{name}</td>
                    <td>{min(values):.4f}</td>
                    <td>{max(values):.4f}</td>
                    <td>{np.mean(values):.4f}</td>
                    <td>{values[-1]:.4f}</td>
                </tr>
                """)
                
        return f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
                <th>Final</th>
            </tr>
            {"".join(rows)}
        </table>
        """
        
    def _generate_metrics_plots(
        self,
        metrics: Dict[str, List[float]]
    ) -> str:
        """Generate metrics plots."""
        if not metrics:
            return ""
            
        fig = go.Figure()
        
        for name, values in metrics.items():
            fig.add_trace(go.Scatter(
                x=list(range(len(values))),
                y=values,
                name=name
            ))
            
        fig.update_layout(
            title="Training Metrics",
            xaxis_title="Step",
            yaxis_title="Value"
        )
        
        return f"""
        <div class="plot">
            {fig.to_html(full_html=False)}
        </div>
        """ 
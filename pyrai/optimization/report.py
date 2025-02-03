from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dataclasses import dataclass
from .analyzer import PerformanceAnalysis
from .monitor import RuntimeStats, PerformanceMonitor, DistributedMonitor
from .predictor import PerformancePrediction
from .parallel import ParallelStrategy

@dataclass
class PerformanceReport:
    """Performance analysis report."""
    timestamp: str
    model_info: Dict[str, Any]
    static_analysis: PerformanceAnalysis
    runtime_analysis: Optional[Dict[str, Any]]
    recommendations: List[str]
    figures: Dict[str, Any]

class ReportGenerator:
    """Generates performance analysis reports."""
    
    def __init__(
        self,
        output_dir: str = "reports",
        include_plots: bool = True
    ):
        """Initialize report generator.
        
        Args:
            output_dir: Output directory for reports
            include_plots: Whether to include plots
        """
        self.output_dir = output_dir
        self.include_plots = include_plots
        
    def generate_report(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        static_analysis: PerformanceAnalysis,
        runtime_history: Optional[List[RuntimeStats]] = None,
        prediction: Optional[PerformancePrediction] = None,
        strategy: Optional[ParallelStrategy] = None
    ) -> PerformanceReport:
        """Generate performance report.
        
        Args:
            model: Model being analyzed
            static_analysis: Static performance analysis
            runtime_history: Optional runtime statistics
            prediction: Optional performance prediction
            strategy: Optional parallel strategy
            
        Returns:
            Generated report
        """
        # Generate model info
        model_info = self._get_model_info(model)
        
        # Analyze runtime data if available
        runtime_analysis = None
        if runtime_history:
            runtime_analysis = self._analyze_runtime(runtime_history)
            
        # Generate recommendations
        recommendations = static_analysis.recommendations.copy()
        if runtime_analysis:
            recommendations.extend(
                self._generate_runtime_recommendations(runtime_analysis)
            )
            
        # Create visualizations
        figures = {}
        if self.include_plots:
            figures.update(static_analysis.visualizations)
            if runtime_history:
                figures.update(self._create_runtime_plots(runtime_history))
                
        return PerformanceReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_info=model_info,
            static_analysis=static_analysis,
            runtime_analysis=runtime_analysis,
            recommendations=recommendations,
            figures=figures
        )
        
    def save_report(
        self,
        report: PerformanceReport,
        format: str = "html"
    ) -> str:
        """Save report to file.
        
        Args:
            report: Report to save
            format: Output format ('html' or 'pdf')
            
        Returns:
            Path to saved report
        """
        if format == "html":
            return self._save_html(report)
        else:
            return self._save_pdf(report)
            
    def _get_model_info(
        self,
        model: Union[torch.nn.Module, tf.keras.Model]
    ) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "type": type(model).__name__,
            "framework": "pytorch" if isinstance(model, torch.nn.Module) else "tensorflow"
        }
        
        if isinstance(model, torch.nn.Module):
            info.update({
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "layers": len(list(model.modules()))
            })
        else:
            info.update({
                "parameters": model.count_params(),
                "trainable_parameters": sum(
                    np.prod(v.shape) for v in model.trainable_variables
                ),
                "layers": len(model.layers)
            })
            
        return info
        
    def _analyze_runtime(
        self,
        history: List[RuntimeStats]
    ) -> Dict[str, Any]:
        """Analyze runtime statistics."""
        df = pd.DataFrame([vars(stat) for stat in history])
        
        return {
            "avg_gpu_utilization": df["gpu_utilization"].apply(np.mean).mean(),
            "avg_gpu_memory": df["gpu_memory"].apply(np.mean).mean(),
            "avg_cpu_utilization": df["cpu_utilization"].mean(),
            "avg_system_memory": df["system_memory"].mean(),
            "avg_throughput": df["throughput"].mean(),
            "avg_latency": df["latency"].mean(),
            "throughput_stability": df["throughput"].std() / df["throughput"].mean()
        }
        
    def _generate_runtime_recommendations(
        self,
        runtime_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations from runtime analysis."""
        recommendations = []
        
        if runtime_analysis["avg_gpu_utilization"] < 70:
            recommendations.append(
                "GPU utilization is low. Consider increasing batch size or reducing "
                "communication overhead."
            )
            
        if runtime_analysis["throughput_stability"] > 0.1:
            recommendations.append(
                "High throughput variance detected. Check for potential system "
                "bottlenecks or load imbalance."
            )
            
        return recommendations
        
    def _create_runtime_plots(
        self,
        history: List[RuntimeStats]
    ) -> Dict[str, Any]:
        """Create runtime visualization plots."""
        df = pd.DataFrame([vars(stat) for stat in history])
        figs = {}
        
        # Throughput over time
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["timestamp"] - df["timestamp"].min(), df["throughput"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title("Training Throughput")
        figs["throughput"] = fig
        
        # Resource utilization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        time = df["timestamp"] - df["timestamp"].min()
        ax1.plot(time, df["gpu_utilization"].apply(np.mean), label="GPU")
        ax1.plot(time, df["cpu_utilization"], label="CPU")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Utilization (%)")
        ax1.legend()
        ax1.set_title("Resource Utilization")
        
        ax2.plot(time, df["gpu_memory"].apply(np.mean), label="GPU")
        ax2.plot(time, df["system_memory"], label="System")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Memory Usage (%)")
        ax2.legend()
        
        figs["resources"] = fig
        
        return figs 

class InteractiveReportGenerator(ReportGenerator):
    """Interactive performance report generator."""
    
    def __init__(
        self,
        output_dir: str = "reports",
        template_dir: str = "templates",
        update_interval: float = 1.0
    ):
        """Initialize interactive report generator.
        
        Args:
            output_dir: Output directory
            template_dir: HTML template directory
            update_interval: Report update interval
        """
        super().__init__(output_dir, include_plots=True)
        self.template_dir = template_dir
        self.update_interval = update_interval
        
    def generate_live_report(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        monitor: Union[PerformanceMonitor, DistributedMonitor],
        analyzer: 'PerformanceAnalyzer'
    ) -> str:
        """Generate live updating report.
        
        Args:
            model: Model being analyzed
            monitor: Performance monitor
            analyzer: Performance analyzer
            
        Returns:
            Path to live report
        """
        from jinja2 import Environment, FileSystemLoader
        import json
        
        # Setup template environment
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template("live_report.html")
        
        # Generate initial report
        initial_data = {
            "model_info": self._get_model_info(model),
            "runtime_stats": monitor.get_stats(),
            "recommendations": []
        }
        
        # Create report file
        report_path = f"{self.output_dir}/live_report_{int(time.time())}.html"
        with open(report_path, "w") as f:
            f.write(template.render(
                initial_data=json.dumps(initial_data),
                update_interval=self.update_interval * 1000  # Convert to ms
            ))
            
        # Start update server
        self._start_update_server(monitor, analyzer)
        
        return report_path
        
    def _start_update_server(
        self,
        monitor: Union[PerformanceMonitor, DistributedMonitor],
        analyzer: 'PerformanceAnalyzer'
    ):
        """Start websocket server for live updates."""
        import asyncio
        import websockets
        import json
        
        async def update_client(websocket, path):
            while True:
                # Get current stats
                stats = (
                    monitor.get_global_stats()
                    if isinstance(monitor, DistributedMonitor)
                    else monitor.get_stats()
                )
                
                # Generate plots
                if self.include_plots:
                    plots = self._create_runtime_plots(monitor.history)
                    # Convert plots to base64
                    encoded_plots = self._encode_plots(plots)
                else:
                    encoded_plots = {}
                    
                # Send update
                await websocket.send(json.dumps({
                    "stats": stats,
                    "plots": encoded_plots,
                    "recommendations": analyzer._generate_runtime_recommendations(stats)
                }))
                
                await asyncio.sleep(self.update_interval)
                
        # Start server
        server = websockets.serve(update_client, "localhost", 8765)
        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()
        
    def _encode_plots(self, plots: Dict[str, plt.Figure]) -> Dict[str, str]:
        """Convert matplotlib figures to base64 strings."""
        import io
        import base64
        
        encoded = {}
        for name, fig in plots.items():
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            encoded[name] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
        return encoded 
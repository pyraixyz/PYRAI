from typing import Any, Dict, List, Optional, Union, Tuple
import time
import json
from dataclasses import dataclass, asdict
from enum import Enum
from .profiler import ProfileResult, ProfilerMetric
from .monitor import RuntimeStats

class ReportFormat(Enum):
    """Performance report formats."""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"

@dataclass
class PerformanceReport:
    """Performance report details."""
    timestamp: float
    runtime_stats: RuntimeStats
    profile_results: List[ProfileResult]
    bottlenecks: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]

class PerformanceReporter:
    """Generates performance analysis reports."""
    
    def __init__(
        self,
        report_dir: str = "reports",
        format: ReportFormat = ReportFormat.MARKDOWN
    ):
        """Initialize reporter.
        
        Args:
            report_dir: Directory for saving reports
            format: Report format
        """
        self.report_dir = report_dir
        self.format = format
        self.reports: List[PerformanceReport] = []
        
    def generate_report(
        self,
        stats: RuntimeStats,
        profile_results: List[ProfileResult]
    ) -> PerformanceReport:
        """Generate performance report.
        
        Args:
            stats: Runtime statistics
            profile_results: Profiling results
            
        Returns:
            Performance report
        """
        # Extract bottlenecks and recommendations
        bottlenecks = []
        recommendations = []
        for result in profile_results:
            if result.bottleneck_score >= 0.7:
                bottlenecks.append(f"{result.metric.value}: {result.bottleneck_score:.2f}")
                recommendations.extend(result.optimization_suggestions)
                
        # Calculate key metrics
        metrics = {
            "throughput": stats.throughput,
            "latency": stats.latency,
            "gpu_utilization": np.mean(stats.gpu_utilization),
            "memory_utilization": np.mean(stats.gpu_memory),
            "communication_overhead": stats.communication_overhead
        }
        
        # Create report
        report = PerformanceReport(
            timestamp=time.time(),
            runtime_stats=stats,
            profile_results=profile_results,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            metrics=metrics
        )
        
        self.reports.append(report)
        return report
        
    def save_report(self, report: PerformanceReport, filename: Optional[str] = None):
        """Save performance report.
        
        Args:
            report: Performance report to save
            filename: Optional filename
        """
        if filename is None:
            filename = f"report_{int(report.timestamp)}"
            
        if self.format == ReportFormat.TEXT:
            self._save_text_report(report, filename)
        elif self.format == ReportFormat.JSON:
            self._save_json_report(report, filename)
        elif self.format == ReportFormat.HTML:
            self._save_html_report(report, filename)
        else:  # Markdown
            self._save_markdown_report(report, filename)
            
    def _save_text_report(self, report: PerformanceReport, filename: str):
        """Save report in text format."""
        with open(f"{self.report_dir}/{filename}.txt", "w") as f:
            f.write("Performance Report\n")
            f.write("=================\n\n")
            
            f.write("Metrics:\n")
            for metric, value in report.metrics.items():
                f.write(f"- {metric}: {value:.3f}\n")
                
            f.write("\nBottlenecks:\n")
            for bottleneck in report.bottlenecks:
                f.write(f"- {bottleneck}\n")
                
            f.write("\nRecommendations:\n")
            for rec in report.recommendations:
                f.write(f"- {rec}\n")
                
    def _save_json_report(self, report: PerformanceReport, filename: str):
        """Save report in JSON format."""
        report_dict = {
            "timestamp": report.timestamp,
            "metrics": report.metrics,
            "bottlenecks": report.bottlenecks,
            "recommendations": report.recommendations,
            "profile_results": [
                {
                    "metric": result.metric.value,
                    "value": result.value,
                    "bottleneck_score": result.bottleneck_score,
                    "suggestions": result.optimization_suggestions
                }
                for result in report.profile_results
            ]
        }
        
        with open(f"{self.report_dir}/{filename}.json", "w") as f:
            json.dump(report_dict, f, indent=2)
            
    def _save_html_report(self, report: PerformanceReport, filename: str):
        """Save report in HTML format."""
        html = f"""
        <html>
        <head>
            <title>Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .bottleneck {{ color: red; }}
                .recommendation {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
            <h2>Metrics</h2>
            {''.join(f'<div class="metric">{k}: {v:.3f}</div>' for k, v in report.metrics.items())}
            
            <h2>Bottlenecks</h2>
            {''.join(f'<div class="bottleneck">{b}</div>' for b in report.bottlenecks)}
            
            <h2>Recommendations</h2>
            {''.join(f'<div class="recommendation">{r}</div>' for r in report.recommendations)}
        </body>
        </html>
        """
        
        with open(f"{self.report_dir}/{filename}.html", "w") as f:
            f.write(html)
            
    def _save_markdown_report(self, report: PerformanceReport, filename: str):
        """Save report in Markdown format."""
        markdown = f"""
# Performance Report

## Metrics
{chr(10).join(f'- **{k}**: {v:.3f}' for k, v in report.metrics.items())}

## Bottlenecks
{chr(10).join(f'- {b}' for b in report.bottlenecks)}

## Recommendations
{chr(10).join(f'- {r}' for r in report.recommendations)}

## Profile Results
{chr(10).join(f'### {r.metric.value}\n- Value: {r.value:.3f}\n- Bottleneck Score: {r.bottleneck_score:.2f}\n- Suggestions:\n  {chr(10).join(f"  - {s}" for s in r.optimization_suggestions)}' for r in report.profile_results)}
        """
        
        with open(f"{self.report_dir}/{filename}.md", "w") as f:
            f.write(markdown) 
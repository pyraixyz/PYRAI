from typing import Any, Dict, List, Optional, Union, Tuple
import os
import json
import csv
import yaml
from pathlib import Path
import pandas as pd
from .experiment_manager import ExperimentResult
from .report_generator import ReportGenerator

class ExportFormat(Enum):
    """Export formats."""
    JSON = "json"
    CSV = "csv"
    YAML = "yaml"
    EXCEL = "excel"
    HTML = "html"
    MARKDOWN = "markdown"

class ExportManager:
    """Manages experiment exports."""
    
    def __init__(
        self,
        export_dir: str = "exports",
        report_generator: Optional[ReportGenerator] = None
    ):
        """Initialize export manager.
        
        Args:
            export_dir: Export directory
            report_generator: Optional report generator
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.report_generator = report_generator
        
    def export_results(
        self,
        results: List[ExperimentResult],
        format: ExportFormat,
        filename: Optional[str] = None
    ) -> str:
        """Export experiment results.
        
        Args:
            results: Experiment results
            format: Export format
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}"
            
        if format == ExportFormat.JSON:
            return self._export_json(results, filename)
        elif format == ExportFormat.CSV:
            return self._export_csv(results, filename)
        elif format == ExportFormat.YAML:
            return self._export_yaml(results, filename)
        elif format == ExportFormat.EXCEL:
            return self._export_excel(results, filename)
        elif format == ExportFormat.HTML:
            return self._export_html(results, filename)
        else:  # MARKDOWN
            return self._export_markdown(results, filename)
            
    def _export_json(
        self,
        results: List[ExperimentResult],
        filename: str
    ) -> str:
        """Export to JSON."""
        path = self.export_dir / f"{filename}.json"
        
        data = [
            {
                "name": r.config.name,
                "description": r.config.description,
                "status": r.status.value,
                "metrics": r.metrics,
                "duration": r.duration,
                "error": r.error
            }
            for r in results
        ]
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
        return str(path)
        
    def _export_csv(
        self,
        results: List[ExperimentResult],
        filename: str
    ) -> str:
        """Export to CSV."""
        path = self.export_dir / f"{filename}.csv"
        
        # Flatten metrics
        rows = []
        for r in results:
            row = {
                "name": r.config.name,
                "description": r.config.description,
                "status": r.status.value,
                "duration": r.duration,
                "error": r.error
            }
            
            for metric, values in r.metrics.items():
                if values:
                    row[f"{metric}_min"] = min(values)
                    row[f"{metric}_max"] = max(values)
                    row[f"{metric}_mean"] = sum(values) / len(values)
                    row[f"{metric}_final"] = values[-1]
                    
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        
        return str(path)
        
    def _export_yaml(
        self,
        results: List[ExperimentResult],
        filename: str
    ) -> str:
        """Export to YAML."""
        path = self.export_dir / f"{filename}.yaml"
        
        data = [
            {
                "name": r.config.name,
                "description": r.config.description,
                "status": r.status.value,
                "metrics": r.metrics,
                "duration": r.duration,
                "error": r.error
            }
            for r in results
        ]
        
        with open(path, "w") as f:
            yaml.dump(data, f, sort_keys=False)
            
        return str(path)
        
    def _export_excel(
        self,
        results: List[ExperimentResult],
        filename: str
    ) -> str:
        """Export to Excel."""
        path = self.export_dir / f"{filename}.xlsx"
        
        with pd.ExcelWriter(path) as writer:
            # Summary sheet
            summary_rows = []
            for r in results:
                row = {
                    "name": r.config.name,
                    "description": r.config.description,
                    "status": r.status.value,
                    "duration": r.duration,
                    "error": r.error
                }
                summary_rows.append(row)
                
            pd.DataFrame(summary_rows).to_excel(
                writer,
                sheet_name="Summary",
                index=False
            )
            
            # Metrics sheets
            for r in results:
                metrics_df = pd.DataFrame(r.metrics)
                metrics_df.to_excel(
                    writer,
                    sheet_name=r.config.name[:31],  # Excel sheet name length limit
                    index=False
                )
                
        return str(path)
        
    def _export_html(
        self,
        results: List[ExperimentResult],
        filename: str
    ) -> str:
        """Export to HTML."""
        if not self.report_generator:
            raise ValueError("Report generator required for HTML export")
            
        path = self.export_dir / f"{filename}.html"
        
        # Generate report
        html = self.report_generator.generate_report(
            [r.config.name for r in results]
        )
        
        # Copy report
        import shutil
        shutil.copy2(html, path)
        
        return str(path)
        
    def _export_markdown(
        self,
        results: List[ExperimentResult],
        filename: str
    ) -> str:
        """Export to Markdown."""
        path = self.export_dir / f"{filename}.md"
        
        lines = ["# Experiment Results\n"]
        
        for r in results:
            lines.extend([
                f"## {r.config.name}",
                f"Description: {r.config.description}",
                f"Status: {r.status.value}",
                f"Duration: {r.duration:.2f}s",
                ""
            ])
            
            if r.error:
                lines.extend([
                    "### Error",
                    f"```",
                    r.error,
                    "```\n"
                ])
                
            lines.append("### Metrics\n")
            
            for metric, values in r.metrics.items():
                if values:
                    lines.extend([
                        f"#### {metric}",
                        f"- Min: {min(values):.4f}",
                        f"- Max: {max(values):.4f}",
                        f"- Mean: {sum(values)/len(values):.4f}",
                        f"- Final: {values[-1]:.4f}\n"
                    ])
                    
        with open(path, "w") as f:
            f.write("\n".join(lines))
            
        return str(path) 
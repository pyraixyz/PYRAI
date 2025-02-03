from typing import Any, Dict, List, Optional, Union, Tuple
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
import json
import yaml
import pandas as pd
from .export_manager import ExportFormat
from .experiment_manager import ExperimentResult

class ExportPreview:
    """Preview experiment exports."""
    
    def __init__(self, max_preview_lines: int = 100):
        """Initialize preview.
        
        Args:
            max_preview_lines: Maximum lines to preview
        """
        self.max_preview_lines = max_preview_lines
        self.root = tk.Tk()
        self.root.title("Export Preview")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup preview UI."""
        # Format selector
        format_frame = ttk.Frame(self.root)
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        self.format_var = tk.StringVar(value=ExportFormat.JSON.value)
        format_menu = ttk.OptionMenu(
            format_frame,
            self.format_var,
            ExportFormat.JSON.value,
            *[f.value for f in ExportFormat],
            command=self.update_preview
        )
        format_menu.pack(side=tk.LEFT, padx=5)
        
        # Preview area
        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview = scrolledtext.ScrolledText(
            preview_frame,
            wrap=tk.WORD,
            width=80,
            height=30
        )
        self.preview.pack(fill=tk.BOTH, expand=True)
        
    def show_preview(
        self,
        results: List[ExperimentResult]
    ):
        """Show export preview.
        
        Args:
            results: Experiment results to preview
        """
        self.results = results
        self.update_preview()
        self.root.mainloop()
        
    def update_preview(self, *args):
        """Update preview content."""
        format = ExportFormat(self.format_var.get())
        content = self._generate_preview(format)
        
        self.preview.delete(1.0, tk.END)
        self.preview.insert(1.0, content)
        
    def _generate_preview(self, format: ExportFormat) -> str:
        """Generate preview content."""
        try:
            if format == ExportFormat.JSON:
                return self._preview_json()
            elif format == ExportFormat.CSV:
                return self._preview_csv()
            elif format == ExportFormat.YAML:
                return self._preview_yaml()
            elif format == ExportFormat.EXCEL:
                return self._preview_excel()
            elif format == ExportFormat.HTML:
                return self._preview_html()
            else:  # MARKDOWN
                return self._preview_markdown()
        except Exception as e:
            return f"Error generating preview: {str(e)}"
            
    def _preview_json(self) -> str:
        """Generate JSON preview."""
        data = [
            {
                "name": r.config.name,
                "description": r.config.description,
                "status": r.status.value,
                "metrics": {
                    k: v[:3] + ["..."] if len(v) > 3 else v
                    for k, v in r.metrics.items()
                }
            }
            for r in self.results[:3]
        ]
        
        if len(self.results) > 3:
            data.append({"...": "..."})
            
        return json.dumps(data, indent=2)
        
    def _preview_csv(self) -> str:
        """Generate CSV preview."""
        rows = []
        for r in self.results[:3]:
            row = {
                "name": r.config.name,
                "description": r.config.description,
                "status": r.status.value
            }
            
            for metric, values in r.metrics.items():
                if values:
                    row[f"{metric}_min"] = min(values)
                    row[f"{metric}_max"] = max(values)
                    row[f"{metric}_mean"] = sum(values) / len(values)
                    
            rows.append(row)
            
        if len(self.results) > 3:
            rows.append({"name": "...", "description": "...", "status": "..."})
            
        df = pd.DataFrame(rows)
        return df.to_string()
        
    def _preview_yaml(self) -> str:
        """Generate YAML preview."""
        data = [
            {
                "name": r.config.name,
                "description": r.config.description,
                "status": r.status.value,
                "metrics": {
                    k: v[:3] + ["..."] if len(v) > 3 else v
                    for k, v in r.metrics.items()
                }
            }
            for r in self.results[:3]
        ]
        
        if len(self.results) > 3:
            data.append({"...": "..."})
            
        return yaml.dump(data, sort_keys=False)
        
    def _preview_excel(self) -> str:
        """Generate Excel preview."""
        preview = ["Excel Export Preview", ""]
        
        # Summary sheet preview
        preview.extend([
            "Sheet: Summary",
            "----------------------------------------"
        ])
        
        rows = []
        for r in self.results[:3]:
            rows.append([
                r.config.name,
                r.config.description,
                r.status.value
            ])
            
        if len(self.results) > 3:
            rows.append(["...", "...", "..."])
            
        preview.extend([
            "Name\tDescription\tStatus",
            *["\t".join(row) for row in rows],
            ""
        ])
        
        # Metrics sheet preview
        if self.results:
            r = self.results[0]
            preview.extend([
                f"Sheet: {r.config.name}",
                "----------------------------------------"
            ])
            
            metrics_preview = []
            for metric, values in r.metrics.items():
                if values:
                    metrics_preview.append([
                        metric,
                        *[str(v) for v in values[:3]],
                        "..." if len(values) > 3 else ""
                    ])
                    
            preview.extend([
                "Metric\tValue1\tValue2\tValue3",
                *["\t".join(row) for row in metrics_preview]
            ])
            
        return "\n".join(preview)
        
    def _preview_html(self) -> str:
        """Generate HTML preview."""
        return """
        <html>
        <head>
            <title>Export Preview</title>
        </head>
        <body>
            <h1>Experiment Results</h1>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Description</th>
                    <th>Status</th>
                </tr>
                {}
            </table>
            ...
        </body>
        </html>
        """.format(
            "\n".join(
                f"""
                <tr>
                    <td>{r.config.name}</td>
                    <td>{r.config.description}</td>
                    <td>{r.status.value}</td>
                </tr>
                """
                for r in self.results[:3]
            )
        )
        
    def _preview_markdown(self) -> str:
        """Generate Markdown preview."""
        lines = ["# Experiment Results\n"]
        
        for r in self.results[:3]:
            lines.extend([
                f"## {r.config.name}",
                f"Description: {r.config.description}",
                f"Status: {r.status.value}",
                "",
                "### Metrics",
                ""
            ])
            
            for metric, values in r.metrics.items():
                if values:
                    lines.extend([
                        f"#### {metric}",
                        "```",
                        str(values[:3]) + (" ..." if len(values) > 3 else ""),
                        "```\n"
                    ])
                    
        if len(self.results) > 3:
            lines.extend([
                "...",
                "",
                "_Preview truncated_"
            ])
            
        return "\n".join(lines) 
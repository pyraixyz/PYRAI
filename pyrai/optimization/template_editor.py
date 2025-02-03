from typing import Any, Dict, List, Optional, Union, Tuple
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
import json
from .report_templates import TemplateManager, CustomTemplate

class TemplateEditor:
    """Interactive template editor."""
    
    def __init__(
        self,
        template_manager: TemplateManager,
        save_dir: str = "templates"
    ):
        """Initialize editor.
        
        Args:
            template_manager: Template manager
            save_dir: Template save directory
        """
        self.template_manager = template_manager
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.root = tk.Tk()
        self.root.title("Template Editor")
        self.setup_ui()
        
    def setup_ui(self):
        """Setup editor UI."""
        # Template list
        list_frame = ttk.Frame(self.root)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(list_frame, text="Templates").pack()
        self.template_list = tk.Listbox(list_frame, width=30)
        self.template_list.pack(fill=tk.Y, expand=True)
        self.template_list.bind('<<ListboxSelect>>', self.load_template)
        
        # Editor
        editor_frame = ttk.Frame(self.root)
        editor_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Template name
        name_frame = ttk.Frame(editor_frame)
        name_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(name_frame)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Template content
        ttk.Label(editor_frame, text="Content:").pack(anchor=tk.W, padx=5)
        self.editor = scrolledtext.ScrolledText(editor_frame, wrap=tk.WORD)
        self.editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(editor_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            button_frame,
            text="New",
            command=self.new_template
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            button_frame,
            text="Save",
            command=self.save_template
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Delete",
            command=self.delete_template
        ).pack(side=tk.LEFT)
        
        self.update_template_list()
        
    def run(self):
        """Run editor."""
        self.root.mainloop()
        
    def update_template_list(self):
        """Update template list."""
        self.template_list.delete(0, tk.END)
        for name in self.template_manager.templates:
            self.template_list.insert(tk.END, name)
            
    def load_template(self, event):
        """Load selected template."""
        selection = self.template_list.curselection()
        if not selection:
            return
            
        name = self.template_list.get(selection[0])
        template = self.template_manager.get_template(name)
        
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, name)
        
        self.editor.delete(1.0, tk.END)
        if isinstance(template, CustomTemplate):
            self.editor.insert(1.0, template.template)
            
    def new_template(self):
        """Create new template."""
        self.name_entry.delete(0, tk.END)
        self.editor.delete(1.0, tk.END)
        
        # Insert template skeleton
        self.editor.insert(1.0, """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            {{ style }}
        </head>
        <body>
            <!-- Add your template here -->
        </body>
        </html>
        """)
        
    def save_template(self):
        """Save current template."""
        name = self.name_entry.get().strip()
        if not name:
            return
            
        content = self.editor.get(1.0, tk.END).strip()
        if not content:
            return
            
        # Create and register template
        template = CustomTemplate(name, content)
        self.template_manager.register_template(template)
        
        # Save to file
        template_path = self.save_dir / f"{name}.html"
        with open(template_path, "w") as f:
            f.write(content)
            
        self.update_template_list()
        
    def delete_template(self):
        """Delete current template."""
        selection = self.template_list.curselection()
        if not selection:
            return
            
        name = self.template_list.get(selection[0])
        if name in ["basic", "detailed"]:
            return  # Don't delete built-in templates
            
        # Remove from manager
        if name in self.template_manager.templates:
            del self.template_manager.templates[name]
            
        # Delete file
        template_path = self.save_dir / f"{name}.html"
        if template_path.exists():
            template_path.unlink()
            
        self.update_template_list()
        self.new_template() 
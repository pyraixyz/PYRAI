from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import jinja2
from .report_generator import ReportGenerator

class ReportTemplate:
    """Base report template."""
    
    def __init__(self, name: str):
        """Initialize template.
        
        Args:
            name: Template name
        """
        self.name = name
        
    def render(
        self,
        generator: ReportGenerator,
        results: List[Any],
        **kwargs
    ) -> str:
        """Render template.
        
        Args:
            generator: Report generator
            results: Experiment results
            **kwargs: Additional template variables
            
        Returns:
            Rendered HTML
        """
        raise NotImplementedError

class BasicTemplate(ReportTemplate):
    """Basic report template."""
    
    def render(
        self,
        generator: ReportGenerator,
        results: List[Any],
        **kwargs
    ) -> str:
        """Render basic template."""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            {{ style }}
        </head>
        <body>
            <h1>{{ title }}</h1>
            <hr>
            {{ summary }}
            <hr>
            {{ comparison }}
            <hr>
            {% if search %}
                {{ search }}
                <hr>
            {% endif %}
            {% for result in results %}
                {{ experiment(result) }}
                <hr>
            {% endfor %}
        </body>
        </html>
        """
        
        env = jinja2.Environment()
        template = env.from_string(template)
        
        return template.render(
            title=kwargs.get("title", "Distillation Report"),
            style=generator._get_style(),
            summary=generator._generate_summary_section(results),
            comparison=generator._generate_comparison_section(results),
            search=(
                generator._generate_search_section()
                if generator.search else ""
            ),
            experiment=generator._generate_experiment_section,
            results=results
        )

class DetailedTemplate(ReportTemplate):
    """Detailed report template."""
    
    def render(
        self,
        generator: ReportGenerator,
        results: List[Any],
        **kwargs
    ) -> str:
        """Render detailed template."""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            {{ style }}
            <style>
                .toc {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 250px;
                    height: 100%;
                    overflow: auto;
                    background: #f5f5f5;
                    padding: 20px;
                }
                .content {
                    margin-left: 270px;
                    padding: 20px;
                }
                .back-to-top {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                }
            </style>
        </head>
        <body>
            <div class="toc">
                <h2>Contents</h2>
                <ul>
                    <li><a href="#summary">Summary</a></li>
                    <li><a href="#comparison">Comparison</a></li>
                    {% if search %}
                        <li><a href="#search">Search</a></li>
                    {% endif %}
                    {% for result in results %}
                        <li>
                            <a href="#exp-{{ result.config.name }}">
                                {{ result.config.name }}
                            </a>
                        </li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="content">
                <h1>{{ title }}</h1>
                <hr>
                
                <section id="summary">
                    <h2>Summary</h2>
                    {{ summary }}
                </section>
                <hr>
                
                <section id="comparison">
                    <h2>Comparison</h2>
                    {{ comparison }}
                </section>
                <hr>
                
                {% if search %}
                    <section id="search">
                        <h2>Search</h2>
                        {{ search }}
                    </section>
                    <hr>
                {% endif %}
                
                {% for result in results %}
                    <section id="exp-{{ result.config.name }}">
                        {{ experiment(result) }}
                    </section>
                    <hr>
                {% endfor %}
            </div>
            
            <a href="#" class="back-to-top">↑ Top</a>
        </body>
        </html>
        """
        
        env = jinja2.Environment()
        template = env.from_string(template)
        
        return template.render(
            title=kwargs.get("title", "Detailed Distillation Report"),
            style=generator._get_style(),
            summary=generator._generate_summary_section(results),
            comparison=generator._generate_comparison_section(results),
            search=(
                generator._generate_search_section()
                if generator.search else ""
            ),
            experiment=generator._generate_experiment_section,
            results=results
        )

class TemplateManager:
    """Manages report templates."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize template manager.
        
        Args:
            template_dir: Optional custom template directory
        """
        self.templates: Dict[str, ReportTemplate] = {
            "basic": BasicTemplate("basic"),
            "detailed": DetailedTemplate("detailed")
        }
        
        if template_dir:
            self._load_custom_templates(template_dir)
            
    def _load_custom_templates(self, template_dir: str):
        """Load custom templates from directory."""
        template_dir = Path(template_dir)
        
        for path in template_dir.glob("*.html"):
            name = path.stem
            with open(path) as f:
                template = f.read()
                
            self.templates[name] = CustomTemplate(
                name,
                template
            )
            
    def get_template(self, name: str) -> ReportTemplate:
        """Get template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
            
        return self.templates[name]
        
    def register_template(self, template: ReportTemplate):
        """Register new template."""
        self.templates[template.name] = template

class CustomTemplate(ReportTemplate):
    """Custom report template."""
    
    def __init__(self, name: str, template: str):
        """Initialize custom template.
        
        Args:
            name: Template name
            template: Template string
        """
        super().__init__(name)
        self.template = template
        
    def render(
        self,
        generator: ReportGenerator,
        results: List[Any],
        **kwargs
    ) -> str:
        """Render custom template."""
        env = jinja2.Environment()
        template = env.from_string(self.template)
        
        return template.render(
            generator=generator,
            results=results,
            **kwargs
        ) 
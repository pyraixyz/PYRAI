from typing import Any, Dict, List, Optional, Union, Tuple
import re
from dataclasses import dataclass
from enum import Enum
import jinja2
from .report_templates import ReportTemplate

class ValidationLevel(Enum):
    """Template validation levels."""
    BASIC = "basic"      # Syntax only
    STANDARD = "standard"  # Variables and structure
    STRICT = "strict"    # Full validation

@dataclass
class ValidationError:
    """Template validation error."""
    line: int
    message: str
    level: ValidationLevel
    context: Optional[str] = None

class TemplateValidator:
    """Validates report templates."""
    
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        required_vars: Optional[List[str]] = None
    ):
        """Initialize validator.
        
        Args:
            level: Validation level
            required_vars: Required template variables
        """
        self.level = level
        self.required_vars = required_vars or [
            "title", "style", "summary", "comparison",
            "search", "experiment", "results"
        ]
        
    def validate(
        self,
        template: Union[str, ReportTemplate]
    ) -> List[ValidationError]:
        """Validate template.
        
        Args:
            template: Template string or object
            
        Returns:
            List of validation errors
        """
        if isinstance(template, ReportTemplate):
            content = template.template
        else:
            content = template
            
        errors = []
        
        # Basic validation
        errors.extend(self._validate_syntax(content))
        
        if self.level == ValidationLevel.BASIC:
            return errors
            
        # Standard validation
        errors.extend(self._validate_variables(content))
        errors.extend(self._validate_structure(content))
        
        if self.level == ValidationLevel.STANDARD:
            return errors
            
        # Strict validation
        errors.extend(self._validate_accessibility(content))
        errors.extend(self._validate_responsive(content))
        
        return errors
        
    def _validate_syntax(self, content: str) -> List[ValidationError]:
        """Validate template syntax."""
        errors = []
        
        try:
            env = jinja2.Environment()
            env.parse(content)
        except jinja2.TemplateSyntaxError as e:
            errors.append(ValidationError(
                line=e.lineno,
                message=str(e),
                level=ValidationLevel.BASIC
            ))
            
        # Check HTML syntax
        for i, line in enumerate(content.split("\n"), 1):
            # Check for unclosed tags
            if re.search(r"<[^/][^>]*[^/]>", line) and not re.search(r"</\w+>", line):
                if not any(tag in line for tag in ["<img", "<br", "<hr", "<input"]):
                    errors.append(ValidationError(
                        line=i,
                        message="Potentially unclosed HTML tag",
                        level=ValidationLevel.BASIC,
                        context=line.strip()
                    ))
                    
        return errors
        
    def _validate_variables(self, content: str) -> List[ValidationError]:
        """Validate template variables."""
        errors = []
        
        # Check for required variables
        for var in self.required_vars:
            if "{{" + var + "}}" not in content and "{{ " + var + " }}" not in content:
                errors.append(ValidationError(
                    line=0,
                    message=f"Missing required variable: {var}",
                    level=ValidationLevel.STANDARD
                ))
                
        # Check for undefined variables
        env = jinja2.Environment()
        ast = env.parse(content)
        
        undefined_vars = set()
        for node in ast.find_all(jinja2.nodes.Name):
            if node.name not in self.required_vars:
                undefined_vars.add(node.name)
                
        for var in undefined_vars:
            errors.append(ValidationError(
                line=0,
                message=f"Undefined variable: {var}",
                level=ValidationLevel.STANDARD
            ))
            
        return errors
        
    def _validate_structure(self, content: str) -> List[ValidationError]:
        """Validate template structure."""
        errors = []
        
        # Check for required sections
        required_sections = [
            ("<head", "</head>"),
            ("<body", "</body>"),
            ("<title", "</title>")
        ]
        
        for start, end in required_sections:
            if start not in content or end not in content:
                errors.append(ValidationError(
                    line=0,
                    message=f"Missing required section: {start}...{end}",
                    level=ValidationLevel.STANDARD
                ))
                
        return errors
        
    def _validate_accessibility(self, content: str) -> List[ValidationError]:
        """Validate template accessibility."""
        errors = []
        
        # Check for image alt text
        for i, line in enumerate(content.split("\n"), 1):
            if "<img" in line and "alt=" not in line:
                errors.append(ValidationError(
                    line=i,
                    message="Image missing alt text",
                    level=ValidationLevel.STRICT,
                    context=line.strip()
                ))
                
        # Check for ARIA labels
        interactive_elements = ["button", "a", "input", "select"]
        for i, line in enumerate(content.split("\n"), 1):
            for element in interactive_elements:
                if f"<{element}" in line and "aria-label=" not in line:
                    errors.append(ValidationError(
                        line=i,
                        message=f"Interactive element missing ARIA label: {element}",
                        level=ValidationLevel.STRICT,
                        context=line.strip()
                    ))
                    
        return errors
        
    def _validate_responsive(self, content: str) -> List[ValidationError]:
        """Validate template responsiveness."""
        errors = []
        
        # Check for viewport meta tag
        if '<meta name="viewport"' not in content:
            errors.append(ValidationError(
                line=0,
                message="Missing viewport meta tag",
                level=ValidationLevel.STRICT
            ))
            
        # Check for responsive units
        fixed_units = ["px", "pt"]
        for i, line in enumerate(content.split("\n"), 1):
            for unit in fixed_units:
                if f":{unit}" in line or f"; {unit}" in line:
                    errors.append(ValidationError(
                        line=i,
                        message=f"Using fixed unit: {unit}",
                        level=ValidationLevel.STRICT,
                        context=line.strip()
                    ))
                    
        return errors 
from typing import Any, Dict, Optional, Type, List
import importlib
from pathlib import Path

class Registry:
    """Model and component registry."""
    
    def __init__(self):
        """Initialize registry."""
        self._models = {}
        self._components = {}
        
    def register_model(
        self,
        name: str,
        model_cls: Optional[Type] = None,
        **kwargs
    ):
        """Register model class.
        
        Args:
            name: Model name
            model_cls: Model class
            **kwargs: Additional model parameters
        """
        def _register(cls):
            if name in self._models:
                raise ValueError(f"Model {name} already registered")
            self._models[name] = {
                "class": cls,
                "params": kwargs
            }
            return cls
            
        if model_cls is not None:
            return _register(model_cls)
        return _register
        
    def register_component(
        self,
        name: str,
        component_cls: Optional[Type] = None,
        **kwargs
    ):
        """Register component class.
        
        Args:
            name: Component name
            component_cls: Component class
            **kwargs: Additional component parameters
        """
        def _register(cls):
            if name in self._components:
                raise ValueError(f"Component {name} already registered")
            self._components[name] = {
                "class": cls,
                "params": kwargs
            }
            return cls
            
        if component_cls is not None:
            return _register(component_cls)
        return _register
        
    def get_model(
        self,
        name: str,
        **kwargs
    ) -> Any:
        """Get model instance.
        
        Args:
            name: Model name
            **kwargs: Model parameters
            
        Returns:
            Model instance
        """
        if name not in self._models:
            raise ValueError(f"Model {name} not found")
            
        model_info = self._models[name]
        model_cls = model_info["class"]
        model_params = model_info["params"].copy()
        model_params.update(kwargs)
        
        return model_cls(**model_params)
        
    def get_component(
        self,
        name: str,
        **kwargs
    ) -> Any:
        """Get component instance.
        
        Args:
            name: Component name
            **kwargs: Component parameters
            
        Returns:
            Component instance
        """
        if name not in self._components:
            raise ValueError(f"Component {name} not found")
            
        component_info = self._components[name]
        component_cls = component_info["class"]
        component_params = component_info["params"].copy()
        component_params.update(kwargs)
        
        return component_cls(**component_params)
        
    def list_models(self) -> List[str]:
        """Get list of registered models."""
        return list(self._models.keys())
        
    def list_components(self) -> List[str]:
        """Get list of registered components."""
        return list(self._components.keys())
        
    @classmethod
    def load_modules(cls, package_dir: str) -> None:
        """Load all modules in package.
        
        Args:
            package_dir: Package directory path
        """
        package_dir = Path(package_dir)
        if not package_dir.exists():
            raise ValueError(f"Package directory not found: {package_dir}")
            
        for file in package_dir.rglob("*.py"):
            if file.name.startswith("_"):
                continue
                
            module_path = str(file.relative_to(package_dir.parent))
            module_name = module_path.replace("/", ".").replace(".py", "")
            
            try:
                importlib.import_module(module_name)
            except Exception as e:
                print(f"Failed to load module {module_name}: {str(e)}")

# Global registry instance
registry = Registry() 
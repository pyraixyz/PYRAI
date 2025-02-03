"""PyRAI SDK client implementation."""

import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin
import json

from .exceptions import PyRAIError, APIError
from .models import (
    DistillationConfig,
    ModelConfig,
    TaskStatus,
    DistillationResult
)

class PyRAIClient:
    """PyRAI SDK client for interacting with PyRAI API.
    
    Example:
        ```python
        # Initialize client
        client = PyRAIClient(
            base_url="http://localhost:8000",
            api_key="your-api-key"
        )
        
        # Create distillation task
        task_id = client.create_distillation(
            teacher_model=teacher_config,
            student_model=student_config,
            config=distill_config
        )
        
        # Monitor task status
        status = client.get_task_status(task_id)
        ```
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 30
    ):
        """Initialize PyRAI client.
        
        Args:
            base_url: API base URL
            api_key: API authentication key
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = urljoin(self.base_url, endpoint)
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {str(e)}")
            
    def create_distillation(
        self,
        teacher_model: ModelConfig,
        student_model: ModelConfig,
        config: DistillationConfig
    ) -> str:
        """Create new distillation task.
        
        Args:
            teacher_model: Teacher model configuration
            student_model: Student model configuration
            config: Distillation configuration
            
        Returns:
            Task ID for monitoring progress
        """
        data = {
            "teacher_model": teacher_model.dict(),
            "student_model": student_model.dict(),
            "config": config.dict()
        }
        
        response = self._request(
            "POST",
            "/api/v1/distill",
            json=data
        )
        return response["task_id"]
        
    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get status of distillation task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Current task status
        """
        response = self._request(
            "GET",
            f"/api/v1/status/{task_id}"
        )
        return TaskStatus(**response)
        
    def get_available_methods(self) -> List[str]:
        """Get available distillation methods."""
        response = self._request("GET", "/api/v1/methods")
        return response["methods"]
        
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information."""
        response = self._request(
            "GET",
            f"/api/v1/models/{model_id}"
        )
        return response
        
    def monitor_distillation(
        self,
        task_id: str,
        poll_interval: int = 5
    ) -> DistillationResult:
        """Monitor distillation task until completion.
        
        Args:
            task_id: Task identifier
            poll_interval: Status check interval in seconds
            
        Returns:
            Final distillation results
        """
        import time
        
        while True:
            status = self.get_task_status(task_id)
            if status.is_complete():
                return status.result
            time.sleep(poll_interval) 
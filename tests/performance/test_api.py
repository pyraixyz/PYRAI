import pytest
from locust import HttpUser, task, between
from typing import Dict, Any
import json
import random

class APIUser(HttpUser):
    wait_time = between(1, 3)
    token: str = None
    
    def on_start(self):
        """Login and get token."""
        response = self.client.post("/auth/token", {
            "username": "test@example.com",
            "password": "test123"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def get_models(self):
        """Test models list endpoint."""
        self.client.get("/models", headers=self.headers)
    
    @task(2)
    def create_model(self):
        """Test model creation endpoint."""
        model_data = {
            "name": f"test_model_{random.randint(1000, 9999)}",
            "description": "Test model",
            "model_type": "classification",
            "framework": "pytorch",
            "config": {
                "layers": [100, 50, 10],
                "activation": "relu"
            }
        }
        self.client.post("/models", json=model_data, headers=self.headers)
    
    @task(3)
    def get_tasks(self):
        """Test tasks list endpoint."""
        self.client.get("/tasks", headers=self.headers)
    
    @task(1)
    def create_task(self):
        """Test task creation endpoint."""
        task_data = {
            "name": f"test_task_{random.randint(1000, 9999)}",
            "task_type": "training",
            "params": {
                "epochs": 10,
                "batch_size": 32
            }
        }
        self.client.post("/tasks", json=task_data, headers=self.headers)
    
    @task(2)
    def get_metrics(self):
        """Test metrics endpoint."""
        self.client.get("/monitoring/metrics", headers=self.headers)

@pytest.mark.benchmark
def test_model_creation_performance(benchmark):
    """Benchmark model creation."""
    def create_model():
        # Setup test client and data
        from fastapi.testclient import TestClient
        from pyrai.api.server import app
        
        client = TestClient(app)
        model_data = {
            "name": "benchmark_model",
            "description": "Benchmark test",
            "model_type": "classification",
            "framework": "pytorch"
        }
        
        # Create model
        response = client.post("/models", json=model_data)
        assert response.status_code == 201
    
    # Run benchmark
    benchmark(create_model)

@pytest.mark.benchmark
def test_task_creation_performance(benchmark):
    """Benchmark task creation."""
    def create_task():
        from fastapi.testclient import TestClient
        from pyrai.api.server import app
        
        client = TestClient(app)
        task_data = {
            "name": "benchmark_task",
            "task_type": "training",
            "params": {"epochs": 10}
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 201
    
    benchmark(create_task)

@pytest.mark.benchmark
def test_metrics_performance(benchmark):
    """Benchmark metrics endpoint."""
    def get_metrics():
        from fastapi.testclient import TestClient
        from pyrai.api.server import app
        
        client = TestClient(app)
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
    
    benchmark(get_metrics) 
import pytest
from fastapi.testclient import TestClient
from pyrai.main import app

client = TestClient(app)

@pytest.mark.benchmark(
    group="api",
    min_rounds=100,
    warmup=False
)
def test_health_endpoint(benchmark):
    def run_request():
        response = client.get("/health")
        assert response.status_code == 200
    
    benchmark(run_request)

@pytest.mark.benchmark(
    group="api",
    min_rounds=100,
    warmup=True,
    warmup_iterations=10
)
def test_api_latency(benchmark):
    def run_request():
        response = client.get("/api/v1/data")
        assert response.status_code == 200
        assert "data" in response.json()
    
    benchmark(run_request)

@pytest.mark.benchmark(
    group="database",
    min_rounds=50,
    warmup=True
)
def test_db_operations(benchmark, db_session):
    from pyrai.models import Item
    
    def run_db_ops():
        # Create
        item = Item(name="test", description="test")
        db_session.add(item)
        db_session.commit()
        
        # Read
        item = db_session.query(Item).first()
        assert item is not None
        
        # Update
        item.name = "updated"
        db_session.commit()
        
        # Delete
        db_session.delete(item)
        db_session.commit()
    
    benchmark(run_db_ops)

@pytest.mark.benchmark(
    group="cache",
    min_rounds=100,
    warmup=True
)
def test_cache_operations(benchmark, redis_client):
    def run_cache_ops():
        # Set
        redis_client.set("test_key", "test_value")
        
        # Get
        value = redis_client.get("test_key")
        assert value == "test_value"
        
        # Delete
        redis_client.delete("test_key")
    
    benchmark(run_cache_ops) 
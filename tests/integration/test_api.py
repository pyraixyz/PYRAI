import pytest
from fastapi import status

def test_health_check(test_app):
    response = test_app.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

def test_metrics(test_app):
    response = test_app.get("/metrics")
    assert response.status_code == status.HTTP_200_OK

@pytest.mark.parametrize(
    "endpoint,method,status_code",
    [
        ("/docs", "GET", status.HTTP_200_OK),
        ("/openapi.json", "GET", status.HTTP_200_OK),
        ("/nonexistent", "GET", status.HTTP_404_NOT_FOUND),
    ],
)
def test_api_endpoints(test_app, endpoint, method, status_code):
    response = test_app.request(method, endpoint)
    assert response.status_code == status_code

def test_protected_endpoint(test_app, auth_headers):
    # Test without auth
    response = test_app.get("/api/v1/protected")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    # Test with auth
    response = test_app.get("/api/v1/protected", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK

def test_database_operations(test_app, db_session, auth_headers):
    # Create
    response = test_app.post(
        "/api/v1/items",
        headers=auth_headers,
        json={"name": "test item", "description": "test description"}
    )
    assert response.status_code == status.HTTP_201_CREATED
    item_id = response.json()["id"]
    
    # Read
    response = test_app.get(f"/api/v1/items/{item_id}", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["name"] == "test item"
    
    # Update
    response = test_app.put(
        f"/api/v1/items/{item_id}",
        headers=auth_headers,
        json={"name": "updated item", "description": "updated description"}
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["name"] == "updated item"
    
    # Delete
    response = test_app.delete(f"/api/v1/items/{item_id}", headers=auth_headers)
    assert response.status_code == status.HTTP_204_NO_CONTENT

def test_redis_operations(test_app, redis_client, auth_headers):
    # Test caching
    response = test_app.get("/api/v1/cached-data", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK
    
    # Verify cache hit
    cached_response = test_app.get("/api/v1/cached-data", headers=auth_headers)
    assert cached_response.status_code == status.HTTP_200_OK
    assert cached_response.headers.get("X-Cache") == "HIT" 
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from redis import Redis

from pyrai.db import Base
from pyrai.core.config import settings

@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

@pytest.fixture(scope="session")
def redis_client():
    client = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    yield client
    client.flushdb()
    client.close()

@pytest.fixture(scope="session")
def test_app():
    from pyrai.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    yield client

@pytest.fixture(scope="session")
def auth_headers(test_app):
    from pyrai.core.security import create_access_token
    
    access_token = create_access_token(
        data={"sub": "test@example.com"}
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    return headers 
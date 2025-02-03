import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from redis import Redis

from pyrai.db import Base
from pyrai.core.config import settings

@pytest.fixture(scope="session")
def db_engine():
    """Create a test database engine."""
    database_url = settings.DATABASE_URL.replace(
        "postgresql://", "postgresql://", 1
    )
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a new database session for a test."""
    connection = db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="session")
def redis_client():
    """Create a Redis client for testing."""
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
def benchmark_data():
    """Generate test data for benchmarks."""
    return {
        "small": {"size": 100, "data": list(range(100))},
        "medium": {"size": 1000, "data": list(range(1000))},
        "large": {"size": 10000, "data": list(range(10000))}
    } 
#!/bin/bash

# Wait for database
echo "Waiting for database..."
while ! nc -z db 5432; do
  sleep 1
done
echo "Database is ready"

# Wait for Redis
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 1
done
echo "Redis is ready"

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Start application
echo "Starting application..."
uvicorn pyrai.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-config logging.yaml 
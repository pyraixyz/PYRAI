#!/bin/bash
set -e

# Function to wait for a service
wait_for() {
    echo "Waiting for $1..."
    until nc -z "$2" "$3"; do
        sleep 1
    done
    echo "$1 is up"
}

# Wait for required services
if [ "$WAIT_FOR_DB" = "true" ]; then
    wait_for "PostgreSQL" "${DB_HOST:-postgres}" "${DB_PORT:-5432}"
fi

# Run database migrations
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Execute command
exec "$@" 
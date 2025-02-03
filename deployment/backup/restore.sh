#!/bin/bash

# Configuration
BACKUP_DIR="/backup"
POSTGRES_HOST="db"
POSTGRES_PORT="5432"
POSTGRES_USER="pyrai"
POSTGRES_DB="pyrai"
REDIS_HOST="redis"
REDIS_PORT="6379"
MODEL_DIR="/app/data/models"

# Check if backup date is provided
if [ -z "$1" ]; then
    echo "Please provide backup date (YYYYMMDD)"
    exit 1
fi

BACKUP_PATH="${BACKUP_DIR}/$1"

# Check if backup exists
if [ ! -d "${BACKUP_PATH}" ]; then
    echo "Backup not found: ${BACKUP_PATH}"
    exit 1
fi

# Verify backup manifest
if [ ! -f "${BACKUP_PATH}/manifest.json" ]; then
    echo "Backup manifest not found"
    exit 1
fi

# Stop application services
echo "Stopping application services..."
docker-compose stop api

# Restore PostgreSQL database
echo "Restoring PostgreSQL database..."
PGPASSWORD=${POSTGRES_PASSWORD} pg_restore \
    -h ${POSTGRES_HOST} \
    -p ${POSTGRES_PORT} \
    -U ${POSTGRES_USER} \
    -d ${POSTGRES_DB} \
    -c \
    "${BACKUP_PATH}/database.dump"

# Restore Redis data
echo "Restoring Redis data..."
docker-compose stop redis
rm -f /data/redis/dump.rdb
cp "${BACKUP_PATH}/redis.rdb" /data/redis/dump.rdb
docker-compose start redis

# Restore model files
echo "Restoring model files..."
rm -rf ${MODEL_DIR}/*
tar -xzf "${BACKUP_PATH}/models.tar.gz" -C ${MODEL_DIR}

# Start application services
echo "Starting application services..."
docker-compose start api

echo "Restore completed successfully" 
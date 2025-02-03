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

# Create backup directory
mkdir -p "${BACKUP_DIR}/$(date +%Y%m%d)"
BACKUP_PATH="${BACKUP_DIR}/$(date +%Y%m%d)"

# Backup PostgreSQL database
echo "Backing up PostgreSQL database..."
PGPASSWORD=${POSTGRES_PASSWORD} pg_dump \
    -h ${POSTGRES_HOST} \
    -p ${POSTGRES_PORT} \
    -U ${POSTGRES_USER} \
    -d ${POSTGRES_DB} \
    -F c \
    -f "${BACKUP_PATH}/database.dump"

# Backup Redis data
echo "Backing up Redis data..."
redis-cli \
    -h ${REDIS_HOST} \
    -p ${REDIS_PORT} \
    --rdb "${BACKUP_PATH}/redis.rdb"

# Backup model files
echo "Backing up model files..."
tar -czf "${BACKUP_PATH}/models.tar.gz" -C ${MODEL_DIR} .

# Create backup manifest
cat > "${BACKUP_PATH}/manifest.json" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "database": "database.dump",
    "redis": "redis.rdb",
    "models": "models.tar.gz",
    "version": "$(cat /app/VERSION)"
}
EOF

# Cleanup old backups (keep last 7 days)
find ${BACKUP_DIR} -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed successfully" 
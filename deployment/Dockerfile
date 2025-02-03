# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY pyrai pyrai/

# Build wheel
RUN ~/.local/bin/poetry build -f wheel

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy wheel from builder
COPY --from=builder /app/dist/*.whl ./

# Install runtime dependencies
RUN pip install --no-cache-dir *.whl && \
    rm *.whl

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config

# Copy configuration
COPY config/production.yml /app/config/
COPY deployment/start.sh /app/

# Set permissions
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Start application
CMD ["/app/start.sh"] 
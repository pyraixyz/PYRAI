# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Solana CLI tools
RUN sh -c "$(curl -sSfL https://release.solana.com/v1.9.13/install)"

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Set environment variables
ENV PYTHONPATH=/app
ENV SOLANA_NETWORK=devnet

# Expose ports
EXPOSE 8000

# Start API server
CMD ["python", "-m", "pyrai.api.server"] 
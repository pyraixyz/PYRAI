# Base stage for both development and production
FROM python:3.13-slim as base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.4.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Development stage
FROM base as development

# Install development dependencies
RUN poetry install --no-interaction --no-ansi --with dev

# Copy source code
COPY . .

# Run the application with hot reload
CMD ["poetry", "run", "uvicorn", "pyrai.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Install production dependencies only
RUN poetry install --no-interaction --no-ansi --without dev

# Copy source code
COPY . .

# Run the application
CMD ["poetry", "run", "gunicorn", "pyrai.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]

# Build stage
FROM python:3.13-slim as builder

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy source code
COPY pyrai pyrai/
COPY README.md ./

# Build package
RUN poetry build

# Runtime stage
FROM python:3.13-slim

WORKDIR /app

# Copy built package from builder
COPY --from=builder /app/dist/*.whl ./

# Install package
RUN pip install *.whl \
    && rm *.whl

# Create non-root user
RUN useradd -m -u 1000 pyrai
USER pyrai

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYRAI_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import pyrai; print(pyrai.__version__)"

# Command
ENTRYPOINT ["python", "-m", "pyrai"]
CMD ["serve"] 
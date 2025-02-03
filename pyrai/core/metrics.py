from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status"]
)

request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# System metrics
cpu_usage_percent = Gauge(
    "system_cpu_usage_percent",
    "System CPU usage percentage"
)

memory_usage_bytes = Gauge(
    "system_memory_usage_bytes",
    "System memory usage in bytes"
)

# Model metrics
model_inference_total = Counter(
    "model_inference_total",
    "Total number of model inferences",
    ["model_name", "version"]
)

model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Model inference duration in seconds",
    ["model_name", "version"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

model_training_duration_seconds = Histogram(
    "model_training_duration_seconds",
    "Model training duration in seconds",
    ["model_name", "version"],
    buckets=[10, 60, 300, 900, 3600]
)

model_load_duration_seconds = Histogram(
    "model_load_duration_seconds",
    "Model loading duration in seconds",
    ["model_name", "version"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

model_memory_usage_bytes = Gauge(
    "model_memory_usage_bytes",
    "Model memory usage in bytes",
    ["model_name", "version"]
)

model_gpu_memory_bytes = Gauge(
    "model_gpu_memory_bytes",
    "Model GPU memory usage in bytes",
    ["model_name", "version", "device"]
)

# Cache metrics
cache_hits_total = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["cache_name"]
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["cache_name"]
)

# Database metrics
db_connections_active = Gauge(
    "db_connections_active",
    "Number of active database connections"
)

db_query_duration_seconds = Histogram(
    "db_query_duration_seconds",
    "Database query duration in seconds",
    ["operation"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

# Task metrics
task_duration_seconds = Histogram(
    "task_duration_seconds",
    "Task duration in seconds",
    ["task_type"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
)

task_memory_usage_bytes = Gauge(
    "task_memory_usage_bytes",
    "Task memory usage in bytes",
    ["task_id"]
)

task_error_total = Counter(
    "task_errors_total",
    "Total number of task errors",
    ["task_type", "error_type"]
)

# API metrics
api_request_size_bytes = Histogram(
    "api_request_size_bytes",
    "API request size in bytes",
    ["method", "path"],
    buckets=[100, 1000, 10000, 100000]
)

api_response_size_bytes = Histogram(
    "api_response_size_bytes",
    "API response size in bytes",
    ["method", "path"],
    buckets=[100, 1000, 10000, 100000]
)

api_errors_total = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["method", "path", "error_type"]
) 
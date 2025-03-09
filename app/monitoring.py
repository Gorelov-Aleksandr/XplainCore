from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
# Temporarily commented out Jaeger exporter due to protobuf compatibility issues
# from opentelemetry.exporter.jaeger.proto.grpc import JaegerExporter
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk
from loguru import logger

from .config import settings

def setup_monitoring():
    """
    Initializes monitoring and observability tools:
    - Sentry for error tracking
    - OpenTelemetry for distributed tracing (Jaeger temporarily disabled)
    """
    # Initialize Sentry for error monitoring
    if settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            traces_sample_rate=1.0,
            environment=settings.sentry_environment,
        )
    
    # Setup OpenTelemetry with basic configuration (Jaeger temporarily disabled)
    trace.set_tracer_provider(
        TracerProvider(
            resource=Resource.create({"service.name": "xai-service"})
        )
    )
    
    try:
        # Try to import and use JaegerExporter if compatible
        from opentelemetry.exporter.jaeger.proto.grpc import JaegerExporter
        
        jaeger_exporter = JaegerExporter(
            agent_host_name=settings.jaeger_host,
            agent_port=settings.jaeger_port,
        )
        
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
        logger.info("Jaeger tracing enabled")
    except (ImportError, TypeError) as e:
        # If there's an issue with the Jaeger exporter, log it but continue without it
        logger.warning(f"Jaeger exporter not available: {str(e)}")
        logger.info("Running without Jaeger tracing")

# Get tracer for the application
tracer = trace.get_tracer(__name__)

# Prometheus metrics
REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status']
)

RESPONSE_TIME = Histogram(
    'http_response_time_seconds',
    'Response time histogram',
    ['endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'active_requests',
    'Active HTTP Requests',
    ['endpoint']
)

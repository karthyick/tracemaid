"""
tracemaid.integrations - Framework integrations for automatic tracing.

This subpackage provides easy-to-use integrations for popular frameworks,
setting up OpenTelemetry auto-instrumentation with TracemaidExporter.

Example (FastAPI):
    >>> from fastapi import FastAPI
    >>> from tracemaid.integrations import setup_fastapi_tracing, instrument_all
    >>>
    >>> app = FastAPI()
    >>> setup_fastapi_tracing(app, service_name="my-api", output_dir="./traces")
    >>>
    >>> # Enable instrumentation for all available libraries
    >>> instrument_all()
    >>>
    >>> # Now all endpoints automatically generate Mermaid diagrams!
"""

from tracemaid.integrations.fastapi_integration import (
    setup_fastapi_tracing,
    instrument_requests,
    instrument_logging,
    instrument_httpx,
    instrument_sqlalchemy,
    instrument_redis,
    instrument_all,
)
from tracemaid.integrations.setup import setup_tracing, get_tracer, shutdown_tracing

__all__ = [
    # Setup
    "setup_fastapi_tracing",
    "setup_tracing",
    "get_tracer",
    "shutdown_tracing",
    # Individual instrumentation
    "instrument_requests",
    "instrument_logging",
    "instrument_httpx",
    "instrument_sqlalchemy",
    "instrument_redis",
    # Convenience
    "instrument_all",
]

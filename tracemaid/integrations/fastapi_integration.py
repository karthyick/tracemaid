"""
FastAPI integration for automatic tracing with Mermaid diagram generation.

This module provides seamless integration with FastAPI, automatically
instrumenting all endpoints and generating Mermaid diagrams for each request.

This follows standard OpenTelemetry behavior:
- Traces are automatically generated for every request
- No separate endpoints needed to retrieve traces
- Mermaid diagrams are exported automatically

Example:
    >>> from fastapi import FastAPI
    >>> from tracemaid.integrations import setup_fastapi_tracing
    >>>
    >>> app = FastAPI()
    >>>
    >>> # One-line setup for automatic tracing
    >>> setup_fastapi_tracing(
    ...     app,
    ...     service_name="my-api",
    ...     output_dir="./traces"
    ... )
    >>>
    >>> @app.get("/users/{user_id}")
    >>> async def get_user(user_id: str):
    ...     return {"id": user_id}
    >>>
    >>> # Every request to /users/{user_id} automatically generates a Mermaid diagram!
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from tracemaid.integrations.setup import setup_tracing, shutdown_tracing

logger = logging.getLogger(__name__)


def setup_fastapi_tracing(
    app,  # FastAPI app - type hint omitted to avoid import
    service_name: str,
    output_dir: Optional[str] = None,
    max_spans: int = 15,
    enable_styling: bool = True,
    include_metadata: bool = False,
    console_output: bool = True,
    on_diagram_generated: Optional[Callable[[str, str], None]] = None,
    excluded_urls: Optional[str] = None,
    flush_interval_seconds: float = 2.0,
    additional_exporters: Optional[list] = None,
) -> None:
    """Setup automatic tracing for a FastAPI application.

    This function configures OpenTelemetry auto-instrumentation for FastAPI
    with TracemaidExporter, so traces are automatically generated and
    Mermaid diagrams are produced for every request.

    Args:
        app: The FastAPI application instance.
        service_name: Name of the service (appears in Mermaid diagrams).
        output_dir: Directory to save Mermaid diagrams. None for console only.
        max_spans: Maximum spans to include in each diagram (default: 15).
        enable_styling: Whether to apply error/slow span styling.
        include_metadata: Whether to include duration/depth in node labels.
        console_output: Whether to print diagrams to console.
        on_diagram_generated: Optional callback(trace_id, diagram) when generated.
        excluded_urls: Regex pattern for URLs to exclude from tracing.
        flush_interval_seconds: Time to wait before considering a trace complete.

    Example:
        >>> from fastapi import FastAPI
        >>> from tracemaid.integrations import setup_fastapi_tracing
        >>>
        >>> app = FastAPI()
        >>> setup_fastapi_tracing(
        ...     app,
        ...     service_name="user-api",
        ...     output_dir="./traces",
        ...     excluded_urls="/health|/metrics"  # Don't trace health checks
        ... )
    """
    # Setup base tracing with TracemaidExporter
    setup_tracing(
        service_name=service_name,
        output_dir=output_dir,
        max_spans=max_spans,
        enable_styling=enable_styling,
        include_metadata=include_metadata,
        console_output=console_output,
        on_diagram_generated=on_diagram_generated,
        flush_interval_seconds=flush_interval_seconds,
        additional_exporters=additional_exporters,
    )

    # Try to import and apply FastAPI instrumentation
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        # Instrument the FastAPI app
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls=excluded_urls,
        )

        logger.info(
            "FastAPI auto-instrumentation enabled for service '%s'",
            service_name,
        )

    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-fastapi not installed. "
            "Install with: pip install tracemaid[fastapi]"
        )
        raise ImportError(
            "FastAPI instrumentation requires additional dependencies. "
            "Install with: pip install tracemaid[fastapi]"
        )

    # Register shutdown handler
    @app.on_event("shutdown")
    async def _shutdown_tracing():
        """Shutdown tracing on app shutdown."""
        shutdown_tracing()

    logger.info(
        "Tracemaid FastAPI integration complete. "
        "Mermaid diagrams will be generated automatically for each request."
    )


def instrument_requests() -> None:
    """Instrument outgoing HTTP requests made with the requests library.

    Call this to automatically trace HTTP calls made from your FastAPI
    service to external APIs.

    Example:
        >>> from tracemaid.integrations.fastapi_integration import instrument_requests
        >>> instrument_requests()
        >>>
        >>> # Now all requests.get(), requests.post(), etc. are traced
        >>> import requests
        >>> response = requests.get("https://api.example.com/users")
    """
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        logger.info("Requests library instrumentation enabled")
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-requests not installed. "
            "Install with: pip install tracemaid[fastapi]"
        )


def instrument_logging() -> None:
    """Instrument Python logging to include trace context.

    Call this to automatically inject trace_id and span_id into log records.

    Example:
        >>> from tracemaid.integrations.fastapi_integration import instrument_logging
        >>> instrument_logging()
        >>>
        >>> # Now logs include trace context
        >>> import logging
        >>> logging.info("User created")  # Includes trace_id, span_id
    """
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        LoggingInstrumentor().instrument()
        logger.info("Logging instrumentation enabled")
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-logging not installed. "
            "Install with: pip install tracemaid[fastapi]"
        )


def instrument_httpx() -> None:
    """Instrument outgoing HTTP requests made with the httpx library.

    Call this to automatically trace async HTTP calls made from your
    service to external APIs.

    Example:
        >>> from tracemaid.integrations.fastapi_integration import instrument_httpx
        >>> instrument_httpx()
        >>>
        >>> # Now all httpx.get(), httpx.post(), etc. are traced
        >>> import httpx
        >>> async with httpx.AsyncClient() as client:
        ...     response = await client.get("https://api.example.com/users")
    """
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX library instrumentation enabled")
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-httpx not installed. "
            "Install with: pip install tracemaid[instrumentation]"
        )


def instrument_sqlalchemy(engine=None) -> None:
    """Instrument SQLAlchemy database operations.

    Call this to automatically trace all database queries and operations.

    Args:
        engine: Optional SQLAlchemy engine to instrument. If None, instruments
                all engines created after this call.

    Example:
        >>> from tracemaid.integrations.fastapi_integration import instrument_sqlalchemy
        >>> from sqlalchemy import create_engine
        >>>
        >>> # Option 1: Instrument globally before creating engines
        >>> instrument_sqlalchemy()
        >>> engine = create_engine("sqlite:///app.db")
        >>>
        >>> # Option 2: Instrument a specific engine
        >>> engine = create_engine("postgresql://localhost/mydb")
        >>> instrument_sqlalchemy(engine=engine)
    """
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        if engine:
            SQLAlchemyInstrumentor().instrument(engine=engine)
        else:
            SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy instrumentation enabled")
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-sqlalchemy not installed. "
            "Install with: pip install tracemaid[instrumentation]"
        )


def instrument_redis(client=None) -> None:
    """Instrument Redis operations.

    Call this to automatically trace all Redis commands.

    Args:
        client: Optional Redis client instance to instrument. If None,
                instruments all Redis clients created after this call.

    Example:
        >>> from tracemaid.integrations.fastapi_integration import instrument_redis
        >>> import redis
        >>>
        >>> # Option 1: Instrument globally
        >>> instrument_redis()
        >>> client = redis.Redis()
        >>>
        >>> # Option 2: Instrument specific client
        >>> client = redis.Redis(host="localhost", port=6379)
        >>> instrument_redis(client=client)
    """
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        if client:
            RedisInstrumentor().instrument(client=client)
        else:
            RedisInstrumentor().instrument()
        logger.info("Redis instrumentation enabled")
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-redis not installed. "
            "Install with: pip install tracemaid[instrumentation]"
        )


def instrument_all(
    sqlalchemy_engine=None,
    redis_client=None,
) -> dict[str, bool]:
    """Instrument all available libraries at once.

    Convenience function to enable tracing for all supported libraries.
    Silently skips libraries that aren't installed.

    Args:
        sqlalchemy_engine: Optional SQLAlchemy engine to instrument.
        redis_client: Optional Redis client to instrument.

    Returns:
        Dict mapping library name to whether instrumentation succeeded.

    Example:
        >>> from tracemaid.integrations.fastapi_integration import instrument_all
        >>>
        >>> # Instrument everything available
        >>> results = instrument_all()
        >>> print(results)
        {'requests': True, 'httpx': True, 'sqlalchemy': False, 'redis': True, 'logging': True}
    """
    results = {}

    # Requests
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        RequestsInstrumentor().instrument()
        results["requests"] = True
        logger.info("Requests instrumentation enabled")
    except ImportError:
        results["requests"] = False

    # HTTPX
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        results["httpx"] = True
        logger.info("HTTPX instrumentation enabled")
    except ImportError:
        results["httpx"] = False

    # SQLAlchemy
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        if sqlalchemy_engine:
            SQLAlchemyInstrumentor().instrument(engine=sqlalchemy_engine)
        else:
            SQLAlchemyInstrumentor().instrument()
        results["sqlalchemy"] = True
        logger.info("SQLAlchemy instrumentation enabled")
    except ImportError:
        results["sqlalchemy"] = False

    # Redis
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        if redis_client:
            RedisInstrumentor().instrument(client=redis_client)
        else:
            RedisInstrumentor().instrument()
        results["redis"] = True
        logger.info("Redis instrumentation enabled")
    except ImportError:
        results["redis"] = False

    # Logging
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        LoggingInstrumentor().instrument()
        results["logging"] = True
        logger.info("Logging instrumentation enabled")
    except ImportError:
        results["logging"] = False

    enabled = [k for k, v in results.items() if v]
    logger.info("Auto-instrumentation complete: %s", ", ".join(enabled) or "none")

    return results

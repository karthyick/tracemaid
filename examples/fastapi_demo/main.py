"""
FastAPI Demo Application with Automatic Tracemaid Integration.

This module provides a FastAPI application that automatically generates
Mermaid diagrams for every request using OpenTelemetry tracing with
TracemaidExporter.

This follows standard OpenTelemetry behavior:
- Traces are automatically generated for every request
- No separate endpoints needed to retrieve traces
- Mermaid diagrams are exported automatically

Example:
    Run the application with uvicorn:

        $ uvicorn examples.fastapi_demo.main:app --reload

    Or run directly:

        $ python -m examples.fastapi_demo.main

    Every request automatically generates a Mermaid diagram!

Endpoints:
    GET /health - Health check endpoint
    GET /api/v1/users - List all users
    POST /api/v1/users - Create a new user
    GET /api/v1/users/{user_id} - Get user by ID
    DELETE /api/v1/users/{user_id} - Delete user by ID
    GET /api/v1/orders - List all orders
    POST /api/v1/orders - Create a new order
    GET /api/v1/orders/{order_id} - Get order by ID
    POST /api/v1/orders/{order_id}/process - Process an order
    POST /api/v1/orders/{order_id}/cancel - Cancel an order
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

# Configure logging FIRST
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTANT: Initialize tracing BEFORE importing FastAPI
# OpenTelemetry global instrumentation must happen before FastAPI class is loaded
# ============================================================================

# Application metadata (needed for tracing config)
APP_NAME = "tracemaid-fastapi-demo"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "FastAPI demo with automatic Mermaid diagram generation"
TRACES_OUTPUT_DIR = "./traces"

def _init_tracing() -> bool:
    """Initialize tracing at module load time (BEFORE FastAPI import)."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from tracemaid.exporters import TracemaidExporter
        from tracemaid.integrations import instrument_all

        # Create resource with service name
        resource = Resource.create({SERVICE_NAME: APP_NAME})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Create TracemaidExporter
        tracemaid_exporter = TracemaidExporter(
            output_dir=TRACES_OUTPUT_DIR,
            max_spans=10,  # Select up to 10 most important spans
            enable_styling=True,
            include_metadata=True,
            console_output=True,
        )

        # Use BatchSpanProcessor (standard OTel behavior)
        provider.add_span_processor(BatchSpanProcessor(tracemaid_exporter))

        # Set as global provider
        trace.set_tracer_provider(provider)

        # Instrument FastAPI BEFORE FastAPI class is imported
        FastAPIInstrumentor().instrument(
            excluded_urls="/health,/docs,/redoc,/openapi.json"
        )

        # Instrument all available libraries (requests, httpx, sqlalchemy, redis, logging)
        instrumented = instrument_all()
        enabled_libs = [k for k, v in instrumented.items() if v]
        if enabled_libs:
            logger.info("Auto-instrumented libraries: %s", ", ".join(enabled_libs))

        logger.info("Tracing initialized with FastAPI auto-instrumentation")
        return True
    except ImportError as e:
        logger.warning("Tracing not available: %s", e)
        return False

# Initialize tracing BEFORE importing FastAPI
_tracing_enabled = _init_tracing()

# ============================================================================
# Now we can safely import FastAPI (after instrumentation is set up)
# ============================================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from examples.fastapi_demo.routes import user_router, order_router

def setup_tracing(app: FastAPI) -> None:
    """Log tracing status (instrumentation done at module load).

    Args:
        app: The FastAPI application instance
    """
    if _tracing_enabled:
        logger.info(
            "Tracemaid tracing active - diagrams will be saved to: %s",
            TRACES_OUTPUT_DIR
        )
    else:
        logger.warning("Tracing not available")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles application startup and shutdown events.

    Args:
        app: The FastAPI application instance
    """
    # Startup
    logger.info(
        "Application starting: %s v%s",
        APP_NAME,
        APP_VERSION
    )

    # Setup tracing (must be done before app starts accepting requests)
    setup_tracing(app)

    logger.info("Application startup complete - ready to accept requests")

    yield  # Application runs here

    # Shutdown
    logger.info("Application shutting down: %s", APP_NAME)

    # Flush any pending traces
    try:
        from tracemaid.integrations.setup import shutdown_tracing
        shutdown_tracing()
    except ImportError:
        pass

    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with API prefix
app.include_router(user_router, prefix="/api/v1")
app.include_router(order_router, prefix="/api/v1")


@app.get("/health", tags=["monitoring"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint.

    Returns the current health status of the application including
    version information.

    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tracemaid": {
            "enabled": True,
            "output_dir": TRACES_OUTPUT_DIR,
            "description": "Mermaid diagrams are generated automatically for each request"
        }
    }


@app.get("/", tags=["root"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information.

    Returns:
        API information dictionary
    """
    return {
        "message": f"Welcome to {APP_NAME}",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "tracing": "Automatic - Mermaid diagrams exported to ./traces/"
    }


# Main entry point for running directly
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting application via __main__")
    uvicorn.run(
        "examples.fastapi_demo.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

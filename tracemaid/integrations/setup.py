"""
OpenTelemetry tracing setup with TracemaidExporter.

This module provides functions to configure OpenTelemetry tracing with
automatic Mermaid diagram generation via TracemaidExporter.

Example:
    >>> from tracemaid.integrations import setup_tracing, get_tracer
    >>>
    >>> # Setup tracing
    >>> setup_tracing(
    ...     service_name="my-service",
    ...     output_dir="./traces",
    ...     console_output=True
    ... )
    >>>
    >>> # Get a tracer and create spans
    >>> tracer = get_tracer("my-module")
    >>> with tracer.start_as_current_span("my-operation") as span:
    ...     # Your code here
    ...     span.set_attribute("user.id", "123")
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    ConsoleSpanExporter,
)

from tracemaid.exporters import TracemaidExporter

logger = logging.getLogger(__name__)

# Global reference to the configured provider
_tracer_provider: Optional[TracerProvider] = None


def setup_tracing(
    service_name: str,
    output_dir: Optional[str] = None,
    max_spans: int = 15,
    enable_styling: bool = True,
    include_metadata: bool = False,
    console_output: bool = True,
    on_diagram_generated: Optional[Callable[[str, str], None]] = None,
    use_batch_processor: bool = True,
    additional_exporters: Optional[list[SpanExporter]] = None,
    flush_interval_seconds: float = 2.0,
) -> TracerProvider:
    """Setup OpenTelemetry tracing with TracemaidExporter.

    This function configures the global TracerProvider with TracemaidExporter
    for automatic Mermaid diagram generation from traces.

    Args:
        service_name: Name of the service (appears in diagrams).
        output_dir: Directory to save Mermaid diagrams. None for console only.
        max_spans: Maximum spans to include in diagram (default: 15).
        enable_styling: Whether to apply error/slow span styling.
        include_metadata: Whether to include duration/depth in labels.
        console_output: Whether to print diagrams to console.
        on_diagram_generated: Optional callback when diagram is generated.
        use_batch_processor: Use BatchSpanProcessor (True) or SimpleSpanProcessor.
        additional_exporters: Additional SpanExporters to use alongside Tracemaid.
        flush_interval_seconds: Time to wait before considering a trace complete.

    Returns:
        The configured TracerProvider.

    Example:
        >>> provider = setup_tracing(
        ...     service_name="user-api",
        ...     output_dir="./traces",
        ...     max_spans=20
        ... )
    """
    global _tracer_provider

    # Create resource with service name
    resource = Resource.create({
        SERVICE_NAME: service_name,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Create TracemaidExporter
    tracemaid_exporter = TracemaidExporter(
        output_dir=output_dir,
        max_spans=max_spans,
        enable_styling=enable_styling,
        include_metadata=include_metadata,
        console_output=console_output,
        on_diagram_generated=on_diagram_generated,
        flush_interval_seconds=flush_interval_seconds,
    )

    # Add span processor
    if use_batch_processor:
        processor = BatchSpanProcessor(tracemaid_exporter)
    else:
        processor = SimpleSpanProcessor(tracemaid_exporter)

    provider.add_span_processor(processor)

    # Add any additional exporters
    if additional_exporters:
        for exporter in additional_exporters:
            if use_batch_processor:
                provider.add_span_processor(BatchSpanProcessor(exporter))
            else:
                provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Set as global provider
    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    logger.info(
        "Tracing configured: service=%s, output_dir=%s, max_spans=%d",
        service_name,
        output_dir,
        max_spans,
    )

    return provider


def get_tracer(name: str, version: Optional[str] = None) -> trace.Tracer:
    """Get a tracer instance.

    This is a convenience function to get a tracer from the configured provider.

    Args:
        name: Name of the tracer (usually module name).
        version: Optional version of the tracer.

    Returns:
        A Tracer instance for creating spans.

    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my-operation"):
        ...     # Your code here
        ...     pass
    """
    provider = trace.get_tracer_provider()
    return provider.get_tracer(name, version)


def shutdown_tracing() -> None:
    """Shutdown the tracing system.

    Flushes all pending spans and shuts down the TracerProvider.
    Call this on application shutdown to ensure all traces are exported.
    """
    global _tracer_provider

    if _tracer_provider:
        logger.info("Shutting down tracing...")
        _tracer_provider.shutdown()
        _tracer_provider = None
        logger.info("Tracing shutdown complete")

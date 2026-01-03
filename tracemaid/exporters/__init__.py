"""
tracemaid.exporters - OpenTelemetry SpanExporter implementations.

This subpackage provides custom SpanExporter implementations that automatically
convert OpenTelemetry traces to Mermaid diagrams.

Example:
    >>> from opentelemetry import trace
    >>> from opentelemetry.sdk.trace import TracerProvider
    >>> from opentelemetry.sdk.trace.export import BatchSpanProcessor
    >>> from tracemaid.exporters import TracemaidExporter
    >>>
    >>> # Configure the exporter
    >>> exporter = TracemaidExporter(output_dir="./traces")
    >>> provider = TracerProvider()
    >>> provider.add_span_processor(BatchSpanProcessor(exporter))
    >>> trace.set_tracer_provider(provider)
    >>>
    >>> # Now all spans are automatically exported as Mermaid diagrams!
"""

from tracemaid.exporters.mermaid_exporter import TracemaidExporter

__all__ = ["TracemaidExporter"]

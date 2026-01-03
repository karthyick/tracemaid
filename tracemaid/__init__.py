"""
tracemaid - OpenTelemetry trace analysis and Mermaid diagram generation.

This package provides tools for parsing OpenTelemetry traces, extracting
feature vectors, selecting important spans using clustering algorithms,
and generating Mermaid diagrams for visualization.

Standard Usage (Automatic - Recommended):
    Traces are automatically generated and exported following standard
    OpenTelemetry behavior. No separate endpoints needed!

    >>> from fastapi import FastAPI
    >>> from tracemaid.integrations import setup_fastapi_tracing
    >>>
    >>> app = FastAPI()
    >>> setup_fastapi_tracing(app, service_name="my-api", output_dir="./traces")
    >>>
    >>> # Every request now automatically generates a Mermaid diagram!

Manual Usage (for parsing existing trace files):
    >>> from tracemaid import OTelParser, SpanSelector, MermaidGenerator
    >>> parser = OTelParser()
    >>> trace = parser.parse_json(json_str)
    >>> selector = SpanSelector()
    >>> important = selector.select_from_trace(trace, max_spans=10)
    >>> generator = MermaidGenerator()
    >>> diagram = generator.generate(important, trace)
"""

__version__ = "0.1.3"
__author__ = "KR"
__email__ = "Karthickrajam18@gmail.com"

# Core components for manual trace analysis
from tracemaid.core.parser import Span, Trace, OTelParser
from tracemaid.core.features import FeatureExtractor, FEATURE_NAMES
from tracemaid.core.selector import SpanSelector
from tracemaid.core.mermaid import MermaidGenerator, MermaidStyle

# Exporter for automatic trace-to-mermaid conversion
from tracemaid.exporters import TracemaidExporter

__all__ = [
    # Core components
    "Span",
    "Trace",
    "OTelParser",
    "FeatureExtractor",
    "FEATURE_NAMES",
    "SpanSelector",
    "MermaidGenerator",
    "MermaidStyle",
    # Exporter
    "TracemaidExporter",
]

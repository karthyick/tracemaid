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
    >>> from tracemaid import OTelParser, MermaidGenerator
    >>> parser = OTelParser()
    >>> trace = parser.parse_json(json_str)
    >>> generator = MermaidGenerator()
    >>> diagram = generator.generate(important, trace)
"""

__version__ = "0.1.3"
__author__ = "KR"
__email__ = "Karthickrajam18@gmail.com"

# Core components for manual trace analysis
from tracemaid.core.parser import Span, Trace, OTelParser
from tracemaid.core.mermaid import MermaidGenerator, MermaidStyle
from tracemaid.core.plantuml import PlantUMLGenerator, PlantUMLStyle

__all__ = [
    # Core components
    "Span",
    "Trace",
    "OTelParser",
    "MermaidGenerator",
    "MermaidStyle",
    "PlantUMLGenerator",
    "PlantUMLStyle",
]

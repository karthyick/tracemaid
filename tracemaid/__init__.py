"""
tracemaid - OpenTelemetry trace analysis and Mermaid diagram generation.

This package provides tools for parsing OpenTelemetry traces, extracting
feature vectors, selecting important spans using clustering algorithms,
and generating Mermaid diagrams for visualization.

Example:
    >>> from tracemaid import OTelParser, SpanSelector, MermaidGenerator
    >>> parser = OTelParser()
    >>> trace = parser.parse_json(json_str)
    >>> selector = SpanSelector()
    >>> important = selector.select_from_trace(trace, max_spans=10)
    >>> generator = MermaidGenerator()
    >>> diagram = generator.generate(important, trace)
"""

__version__ = "0.1.0"
__author__ = "KR"
__email__ = "Karthickrajam18@gmail.com"

from tracemaid.core.parser import Span, Trace, OTelParser
from tracemaid.core.features import FeatureExtractor, FEATURE_NAMES
from tracemaid.core.selector import SpanSelector
from tracemaid.core.mermaid import MermaidGenerator, MermaidStyle

__all__ = [
    "Span",
    "Trace",
    "OTelParser",
    "FeatureExtractor",
    "FEATURE_NAMES",
    "SpanSelector",
    "MermaidGenerator",
    "MermaidStyle",
]

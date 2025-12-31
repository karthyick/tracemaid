"""
tracemaid.core - Core modules for trace parsing, feature extraction, and diagram generation.

This subpackage contains the main functionality:
- parser: Span and Trace dataclasses, OTelParser for parsing OpenTelemetry traces
- features: FeatureExtractor for 6D feature vector extraction
- selector: SpanSelector using ConvexHull and K-Means clustering
- mermaid: MermaidGenerator for diagram generation
"""

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

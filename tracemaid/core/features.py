"""
tracemaid.core.features - Feature extraction for OpenTelemetry spans.

This module extracts 6-dimensional feature vectors from trace spans for use
in clustering and importance analysis. Each span is converted to a vector
that captures its structural and behavioral characteristics.

6D Feature Vector Structure:
===========================

Feature 0 - duration_normalized (0.0 to 1.0):
    The span's duration relative to the total trace duration.
    Calculated as: span.duration / trace.total_duration
    - 0.0 = instant spans
    - 1.0 = span takes full trace duration

Feature 1 - depth_normalized (0.0 to 1.0):
    The span's nesting depth relative to the maximum depth in the trace.
    Calculated as: span.depth / max_depth_in_trace
    - 0.0 = root spans (depth 0)
    - 1.0 = deepest nested spans

Feature 2 - child_count (0.0 to 1.0):
    The number of direct children relative to the maximum child count.
    Calculated as: len(span.children) / max_children_in_trace
    - 0.0 = leaf spans (no children)
    - 1.0 = span with most children

Feature 3 - is_error (0.0 or 1.0):
    Binary indicator of whether the span has an error status.
    - 0.0 = OK or UNSET status
    - 1.0 = ERROR status

Feature 4 - is_root (0.0 or 1.0):
    Binary indicator of whether the span is a root span.
    - 0.0 = has a parent span
    - 1.0 = root span (no parent)

Feature 5 - relative_start_time (0.0 to 1.0):
    The span's start time relative to the trace's total duration.
    Calculated based on the span's position in the execution sequence.
    - 0.0 = span starts at trace beginning
    - 1.0 = span starts at trace end

Classes:
    FeatureExtractor: Extracts 6D feature vectors from Trace objects
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from tracemaid.core.parser import Span, Trace


# Feature names constant defining the 6 dimensions
FEATURE_NAMES: Tuple[str, ...] = (
    "duration_normalized",
    "depth_normalized",
    "child_count",
    "is_error",
    "is_root",
    "relative_start_time",
)


class FeatureExtractor:
    """Extracts 6-dimensional feature vectors from trace spans.

    The feature extractor analyzes a trace and converts each span into a
    normalized 6D vector suitable for clustering algorithms. By default,
    all features are in their natural ranges (mostly [0.0, 1.0]).

    When normalize=True is passed to extract(), StandardScaler is applied
    to produce features with zero mean and unit variance.

    Attributes:
        FEATURE_NAMES: Tuple of feature dimension names
        scaler: StandardScaler instance (populated when normalize=True)

    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract(trace)
        >>> print(features.shape)  # (N, 6) where N is number of spans
        >>>
        >>> # With normalization (zero mean, unit variance)
        >>> normalized = extractor.extract(trace, normalize=True)
        >>> print(normalized.mean(axis=0))  # Close to [0, 0, 0, 0, 0, 0]
    """

    FEATURE_NAMES = FEATURE_NAMES

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        self._span_start_times: Dict[str, float] = {}
        self._scaler: Optional[StandardScaler] = None

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """Get the StandardScaler used for normalization.

        Returns:
            StandardScaler instance if normalize=True was used, None otherwise
        """
        return self._scaler

    def extract(self, trace: Trace, normalize: bool = False) -> np.ndarray:
        """Extract 6D feature vectors for all spans in a trace.

        Args:
            trace: Trace object containing spans to analyze
            normalize: If True, apply StandardScaler to normalize features
                      to zero mean and unit variance. Default False.

        Returns:
            numpy.ndarray of shape (N, 6) where N is the number of spans.
            Each row is a 6D feature vector.
            - Without normalization: values mostly in [0, 1] range
            - With normalization: zero mean, unit variance
            Returns empty array of shape (0, 6) for traces with no spans.
        """
        if not trace.spans:
            return np.zeros((0, 6), dtype=np.float64)

        # Pre-compute trace-level statistics
        max_depth = self._get_max_depth(trace)
        max_children = self._get_max_children(trace)
        total_duration = max(trace.total_duration, 1)  # Avoid division by zero

        # Compute relative start times for all spans
        self._compute_relative_start_times(trace)

        # Extract features for each span
        n_spans = len(trace.spans)
        features = np.zeros((n_spans, 6), dtype=np.float64)

        for i, span in enumerate(trace.spans):
            features[i, 0] = self._normalize_duration(span, total_duration)
            features[i, 1] = self._normalize_depth(span, max_depth)
            features[i, 2] = self._normalize_child_count(span, max_children)
            features[i, 3] = self._get_error_flag(span)
            features[i, 4] = self._get_root_flag(span)
            features[i, 5] = self._get_relative_start_time(span)

        # Apply StandardScaler if requested
        if normalize and n_spans > 1:
            self._scaler = StandardScaler()
            features = self._scaler.fit_transform(features)
        elif normalize and n_spans == 1:
            # With only one sample, StandardScaler would produce all zeros
            # Store the scaler anyway for API consistency
            self._scaler = StandardScaler()
            # Fit on the single sample but return original features
            # (can't normalize a single sample meaningfully)
            self._scaler.fit(features)
        else:
            self._scaler = None

        return features

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform normalized features back to original scale.

        Only works if the previous extract() call used normalize=True.

        Args:
            features: Normalized feature array of shape (N, 6)

        Returns:
            Features in original scale

        Raises:
            ValueError: If no scaler is available (extract was not called
                       with normalize=True)
        """
        if self._scaler is None:
            raise ValueError(
                "No scaler available. Call extract() with normalize=True first."
            )
        return self._scaler.inverse_transform(features)

    def _get_max_depth(self, trace: Trace) -> int:
        """Get the maximum depth among all spans.

        Args:
            trace: Trace to analyze

        Returns:
            Maximum depth, minimum 1 to avoid division by zero for flat traces
        """
        if not trace.spans:
            return 1
        max_d = max(span.depth for span in trace.spans)
        return max(max_d, 1)  # Avoid division by zero for flat traces

    def _get_max_children(self, trace: Trace) -> int:
        """Get the maximum child count among all spans.

        Args:
            trace: Trace to analyze

        Returns:
            Maximum child count, minimum 1 to avoid division by zero
        """
        if not trace.spans:
            return 1
        max_c = max(len(span.children) for span in trace.spans)
        return max(max_c, 1)  # Avoid division by zero

    def _compute_relative_start_times(self, trace: Trace) -> None:
        """Compute relative start times for all spans.

        Since spans don't have explicit start times in our model,
        we estimate based on depth and position in the tree.
        Root spans start at 0, and child spans start after accounting
        for parent duration proportionally.

        Args:
            trace: Trace to analyze
        """
        self._span_start_times = {}

        if not trace.spans:
            return

        total_duration = max(trace.total_duration, 1)

        # Find root spans
        root_spans = [s for s in trace.spans if s.parentSpanId is None]

        # If no explicit roots, find spans whose parent isn't in the trace
        if not root_spans:
            span_ids = {s.spanId for s in trace.spans}
            root_spans = [
                s for s in trace.spans
                if s.parentSpanId not in span_ids
            ]

        # Process spans by depth level (BFS-style)
        # Root spans start at time 0
        for root in root_spans:
            self._assign_start_times_recursive(root, 0.0, total_duration)

    def _assign_start_times_recursive(
        self, span: Span, start_offset: float, total_duration: int
    ) -> None:
        """Recursively assign start times to span and its children.

        Args:
            span: Current span to process
            start_offset: Relative start time (0.0 to 1.0)
            total_duration: Total trace duration for normalization
        """
        # Record this span's start time
        self._span_start_times[span.spanId] = min(1.0, max(0.0, start_offset))

        if not span.children:
            return

        # Distribute children's start times across the span's duration
        # Each child starts after the previous one's estimated position
        span_duration_ratio = span.duration / total_duration if total_duration > 0 else 0

        n_children = len(span.children)
        for i, child in enumerate(span.children):
            # Child starts at parent's start plus a fraction of parent's duration
            child_start = start_offset + (i / max(n_children, 1)) * span_duration_ratio
            self._assign_start_times_recursive(child, child_start, total_duration)

    def _normalize_duration(self, span: Span, total_duration: int) -> float:
        """Normalize span duration relative to total trace duration.

        Args:
            span: Span to normalize
            total_duration: Total trace duration in microseconds

        Returns:
            Normalized duration value between 0.0 and 1.0
        """
        if total_duration <= 0:
            return 0.0
        return min(1.0, max(0.0, span.duration / total_duration))

    def _normalize_depth(self, span: Span, max_depth: int) -> float:
        """Normalize span depth to [0, 1] range.

        Args:
            span: Span to normalize
            max_depth: Maximum depth in the trace

        Returns:
            Normalized depth value between 0.0 and 1.0
        """
        if max_depth <= 0:
            return 0.0
        return min(1.0, max(0.0, span.depth / max_depth))

    def _normalize_child_count(self, span: Span, max_children: int) -> float:
        """Normalize child count to [0, 1] range.

        Args:
            span: Span to normalize
            max_children: Maximum child count in the trace

        Returns:
            Normalized child count value between 0.0 and 1.0
        """
        if max_children <= 0:
            return 0.0
        return min(1.0, max(0.0, len(span.children) / max_children))

    def _get_error_flag(self, span: Span) -> float:
        """Get error flag for span.

        Args:
            span: Span to check

        Returns:
            1.0 if span has ERROR status, 0.0 otherwise
        """
        return 1.0 if span.status.upper() == "ERROR" else 0.0

    def _get_root_flag(self, span: Span) -> float:
        """Get root flag for span.

        Args:
            span: Span to check

        Returns:
            1.0 if span is a root span (no parent), 0.0 otherwise
        """
        return 1.0 if span.parentSpanId is None else 0.0

    def _get_relative_start_time(self, span: Span) -> float:
        """Get relative start time for span.

        Args:
            span: Span to get start time for

        Returns:
            Relative start time between 0.0 and 1.0
        """
        return self._span_start_times.get(span.spanId, 0.0)

"""
tracemaid.core.selector - Span selection using ConvexHull and K-Means algorithms.

This module provides intelligent span selection for trace visualization.
It identifies the most "important" spans using geometric and clustering
algorithms applied to 6D feature vectors.

Selection Strategies:
====================

1. ConvexHull Selection:
   Finds spans on the boundary of the feature space. These represent
   extreme cases - the fastest, slowest, deepest, shallowest, etc.
   Useful for identifying outliers and edge cases.

2. K-Means Clustering:
   Groups spans into clusters and selects representatives from each.
   Useful for finding diverse, representative spans that cover the
   full range of behaviors in the trace.

3. Combined Selection:
   Merges hull points and cluster centers, deduplicating and limiting
   to a maximum number of spans. This provides both outliers and
   representative spans for comprehensive visualization.

Classes:
    SpanSelector: Main class for span selection with multiple algorithms

Example:
    >>> from tracemaid.core.selector import SpanSelector
    >>> from tracemaid.core.features import FeatureExtractor
    >>>
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract(trace)
    >>>
    >>> selector = SpanSelector()
    >>> important_indices = selector.select_important(features, max_spans=10)
    >>>
    >>> # Or use the convenience method
    >>> important_spans = selector.select_from_trace(trace, max_spans=10)
"""

from __future__ import annotations

from typing import List, Optional, Set, TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from tracemaid.core.parser import Span, Trace


class SpanSelector:
    """Selects important spans using geometric and clustering algorithms.

    This class provides multiple selection strategies for identifying
    the most important spans in a trace based on their 6D feature vectors.

    The selection algorithms work on normalized feature vectors and return
    indices into the original feature array. These indices correspond to
    positions in the trace.spans list.

    Attributes:
        min_points_for_hull: Minimum number of points needed for ConvexHull
        default_n_clusters: Default number of clusters for K-Means

    Example:
        >>> selector = SpanSelector()
        >>>
        >>> # Using individual methods
        >>> hull_indices = selector.select_hull_points(features)
        >>> cluster_indices = selector.select_cluster_centers(features, n_clusters=5)
        >>>
        >>> # Using combined selection
        >>> important = selector.select_important(features, max_spans=10)
        >>>
        >>> # Using trace context for complete workflow
        >>> spans = selector.select_from_trace(trace, max_spans=10)
    """

    # Minimum number of points needed for ConvexHull in N dimensions
    # In 6D, we need at least 7 points to form a valid simplex
    min_points_for_hull: int = 7

    # Default number of clusters when not specified
    default_n_clusters: int = 5

    def __init__(self) -> None:
        """Initialize the span selector."""
        self._last_features: Optional[np.ndarray] = None
        self._last_hull_indices: Optional[List[int]] = None
        self._last_cluster_indices: Optional[List[int]] = None

    def select_hull_points(self, features: np.ndarray) -> List[int]:
        """Select spans on the convex hull boundary of the feature space.

        Uses scipy.spatial.ConvexHull to find points that form the
        convex hull in the 6D feature space. These boundary points
        represent extreme cases - spans with the most unusual
        combinations of features.

        Args:
            features: numpy array of shape (N, 6) containing feature vectors

        Returns:
            List of indices (0 to N-1) of spans that lie on the convex hull.
            If fewer than min_points_for_hull samples, returns indices of
            all points. If hull computation fails, falls back to selecting
            points with extreme values in each dimension.

        Note:
            ConvexHull requires at least d+1 points in d dimensions.
            For 6D, this means at least 7 points are needed.

        Example:
            >>> features = extractor.extract(trace)
            >>> hull_indices = selector.select_hull_points(features)
            >>> hull_spans = [trace.spans[i] for i in hull_indices]
        """
        n_samples = features.shape[0]

        # Handle edge cases
        if n_samples == 0:
            return []

        if n_samples < self.min_points_for_hull:
            # Not enough points for hull, return all indices
            return list(range(n_samples))

        try:
            # Attempt to compute convex hull
            hull = ConvexHull(features)
            # hull.vertices contains unique vertex indices
            hull_indices = list(set(hull.vertices))
            self._last_hull_indices = hull_indices
            return hull_indices

        except Exception:
            # Hull computation can fail for various reasons:
            # - Degenerate geometry (all points nearly coplanar)
            # - Numerical precision issues
            # - QHull errors
            # Fall back to selecting extreme points in each dimension
            return self._select_extreme_points(features)

    def _select_extreme_points(self, features: np.ndarray) -> List[int]:
        """Fallback selection when ConvexHull fails.

        Selects points with minimum and maximum values in each feature
        dimension, providing a reasonable approximation of boundary points.

        Args:
            features: numpy array of shape (N, 6)

        Returns:
            List of unique indices for extreme points
        """
        n_samples, n_features = features.shape
        extreme_indices: Set[int] = set()

        for dim in range(n_features):
            # Add index of minimum value in this dimension
            min_idx = int(np.argmin(features[:, dim]))
            extreme_indices.add(min_idx)

            # Add index of maximum value in this dimension
            max_idx = int(np.argmax(features[:, dim]))
            extreme_indices.add(max_idx)

        return list(extreme_indices)

    def select_cluster_centers(
        self, features: np.ndarray, n_clusters: int = 5
    ) -> List[int]:
        """Select representative spans using K-Means clustering.

        Clusters the spans based on their feature vectors and selects
        the span closest to each cluster centroid. This ensures
        selection of diverse, representative spans.

        Args:
            features: numpy array of shape (N, 6) containing feature vectors
            n_clusters: Number of clusters (and thus representatives) to find.
                       Will be reduced to N if N < n_clusters.

        Returns:
            List of indices (0 to N-1) of spans closest to cluster centroids.
            Returns exactly min(n_clusters, N) indices.

        Note:
            Uses K-Means with multiple initializations (n_init=10) for
            stability. The random_state is not fixed to allow different
            selections on repeated calls if desired.

        Example:
            >>> features = extractor.extract(trace)
            >>> center_indices = selector.select_cluster_centers(features, n_clusters=8)
            >>> representative_spans = [trace.spans[i] for i in center_indices]
        """
        n_samples = features.shape[0]

        # Handle edge cases
        if n_samples == 0:
            return []

        if n_samples == 1:
            return [0]

        # Adjust n_clusters if we have fewer samples
        actual_n_clusters = min(n_clusters, n_samples)

        if actual_n_clusters == n_samples:
            # Every point is its own cluster, return all
            return list(range(n_samples))

        # Perform K-Means clustering
        kmeans = KMeans(
            n_clusters=actual_n_clusters,
            n_init=10,  # Multiple initializations for stability
            random_state=42,  # Fixed seed for reproducibility
            max_iter=300,
        )
        kmeans.fit(features)

        # Get cluster centroids
        centroids = kmeans.cluster_centers_

        # Find the span closest to each centroid
        # Compute distances from each point to each centroid
        distances = cdist(features, centroids, metric="euclidean")

        # For each centroid (column), find the closest point (row)
        closest_indices: List[int] = []
        for cluster_idx in range(actual_n_clusters):
            # Get distances to this centroid
            cluster_distances = distances[:, cluster_idx]
            # Find index of minimum distance
            closest_idx = int(np.argmin(cluster_distances))
            closest_indices.append(closest_idx)

        # Remove duplicates while preserving order
        seen: Set[int] = set()
        unique_indices: List[int] = []
        for idx in closest_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        self._last_cluster_indices = unique_indices
        return unique_indices

    def select_important(
        self, features: np.ndarray, max_spans: int = 10
    ) -> List[int]:
        """Select important spans by combining hull and cluster methods.

        Combines ConvexHull boundary points with K-Means cluster
        representatives to get a comprehensive selection of important
        spans. The selection is deduplicated and limited to max_spans.

        Priority Order:
        1. ConvexHull boundary points (outliers/extremes)
        2. K-Means cluster centers (representatives)

        Args:
            features: numpy array of shape (N, 6) containing feature vectors
            max_spans: Maximum number of spans to select

        Returns:
            List of unique indices, at most max_spans elements.
            Hull points are prioritized, then cluster centers fill remaining.

        Example:
            >>> features = extractor.extract(trace)
            >>> important = selector.select_important(features, max_spans=15)
            >>> print(f"Selected {len(important)} important spans")
        """
        n_samples = features.shape[0]

        # Handle edge cases
        if n_samples == 0:
            return []

        if n_samples <= max_spans:
            # All spans fit within the limit
            return list(range(n_samples))

        # Get hull points first (outliers/extremes)
        hull_indices = self.select_hull_points(features)

        # Determine how many cluster representatives we need
        # Use at least as many clusters as remaining slots
        n_clusters = max(self.default_n_clusters, max_spans - len(hull_indices))
        cluster_indices = self.select_cluster_centers(features, n_clusters=n_clusters)

        # Combine: hull points first, then cluster centers
        combined: List[int] = []
        seen: Set[int] = set()

        # Add hull points first (prioritized)
        for idx in hull_indices:
            if idx not in seen:
                seen.add(idx)
                combined.append(idx)
                if len(combined) >= max_spans:
                    return combined

        # Add cluster centers to fill remaining slots
        for idx in cluster_indices:
            if idx not in seen:
                seen.add(idx)
                combined.append(idx)
                if len(combined) >= max_spans:
                    return combined

        return combined

    def select_from_trace(
        self, trace: "Trace", max_spans: int = 10
    ) -> List["Span"]:
        """Select important spans directly from a Trace object.

        Convenience method that handles feature extraction and span
        lookup internally. This is the recommended high-level API.

        Args:
            trace: Trace object containing spans to analyze
            max_spans: Maximum number of spans to select

        Returns:
            List of Span objects identified as important.
            Returns up to max_spans spans, or all spans if trace is smaller.

        Example:
            >>> selector = SpanSelector()
            >>> important_spans = selector.select_from_trace(trace, max_spans=10)
            >>> for span in important_spans:
            ...     print(f"{span.service}: {span.operation}")
        """
        # Import here to avoid circular imports
        from tracemaid.core.features import FeatureExtractor

        if not trace.spans:
            return []

        n_spans = len(trace.spans)
        if n_spans <= max_spans:
            # All spans fit within limit
            return list(trace.spans)

        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract(trace)

        # Store for potential debugging/inspection
        self._last_features = features

        # Select important indices
        important_indices = self.select_important(features, max_spans=max_spans)

        # Map indices to actual Span objects
        important_spans = [trace.spans[i] for i in important_indices]

        return important_spans

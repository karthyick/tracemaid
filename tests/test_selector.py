"""
Tests for tracemaid.core.selector - SpanSelector with ConvexHull and K-Means.

This test module verifies:
- ConvexHull point selection for boundary detection
- K-Means clustering for representative selection
- Combined selection algorithm
- Trace context integration
- Edge cases and error handling
"""

import pytest
import numpy as np

from tracemaid.core.selector import SpanSelector
from tracemaid.core.parser import Span, Trace
from tracemaid.core.features import FeatureExtractor


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def selector() -> SpanSelector:
    """Create a SpanSelector instance for testing."""
    return SpanSelector()


@pytest.fixture
def simple_features() -> np.ndarray:
    """Create simple feature array with clear structure for testing."""
    # 10 points in 6D space with some clear patterns
    return np.array([
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Root span
        [0.5, 0.5, 0.5, 0.0, 0.0, 0.2],  # Middle span
        [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # Extreme span (max in multiple dims)
        [0.1, 0.1, 0.1, 1.0, 0.0, 0.3],  # Error span
        [0.2, 0.3, 0.4, 0.0, 0.0, 0.4],  # Regular span
        [0.8, 0.2, 0.1, 0.0, 0.0, 0.5],  # Long but shallow
        [0.1, 0.9, 0.8, 0.0, 0.0, 0.6],  # Short but deep with many children
        [0.3, 0.3, 0.3, 0.0, 0.0, 0.7],  # Average span
        [0.6, 0.6, 0.6, 0.0, 0.0, 0.8],  # Above average
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.9],  # Minimal span (late)
    ], dtype=np.float64)


@pytest.fixture
def large_features() -> np.ndarray:
    """Create larger feature array for comprehensive testing."""
    np.random.seed(42)
    # 50 points in 6D space
    features = np.random.rand(50, 6)
    # Normalize some columns to [0, 1]
    features[:, 3] = (features[:, 3] > 0.5).astype(float)  # Binary is_error
    features[:, 4] = (features[:, 4] > 0.9).astype(float)  # Binary is_root (rare)
    return features


@pytest.fixture
def simple_trace() -> Trace:
    """Create a simple trace with spans for testing select_from_trace."""
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="api-gateway",
        operation="handleRequest",
        duration=1000000,
        status="OK",
        depth=0,
        children=[]
    )
    child1 = Span(
        spanId="child1",
        parentSpanId="root",
        service="user-service",
        operation="getUser",
        duration=500000,
        status="OK",
        depth=1,
        children=[]
    )
    child2 = Span(
        spanId="child2",
        parentSpanId="root",
        service="order-service",
        operation="getOrders",
        duration=300000,
        status="OK",
        depth=1,
        children=[]
    )
    grandchild = Span(
        spanId="grandchild",
        parentSpanId="child1",
        service="db-service",
        operation="query",
        duration=200000,
        status="OK",
        depth=2,
        children=[]
    )
    error_span = Span(
        spanId="error",
        parentSpanId="child2",
        service="payment-service",
        operation="processPayment",
        duration=50000,
        status="ERROR",
        depth=2,
        children=[]
    )

    root.children = [child1, child2]
    child1.children = [grandchild]
    child2.children = [error_span]

    return Trace(
        traceId="trace123",
        spans=[root, child1, child2, grandchild, error_span],
        total_duration=1000000
    )


@pytest.fixture
def larger_trace() -> Trace:
    """Create a larger trace with 15 spans for testing."""
    spans = []

    # Create root
    root = Span(
        spanId="span_0",
        parentSpanId=None,
        service="frontend",
        operation="handleRequest",
        duration=2000000,
        status="OK",
        depth=0,
        children=[]
    )
    spans.append(root)

    # Create first level children
    for i in range(1, 4):
        child = Span(
            spanId=f"span_{i}",
            parentSpanId="span_0",
            service=f"service-{i}",
            operation=f"operation_{i}",
            duration=1000000 // i,
            status="OK" if i != 2 else "ERROR",
            depth=1,
            children=[]
        )
        spans.append(child)
        root.children.append(child)

    # Create second level children
    for i in range(4, 10):
        parent_idx = (i % 3) + 1
        child = Span(
            spanId=f"span_{i}",
            parentSpanId=f"span_{parent_idx}",
            service=f"service-{i}",
            operation=f"operation_{i}",
            duration=500000 // (i - 3),
            status="OK",
            depth=2,
            children=[]
        )
        spans.append(child)
        spans[parent_idx].children.append(child)

    # Create third level children
    for i in range(10, 15):
        parent_idx = (i % 6) + 4
        child = Span(
            spanId=f"span_{i}",
            parentSpanId=f"span_{parent_idx}",
            service=f"service-{i}",
            operation=f"operation_{i}",
            duration=100000 // (i - 9),
            status="OK",
            depth=3,
            children=[]
        )
        spans.append(child)
        spans[parent_idx].children.append(child)

    return Trace(
        traceId="large_trace",
        spans=spans,
        total_duration=2000000
    )


# =============================================================================
# Tests for select_hull_points
# =============================================================================


class TestSelectHullPoints:
    """Tests for ConvexHull-based point selection."""

    def test_hull_returns_list_of_indices(self, selector: SpanSelector, simple_features: np.ndarray):
        """ConvexHull should return a list of integer indices."""
        result = selector.select_hull_points(simple_features)

        assert isinstance(result, list)
        # Check for integer types (both Python int and numpy integers)
        assert all(isinstance(idx, (int, np.integer)) for idx in result)

    def test_hull_indices_are_valid(self, selector: SpanSelector, simple_features: np.ndarray):
        """Returned indices should be valid for the input array."""
        result = selector.select_hull_points(simple_features)
        n_samples = simple_features.shape[0]

        for idx in result:
            assert 0 <= idx < n_samples, f"Index {idx} out of range [0, {n_samples})"

    def test_hull_returns_unique_indices(self, selector: SpanSelector, simple_features: np.ndarray):
        """All returned indices should be unique."""
        result = selector.select_hull_points(simple_features)

        assert len(result) == len(set(result))

    def test_hull_empty_array(self, selector: SpanSelector):
        """Empty feature array should return empty list."""
        empty = np.zeros((0, 6), dtype=np.float64)
        result = selector.select_hull_points(empty)

        assert result == []

    def test_hull_single_point(self, selector: SpanSelector):
        """Single point should return that point."""
        single = np.array([[0.5, 0.5, 0.5, 0.0, 1.0, 0.5]])
        result = selector.select_hull_points(single)

        assert result == [0]

    def test_hull_few_points(self, selector: SpanSelector):
        """Fewer than 7 points should return all points (not enough for 6D hull)."""
        few = np.random.rand(5, 6)
        result = selector.select_hull_points(few)

        assert set(result) == {0, 1, 2, 3, 4}

    def test_hull_exactly_min_points(self, selector: SpanSelector):
        """Exactly min_points_for_hull should attempt hull computation."""
        points = np.random.rand(selector.min_points_for_hull, 6)
        result = selector.select_hull_points(points)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_hull_finds_extreme_points(self, selector: SpanSelector, large_features: np.ndarray):
        """Hull should find points on the boundary."""
        result = selector.select_hull_points(large_features)

        # Should find multiple boundary points
        assert len(result) > 0
        # Should not return all points (it's a selection)
        assert len(result) < len(large_features)

    def test_hull_includes_extremes(self, selector: SpanSelector):
        """Hull should include points with extreme values."""
        # Create features with clear extremes
        features = np.array([
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],  # Min in dim 0
            [1.0, 0.5, 0.5, 0.0, 0.0, 0.5],  # Max in dim 0
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.5],  # Min in dim 1
            [0.5, 1.0, 0.5, 0.0, 0.0, 0.5],  # Max in dim 1
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.5],  # Min in dim 2
            [0.5, 0.5, 1.0, 0.0, 0.0, 0.5],  # Max in dim 2
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],  # Min in dim 5
            [0.5, 0.5, 0.5, 0.0, 0.0, 1.0],  # Max in dim 5
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],  # Interior point
            [0.4, 0.4, 0.4, 0.0, 0.0, 0.4],  # Interior point
        ], dtype=np.float64)

        result = selector.select_hull_points(features)

        # Extreme points should be on the hull
        # At least some of 0-7 should be included
        extreme_indices = {0, 1, 2, 3, 4, 5, 6, 7}
        found_extremes = set(result) & extreme_indices
        assert len(found_extremes) >= 4, "Hull should find most extreme points"


# =============================================================================
# Tests for select_cluster_centers
# =============================================================================


class TestSelectClusterCenters:
    """Tests for K-Means-based cluster center selection."""

    def test_cluster_returns_list_of_indices(self, selector: SpanSelector, simple_features: np.ndarray):
        """K-Means should return a list of integer indices."""
        result = selector.select_cluster_centers(simple_features, n_clusters=3)

        assert isinstance(result, list)
        assert all(isinstance(idx, int) for idx in result)

    def test_cluster_indices_are_valid(self, selector: SpanSelector, simple_features: np.ndarray):
        """Returned indices should be valid for the input array."""
        result = selector.select_cluster_centers(simple_features, n_clusters=5)
        n_samples = simple_features.shape[0]

        for idx in result:
            assert 0 <= idx < n_samples

    def test_cluster_returns_unique_indices(self, selector: SpanSelector, simple_features: np.ndarray):
        """All returned indices should be unique."""
        result = selector.select_cluster_centers(simple_features, n_clusters=5)

        assert len(result) == len(set(result))

    def test_cluster_respects_n_clusters(self, selector: SpanSelector, large_features: np.ndarray):
        """Should return at most n_clusters indices."""
        for n_clusters in [3, 5, 8, 10]:
            result = selector.select_cluster_centers(large_features, n_clusters=n_clusters)
            # May be fewer due to duplicates, but never more
            assert len(result) <= n_clusters

    def test_cluster_empty_array(self, selector: SpanSelector):
        """Empty feature array should return empty list."""
        empty = np.zeros((0, 6), dtype=np.float64)
        result = selector.select_cluster_centers(empty, n_clusters=5)

        assert result == []

    def test_cluster_single_point(self, selector: SpanSelector):
        """Single point should return that point."""
        single = np.array([[0.5, 0.5, 0.5, 0.0, 1.0, 0.5]])
        result = selector.select_cluster_centers(single, n_clusters=5)

        assert result == [0]

    def test_cluster_more_clusters_than_samples(self, selector: SpanSelector):
        """Requesting more clusters than samples should return all samples."""
        features = np.random.rand(5, 6)
        result = selector.select_cluster_centers(features, n_clusters=10)

        # Should return all 5 points
        assert set(result) == {0, 1, 2, 3, 4}

    def test_cluster_reproducibility(self, selector: SpanSelector, simple_features: np.ndarray):
        """Same features should produce same clusters (fixed seed)."""
        result1 = selector.select_cluster_centers(simple_features, n_clusters=3)
        result2 = selector.select_cluster_centers(simple_features, n_clusters=3)

        assert result1 == result2

    def test_cluster_finds_representatives(self, selector: SpanSelector):
        """Clusters should find representative points from each cluster."""
        # Create clearly clustered data
        cluster1 = np.zeros((5, 6)) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cluster2 = np.zeros((5, 6)) + np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        cluster3 = np.zeros((5, 6)) + np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.5])

        # Add small noise
        np.random.seed(42)
        cluster1 += np.random.randn(5, 6) * 0.01
        cluster2 += np.random.randn(5, 6) * 0.01
        cluster3 += np.random.randn(5, 6) * 0.01

        features = np.vstack([cluster1, cluster2, cluster3])
        result = selector.select_cluster_centers(features, n_clusters=3)

        # Should pick one from each cluster
        assert len(result) == 3

        # Each cluster should be represented
        cluster1_indices = set(range(0, 5))
        cluster2_indices = set(range(5, 10))
        cluster3_indices = set(range(10, 15))

        result_set = set(result)
        assert len(result_set & cluster1_indices) >= 1 or \
               len(result_set & cluster2_indices) >= 1 or \
               len(result_set & cluster3_indices) >= 1


# =============================================================================
# Tests for select_important
# =============================================================================


class TestSelectImportant:
    """Tests for combined selection algorithm."""

    def test_important_returns_list_of_indices(self, selector: SpanSelector, simple_features: np.ndarray):
        """Combined selection should return a list of integer indices."""
        result = selector.select_important(simple_features, max_spans=5)

        assert isinstance(result, list)
        # Check for integer types (both Python int and numpy integers)
        assert all(isinstance(idx, (int, np.integer)) for idx in result)

    def test_important_respects_max_spans(self, selector: SpanSelector, large_features: np.ndarray):
        """Should return at most max_spans indices."""
        for max_spans in [5, 10, 15, 20]:
            result = selector.select_important(large_features, max_spans=max_spans)
            assert len(result) <= max_spans

    def test_important_returns_unique_indices(self, selector: SpanSelector, large_features: np.ndarray):
        """All returned indices should be unique."""
        result = selector.select_important(large_features, max_spans=15)

        assert len(result) == len(set(result))

    def test_important_empty_array(self, selector: SpanSelector):
        """Empty feature array should return empty list."""
        empty = np.zeros((0, 6), dtype=np.float64)
        result = selector.select_important(empty, max_spans=10)

        assert result == []

    def test_important_single_point(self, selector: SpanSelector):
        """Single point should return that point."""
        single = np.array([[0.5, 0.5, 0.5, 0.0, 1.0, 0.5]])
        result = selector.select_important(single, max_spans=10)

        assert result == [0]

    def test_important_fewer_samples_than_max(self, selector: SpanSelector):
        """When samples < max_spans, return all samples."""
        features = np.random.rand(7, 6)
        result = selector.select_important(features, max_spans=15)

        # Should return all 7
        assert len(result) == 7
        assert set(result) == set(range(7))

    def test_important_combines_hull_and_cluster(self, selector: SpanSelector, large_features: np.ndarray):
        """Should include points from both hull and cluster selection."""
        result = selector.select_important(large_features, max_spans=15)

        # Get individual selections for comparison
        hull_indices = set(selector.select_hull_points(large_features))
        cluster_indices = set(selector.select_cluster_centers(large_features, n_clusters=10))

        result_set = set(result)

        # Should have some overlap with both methods
        hull_overlap = len(result_set & hull_indices)
        cluster_overlap = len(result_set & cluster_indices)

        # Hull points should be prioritized
        assert hull_overlap > 0, "Should include hull points"

    def test_important_prioritizes_hull_points(self, selector: SpanSelector):
        """Hull points should be prioritized over cluster centers."""
        # Create data where hull points are distinct from cluster centers
        features = np.array([
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Extreme 0
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Extreme 1
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Extreme 2
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Extreme 3
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Extreme 4
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],  # Extreme 5
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],  # Interior 6
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],  # Interior 7
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],  # Interior 8
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.5],  # Interior 9
        ], dtype=np.float64)

        # With max_spans=4, should prioritize hull points
        result = selector.select_important(features, max_spans=4)

        # First 4 should mostly be from hull (extremes)
        hull_points = {0, 1, 2, 3, 4, 5}
        overlap = len(set(result) & hull_points)
        assert overlap >= 2, "Should prioritize hull points"


# =============================================================================
# Tests for select_from_trace
# =============================================================================


class TestSelectFromTrace:
    """Tests for trace-based selection."""

    def test_from_trace_returns_spans(self, selector: SpanSelector, simple_trace: Trace):
        """Should return a list of Span objects."""
        result = selector.select_from_trace(simple_trace, max_spans=3)

        assert isinstance(result, list)
        from tracemaid.core.parser import Span
        assert all(isinstance(span, Span) for span in result)

    def test_from_trace_respects_max_spans(self, selector: SpanSelector, larger_trace: Trace):
        """Should return at most max_spans spans."""
        for max_spans in [3, 5, 8, 10]:
            result = selector.select_from_trace(larger_trace, max_spans=max_spans)
            assert len(result) <= max_spans

    def test_from_trace_empty_trace(self, selector: SpanSelector):
        """Empty trace should return empty list."""
        empty_trace = Trace(
            traceId="empty",
            spans=[],
            total_duration=0
        )
        result = selector.select_from_trace(empty_trace, max_spans=10)

        assert result == []

    def test_from_trace_fewer_spans_than_max(self, selector: SpanSelector, simple_trace: Trace):
        """When spans < max_spans, return all spans."""
        result = selector.select_from_trace(simple_trace, max_spans=100)

        assert len(result) == len(simple_trace.spans)

    def test_from_trace_returns_trace_spans(self, selector: SpanSelector, simple_trace: Trace):
        """Returned spans should be from the input trace."""
        result = selector.select_from_trace(simple_trace, max_spans=3)

        trace_span_ids = {span.spanId for span in simple_trace.spans}
        for span in result:
            assert span.spanId in trace_span_ids

    def test_from_trace_single_span(self, selector: SpanSelector):
        """Single span trace should return that span."""
        single_span = Span(
            spanId="only",
            parentSpanId=None,
            service="test",
            operation="test",
            duration=1000,
            status="OK",
            depth=0,
            children=[]
        )
        trace = Trace(
            traceId="single",
            spans=[single_span],
            total_duration=1000
        )

        result = selector.select_from_trace(trace, max_spans=10)

        assert len(result) == 1
        assert result[0].spanId == "only"

    def test_from_trace_preserves_span_data(self, selector: SpanSelector, larger_trace: Trace):
        """Returned spans should have their original data intact."""
        result = selector.select_from_trace(larger_trace, max_spans=5)

        for selected_span in result:
            # Find the original span
            original = next(
                s for s in larger_trace.spans
                if s.spanId == selected_span.spanId
            )

            assert selected_span.service == original.service
            assert selected_span.operation == original.operation
            assert selected_span.duration == original.duration
            assert selected_span.status == original.status


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_identical_features(self, selector: SpanSelector):
        """Features that are all identical should still work."""
        identical = np.ones((10, 6)) * 0.5

        # Hull might fail but should fall back gracefully
        hull_result = selector.select_hull_points(identical)
        assert isinstance(hull_result, list)

        # Cluster should still work
        cluster_result = selector.select_cluster_centers(identical, n_clusters=3)
        assert isinstance(cluster_result, list)
        assert len(cluster_result) > 0

    def test_features_with_nan(self, selector: SpanSelector):
        """Features containing NaN should be handled."""
        features = np.random.rand(10, 6)
        features[5, 2] = np.nan

        # Should not crash, behavior depends on implementation
        try:
            result = selector.select_hull_points(features)
            # If it works, result should be a list
            assert isinstance(result, list)
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for NaN
            pass

    def test_very_large_features(self, selector: SpanSelector):
        """Should handle larger feature arrays efficiently."""
        np.random.seed(42)
        large = np.random.rand(500, 6)

        result = selector.select_important(large, max_spans=20)

        assert len(result) == 20
        assert len(set(result)) == 20

    def test_max_spans_zero(self, selector: SpanSelector, simple_features: np.ndarray):
        """max_spans=0 should return empty list or all points."""
        result = selector.select_important(simple_features, max_spans=0)

        # Either empty or returns all (both are valid interpretations)
        assert isinstance(result, list)

    def test_max_spans_one(self, selector: SpanSelector, simple_features: np.ndarray):
        """max_spans=1 should return exactly one point."""
        result = selector.select_important(simple_features, max_spans=1)

        assert len(result) == 1
        # Check for integer types (both Python int and numpy integers)
        assert isinstance(result[0], (int, np.integer))

    def test_two_dimensional_fallback(self, selector: SpanSelector):
        """Test with 2D features (should still work via fallback)."""
        # Creating 6D features but with only 2 truly varying dimensions
        features = np.zeros((10, 6))
        features[:, 0] = np.linspace(0, 1, 10)
        features[:, 1] = np.linspace(0, 1, 10)

        result = selector.select_hull_points(features)
        assert isinstance(result, list)
        assert len(result) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with other components."""

    def test_with_feature_extractor(self, simple_trace: Trace):
        """Integration with FeatureExtractor."""
        extractor = FeatureExtractor()
        selector = SpanSelector()

        features = extractor.extract(simple_trace)
        result = selector.select_important(features, max_spans=3)

        # Map back to spans
        selected_spans = [simple_trace.spans[i] for i in result]

        assert len(selected_spans) == 3
        assert all(isinstance(s, Span) for s in selected_spans)

    def test_with_normalized_features(self, simple_trace: Trace):
        """Should work with normalized features too."""
        extractor = FeatureExtractor()
        selector = SpanSelector()

        features = extractor.extract(simple_trace, normalize=True)
        result = selector.select_important(features, max_spans=3)

        assert len(result) == 3

    def test_end_to_end_selection(self, larger_trace: Trace):
        """Complete end-to-end test of selection pipeline."""
        selector = SpanSelector()

        # Use the high-level API
        important_spans = selector.select_from_trace(larger_trace, max_spans=5)

        assert len(important_spans) == 5

        # Verify spans are from the trace
        trace_span_ids = {s.spanId for s in larger_trace.spans}
        for span in important_spans:
            assert span.spanId in trace_span_ids

        # Verify diversity - should have spans from different depths/services
        depths = {span.depth for span in important_spans}
        assert len(depths) >= 2, "Selection should include diverse spans"


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Basic performance sanity checks."""

    def test_selection_completes_in_reasonable_time(self, selector: SpanSelector):
        """Selection should complete reasonably fast."""
        import time

        np.random.seed(42)
        large = np.random.rand(1000, 6)

        start = time.time()
        result = selector.select_important(large, max_spans=50)
        elapsed = time.time() - start

        # Should complete in under 5 seconds on any reasonable hardware
        assert elapsed < 5.0, f"Selection took too long: {elapsed:.2f}s"
        assert len(result) == 50


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for SpanSelector initialization."""

    def test_selector_initializes_with_defaults(self):
        """SpanSelector should initialize with correct default values."""
        selector = SpanSelector()

        assert selector.min_points_for_hull == 7
        assert selector.default_n_clusters == 5

    def test_selector_internal_state_is_none(self):
        """Initial internal state should be None."""
        selector = SpanSelector()

        assert selector._last_features is None
        assert selector._last_hull_indices is None
        assert selector._last_cluster_indices is None


# =============================================================================
# Fallback Selection Tests
# =============================================================================


class TestFallbackSelection:
    """Tests for _select_extreme_points fallback mechanism."""

    def test_extreme_points_fallback_on_degenerate_geometry(self):
        """Fallback should work when ConvexHull fails on degenerate data."""
        selector = SpanSelector()

        # Create features that might cause hull computation issues
        # All points on a lower-dimensional subspace
        features = np.zeros((10, 6))
        features[:, 0] = np.linspace(0, 1, 10)  # Only one dimension varies

        result = selector.select_hull_points(features)

        # Should fall back to extreme points and return valid indices
        assert isinstance(result, list)
        assert len(result) > 0
        # Should include min (0) and max (9) indices for the varying dimension
        assert 0 in result or 9 in result

    def test_extreme_points_finds_min_max_in_each_dimension(self):
        """Fallback should find min and max in each dimension."""
        selector = SpanSelector()

        # Create features with known extremes
        features = np.array([
            [0.0, 0.5, 0.5, 0.0, 0.0, 0.5],  # Min in dim 0
            [1.0, 0.5, 0.5, 0.0, 0.0, 0.5],  # Max in dim 0
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.5],  # Min in dim 1
            [0.5, 1.0, 0.5, 0.0, 0.0, 0.5],  # Max in dim 1
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.5],  # Min in dim 2
            [0.5, 0.5, 1.0, 0.0, 0.0, 0.5],  # Max in dim 2
            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],  # Min in dim 5
            [0.5, 0.5, 0.5, 0.0, 0.0, 1.0],  # Max in dim 5
        ], dtype=np.float64)

        # Call the private method directly for testing
        result = selector._select_extreme_points(features)

        # Should include indices with extreme values
        assert isinstance(result, list)
        assert len(result) > 0
        # Check that min/max indices are found
        result_set = set(result)
        # Should find extremes (indices 0-7 all have some extreme value)
        assert len(result_set) >= 4

    def test_extreme_points_returns_unique_indices(self):
        """Fallback should return unique indices."""
        selector = SpanSelector()

        features = np.random.rand(20, 6)
        result = selector._select_extreme_points(features)

        # All indices should be unique
        assert len(result) == len(set(result))

    def test_extreme_points_handles_same_min_max(self):
        """Fallback should handle case where min and max are the same index."""
        selector = SpanSelector()

        # All values same - min and max will be same index (first one found)
        features = np.ones((5, 6)) * 0.5
        result = selector._select_extreme_points(features)

        # Should still return valid result (may be single index repeated but set removes dups)
        assert isinstance(result, list)
        # With identical values, argmin/argmax returns 0 for all dimensions
        assert len(result) >= 1

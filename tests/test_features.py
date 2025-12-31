"""
Tests for tracemaid.core.features module.

This module contains comprehensive tests for the FeatureExtractor class,
covering feature extraction, normalization, and edge cases.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from tracemaid.core.parser import Span, Trace
from tracemaid.core.features import FeatureExtractor, FEATURE_NAMES


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_trace() -> Trace:
    """Create a simple trace with 3 spans: root -> child1, child2."""
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="api",
        operation="handle_request",
        duration=1000,
        status="OK",
        depth=0,
        children=[],
    )
    child1 = Span(
        spanId="child1",
        parentSpanId="root",
        service="api",
        operation="process",
        duration=500,
        status="OK",
        depth=1,
        children=[],
    )
    child2 = Span(
        spanId="child2",
        parentSpanId="root",
        service="db",
        operation="query",
        duration=300,
        status="OK",
        depth=1,
        children=[],
    )
    root.children = [child1, child2]

    return Trace(
        traceId="trace1",
        spans=[root, child1, child2],
        total_duration=1000,
    )


@pytest.fixture
def deep_trace() -> Trace:
    """Create a trace with deep nesting (4 levels)."""
    level0 = Span(
        spanId="level0",
        parentSpanId=None,
        service="gateway",
        operation="entry",
        duration=1000,
        status="OK",
        depth=0,
        children=[],
    )
    level1 = Span(
        spanId="level1",
        parentSpanId="level0",
        service="api",
        operation="process",
        duration=800,
        status="OK",
        depth=1,
        children=[],
    )
    level2 = Span(
        spanId="level2",
        parentSpanId="level1",
        service="service",
        operation="compute",
        duration=600,
        status="OK",
        depth=2,
        children=[],
    )
    level3 = Span(
        spanId="level3",
        parentSpanId="level2",
        service="db",
        operation="query",
        duration=400,
        status="OK",
        depth=3,
        children=[],
    )

    level0.children = [level1]
    level1.children = [level2]
    level2.children = [level3]

    return Trace(
        traceId="deep_trace",
        spans=[level0, level1, level2, level3],
        total_duration=1000,
    )


@pytest.fixture
def trace_with_error() -> Trace:
    """Create a trace that includes an error span."""
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="api",
        operation="handle_request",
        duration=1000,
        status="OK",
        depth=0,
        children=[],
    )
    success_child = Span(
        spanId="success",
        parentSpanId="root",
        service="api",
        operation="process",
        duration=500,
        status="OK",
        depth=1,
        children=[],
    )
    error_child = Span(
        spanId="error",
        parentSpanId="root",
        service="db",
        operation="query",
        duration=300,
        status="ERROR",
        depth=1,
        children=[],
    )
    root.children = [success_child, error_child]

    return Trace(
        traceId="error_trace",
        spans=[root, success_child, error_child],
        total_duration=1000,
    )


@pytest.fixture
def single_span_trace() -> Trace:
    """Create a trace with only one span."""
    span = Span(
        spanId="single",
        parentSpanId=None,
        service="api",
        operation="health_check",
        duration=100,
        status="OK",
        depth=0,
        children=[],
    )
    return Trace(
        traceId="single_trace",
        spans=[span],
        total_duration=100,
    )


@pytest.fixture
def empty_trace() -> Trace:
    """Create an empty trace with no spans."""
    return Trace(
        traceId="empty",
        spans=[],
        total_duration=0,
    )


@pytest.fixture
def wide_trace() -> Trace:
    """Create a trace with one root and many children (wide tree)."""
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="api",
        operation="batch_process",
        duration=2000,
        status="OK",
        depth=0,
        children=[],
    )
    children = []
    for i in range(10):
        child = Span(
            spanId=f"child{i}",
            parentSpanId="root",
            service=f"worker{i}",
            operation="process_item",
            duration=100 + i * 10,
            status="OK" if i != 5 else "ERROR",
            depth=1,
            children=[],
        )
        children.append(child)

    root.children = children

    return Trace(
        traceId="wide_trace",
        spans=[root] + children,
        total_duration=2000,
    )


# ============================================================================
# Test FEATURE_NAMES constant
# ============================================================================


class TestFeatureNames:
    """Tests for FEATURE_NAMES constant."""

    def test_feature_names_is_tuple(self):
        """FEATURE_NAMES should be a tuple."""
        assert isinstance(FEATURE_NAMES, tuple)

    def test_feature_names_has_six_elements(self):
        """FEATURE_NAMES should have exactly 6 elements."""
        assert len(FEATURE_NAMES) == 6

    def test_feature_names_are_strings(self):
        """All feature names should be strings."""
        for name in FEATURE_NAMES:
            assert isinstance(name, str)

    def test_feature_names_values(self):
        """Feature names should match the specification."""
        expected = (
            "duration_normalized",
            "depth_normalized",
            "child_count",
            "is_error",
            "is_root",
            "relative_start_time",
        )
        assert FEATURE_NAMES == expected

    def test_feature_extractor_has_feature_names(self):
        """FeatureExtractor class should have FEATURE_NAMES attribute."""
        assert hasattr(FeatureExtractor, "FEATURE_NAMES")
        assert FeatureExtractor.FEATURE_NAMES == FEATURE_NAMES


# ============================================================================
# Test extract() method - Shape
# ============================================================================


class TestExtractShape:
    """Tests for the shape of extracted features."""

    def test_extract_shape_simple_trace(self, simple_trace):
        """Extract should return (N, 6) array for N spans."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        assert features.shape == (3, 6)

    def test_extract_shape_deep_trace(self, deep_trace):
        """Extract should handle deep traces correctly."""
        extractor = FeatureExtractor()
        features = extractor.extract(deep_trace)

        assert features.shape == (4, 6)

    def test_extract_shape_single_span(self, single_span_trace):
        """Extract should handle single span trace."""
        extractor = FeatureExtractor()
        features = extractor.extract(single_span_trace)

        assert features.shape == (1, 6)

    def test_extract_shape_empty_trace(self, empty_trace):
        """Extract should return (0, 6) for empty trace."""
        extractor = FeatureExtractor()
        features = extractor.extract(empty_trace)

        assert features.shape == (0, 6)

    def test_extract_shape_wide_trace(self, wide_trace):
        """Extract should handle wide traces correctly."""
        extractor = FeatureExtractor()
        features = extractor.extract(wide_trace)

        assert features.shape == (11, 6)  # root + 10 children

    def test_extract_returns_float64(self, simple_trace):
        """Extract should return float64 array."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        assert features.dtype == np.float64


# ============================================================================
# Test extract() method - Feature Ranges
# ============================================================================


class TestFeatureRanges:
    """Tests for feature value ranges."""

    def test_duration_normalized_range(self, simple_trace):
        """Duration normalized should be in [0, 1]."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        duration_col = features[:, 0]
        assert np.all(duration_col >= 0.0)
        assert np.all(duration_col <= 1.0)

    def test_depth_normalized_range(self, deep_trace):
        """Depth normalized should be in [0, 1]."""
        extractor = FeatureExtractor()
        features = extractor.extract(deep_trace)

        depth_col = features[:, 1]
        assert np.all(depth_col >= 0.0)
        assert np.all(depth_col <= 1.0)

    def test_child_count_range(self, wide_trace):
        """Child count should be in [0, 1]."""
        extractor = FeatureExtractor()
        features = extractor.extract(wide_trace)

        child_count_col = features[:, 2]
        assert np.all(child_count_col >= 0.0)
        assert np.all(child_count_col <= 1.0)

    def test_is_error_is_binary(self, trace_with_error):
        """is_error should be 0.0 or 1.0."""
        extractor = FeatureExtractor()
        features = extractor.extract(trace_with_error)

        error_col = features[:, 3]
        for val in error_col:
            assert val in (0.0, 1.0)

    def test_is_root_is_binary(self, simple_trace):
        """is_root should be 0.0 or 1.0."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        root_col = features[:, 4]
        for val in root_col:
            assert val in (0.0, 1.0)

    def test_relative_start_time_range(self, deep_trace):
        """relative_start_time should be in [0, 1]."""
        extractor = FeatureExtractor()
        features = extractor.extract(deep_trace)

        start_time_col = features[:, 5]
        assert np.all(start_time_col >= 0.0)
        assert np.all(start_time_col <= 1.0)


# ============================================================================
# Test extract() method - Feature Values
# ============================================================================


class TestFeatureValues:
    """Tests for specific feature values."""

    def test_duration_normalized_values(self, simple_trace):
        """Test duration normalization calculation."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        # Root: 1000/1000 = 1.0, child1: 500/1000 = 0.5, child2: 300/1000 = 0.3
        assert features[0, 0] == pytest.approx(1.0)
        assert features[1, 0] == pytest.approx(0.5)
        assert features[2, 0] == pytest.approx(0.3)

    def test_depth_normalized_values(self, deep_trace):
        """Test depth normalization calculation."""
        extractor = FeatureExtractor()
        features = extractor.extract(deep_trace)

        # Depths: 0, 1, 2, 3. Max depth = 3
        # Normalized: 0/3=0, 1/3=0.33, 2/3=0.67, 3/3=1
        assert features[0, 1] == pytest.approx(0.0)
        assert features[1, 1] == pytest.approx(1/3, rel=0.01)
        assert features[2, 1] == pytest.approx(2/3, rel=0.01)
        assert features[3, 1] == pytest.approx(1.0)

    def test_child_count_values(self, simple_trace):
        """Test child count normalization."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        # Root has 2 children (max), others have 0
        # Normalized: root=2/2=1.0, children=0/2=0
        assert features[0, 2] == pytest.approx(1.0)
        assert features[1, 2] == pytest.approx(0.0)
        assert features[2, 2] == pytest.approx(0.0)

    def test_error_flag_values(self, trace_with_error):
        """Test error flag values."""
        extractor = FeatureExtractor()
        features = extractor.extract(trace_with_error)

        # Order: root (OK), success (OK), error (ERROR)
        assert features[0, 3] == 0.0  # root - OK
        assert features[1, 3] == 0.0  # success - OK
        assert features[2, 3] == 1.0  # error - ERROR

    def test_root_flag_values(self, simple_trace):
        """Test root flag values."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        # First span is root, others are not
        assert features[0, 4] == 1.0  # root
        assert features[1, 4] == 0.0  # child1
        assert features[2, 4] == 0.0  # child2

    def test_relative_start_time_root_is_zero(self, simple_trace):
        """Root span should have relative_start_time of 0."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        assert features[0, 5] == pytest.approx(0.0)


# ============================================================================
# Test Normalization with StandardScaler
# ============================================================================


class TestNormalization:
    """Tests for StandardScaler normalization."""

    def test_normalize_true_changes_values(self, wide_trace):
        """Normalized features should have different values than raw."""
        extractor = FeatureExtractor()
        raw_features = extractor.extract(wide_trace, normalize=False)
        norm_features = extractor.extract(wide_trace, normalize=True)

        # Values should be different (in general)
        assert not np.allclose(raw_features, norm_features)

    def test_normalize_zero_mean(self, wide_trace):
        """Normalized features should have approximately zero mean."""
        extractor = FeatureExtractor()
        features = extractor.extract(wide_trace, normalize=True)

        column_means = features.mean(axis=0)
        assert_allclose(column_means, np.zeros(6), atol=1e-10)

    def test_normalize_unit_variance(self, wide_trace):
        """Normalized features should have approximately unit variance."""
        extractor = FeatureExtractor()
        features = extractor.extract(wide_trace, normalize=True)

        column_stds = features.std(axis=0)
        # Standard deviation should be ~1 for columns with variance
        # Binary columns may have different behavior
        for i, std in enumerate(column_stds):
            if std > 0:  # Skip constant columns
                assert std == pytest.approx(1.0, rel=0.1)

    def test_scaler_is_stored(self, simple_trace):
        """Scaler should be stored after normalization."""
        extractor = FeatureExtractor()
        assert extractor.scaler is None

        extractor.extract(simple_trace, normalize=True)
        assert extractor.scaler is not None

    def test_scaler_not_stored_without_normalize(self, simple_trace):
        """Scaler should not be stored when normalize=False."""
        extractor = FeatureExtractor()
        extractor.extract(simple_trace, normalize=False)

        assert extractor.scaler is None

    def test_inverse_transform_works(self, wide_trace):
        """Inverse transform should recover original features."""
        extractor = FeatureExtractor()
        raw_features = extractor.extract(wide_trace, normalize=False)
        norm_features = extractor.extract(wide_trace, normalize=True)

        recovered = extractor.inverse_transform(norm_features)
        assert_allclose(recovered, raw_features, rtol=1e-10)

    def test_inverse_transform_error_without_scaler(self, simple_trace):
        """Inverse transform should raise error without prior normalization."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace, normalize=False)

        with pytest.raises(ValueError, match="No scaler available"):
            extractor.inverse_transform(features)

    def test_normalize_single_span(self, single_span_trace):
        """Normalization with single span should handle edge case."""
        extractor = FeatureExtractor()
        # Should not raise error
        features = extractor.extract(single_span_trace, normalize=True)

        assert features.shape == (1, 6)
        # Scaler should still be stored
        assert extractor.scaler is not None


# ============================================================================
# Test Error Span Feature
# ============================================================================


class TestErrorSpanFeature:
    """Tests for error span feature extraction."""

    def test_error_span_has_flag_set(self, trace_with_error):
        """Error spans should have is_error=1.0."""
        extractor = FeatureExtractor()
        features = extractor.extract(trace_with_error)

        # Find the error span (index 2 in our fixture)
        error_span_features = features[2]
        assert error_span_features[3] == 1.0

    def test_ok_span_has_flag_unset(self, trace_with_error):
        """OK spans should have is_error=0.0."""
        extractor = FeatureExtractor()
        features = extractor.extract(trace_with_error)

        ok_span_features = features[0]  # root span
        assert ok_span_features[3] == 0.0

    def test_mixed_error_trace(self, wide_trace):
        """Trace with mixed statuses should have correct flags."""
        extractor = FeatureExtractor()
        features = extractor.extract(wide_trace)

        # child5 (index 6) has ERROR status
        assert features[0, 3] == 0.0   # root - OK
        assert features[6, 3] == 1.0   # child5 - ERROR

        # Count errors
        error_count = np.sum(features[:, 3])
        assert error_count == 1.0

    def test_case_insensitive_error(self):
        """Error detection should be case-insensitive."""
        span_upper = Span(
            spanId="upper",
            parentSpanId=None,
            service="test",
            operation="test",
            duration=100,
            status="ERROR",
            depth=0,
        )
        span_lower = Span(
            spanId="lower",
            parentSpanId=None,
            service="test",
            operation="test",
            duration=100,
            status="error",
            depth=0,
        )
        span_mixed = Span(
            spanId="mixed",
            parentSpanId=None,
            service="test",
            operation="test",
            duration=100,
            status="Error",
            depth=0,
        )

        trace = Trace(
            traceId="case_test",
            spans=[span_upper, span_lower, span_mixed],
            total_duration=100,
        )

        extractor = FeatureExtractor()
        features = extractor.extract(trace)

        # All should be detected as errors
        assert features[0, 3] == 1.0
        assert features[1, 3] == 1.0
        assert features[2, 3] == 1.0


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_same_duration(self):
        """Handle trace where all spans have same duration."""
        spans = [
            Span(
                spanId=f"span{i}",
                parentSpanId=None if i == 0 else "span0",
                service="svc",
                operation="op",
                duration=100,
                status="OK",
                depth=0 if i == 0 else 1,
            )
            for i in range(3)
        ]
        spans[0].children = spans[1:]
        trace = Trace(traceId="same_dur", spans=spans, total_duration=100)

        extractor = FeatureExtractor()
        features = extractor.extract(trace)

        # All should have same normalized duration
        assert np.all(features[:, 0] == features[0, 0])

    def test_zero_duration_spans(self):
        """Handle spans with zero duration."""
        span = Span(
            spanId="zero",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=0,
            status="OK",
            depth=0,
        )
        trace = Trace(traceId="zero_dur", spans=[span], total_duration=0)

        extractor = FeatureExtractor()
        features = extractor.extract(trace)

        # Should not crash, duration should be 0
        assert features[0, 0] == 0.0

    def test_flat_trace_depth(self):
        """Handle flat trace (all spans at depth 0)."""
        # Multiple root spans (parallel traces)
        spans = [
            Span(
                spanId=f"root{i}",
                parentSpanId=None,
                service="svc",
                operation="op",
                duration=100,
                status="OK",
                depth=0,
            )
            for i in range(5)
        ]
        trace = Trace(traceId="flat", spans=spans, total_duration=500)

        extractor = FeatureExtractor()
        features = extractor.extract(trace)

        # All depth normalized should be 0 (all at depth 0)
        assert np.all(features[:, 1] == 0.0)
        # All should be roots
        assert np.all(features[:, 4] == 1.0)

    def test_very_deep_trace(self):
        """Handle very deep trace (100 levels)."""
        spans = []
        for i in range(100):
            span = Span(
                spanId=f"level{i}",
                parentSpanId=None if i == 0 else f"level{i-1}",
                service="svc",
                operation=f"level_{i}",
                duration=100 - i,
                status="OK",
                depth=i,
            )
            spans.append(span)
            if i > 0:
                spans[i-1].children.append(span)

        trace = Trace(traceId="very_deep", spans=spans, total_duration=100)

        extractor = FeatureExtractor()
        features = extractor.extract(trace)

        assert features.shape == (100, 6)
        # Last span should have max depth (normalized to 1.0)
        assert features[99, 1] == pytest.approx(1.0)
        # First span should have depth 0
        assert features[0, 1] == pytest.approx(0.0)

    def test_all_error_spans(self):
        """Handle trace where all spans are errors."""
        spans = [
            Span(
                spanId=f"err{i}",
                parentSpanId=None if i == 0 else "err0",
                service="svc",
                operation="fail",
                duration=100,
                status="ERROR",
                depth=0 if i == 0 else 1,
            )
            for i in range(5)
        ]
        spans[0].children = spans[1:]
        trace = Trace(traceId="all_err", spans=spans, total_duration=100)

        extractor = FeatureExtractor()
        features = extractor.extract(trace)

        # All should have error flag set
        assert np.all(features[:, 3] == 1.0)


# ============================================================================
# Test Multiple Extractions
# ============================================================================


class TestMultipleExtractions:
    """Tests for multiple extraction calls."""

    def test_extract_multiple_traces(self, simple_trace, deep_trace):
        """Extractor should handle multiple traces correctly."""
        extractor = FeatureExtractor()

        features1 = extractor.extract(simple_trace)
        features2 = extractor.extract(deep_trace)

        assert features1.shape == (3, 6)
        assert features2.shape == (4, 6)

    def test_normalize_resets_scaler(self, simple_trace, deep_trace):
        """Each normalized extraction should create new scaler."""
        extractor = FeatureExtractor()

        extractor.extract(simple_trace, normalize=True)
        scaler1 = extractor.scaler

        extractor.extract(deep_trace, normalize=True)
        scaler2 = extractor.scaler

        # Should be different scaler instances
        assert scaler1 is not scaler2

    def test_extraction_is_deterministic(self, wide_trace):
        """Multiple extractions should produce same results."""
        extractor = FeatureExtractor()

        features1 = extractor.extract(wide_trace)
        features2 = extractor.extract(wide_trace)

        assert_array_almost_equal(features1, features2)


# ============================================================================
# Test Integration with Trace
# ============================================================================


class TestTraceIntegration:
    """Tests for integration with Trace and Span classes."""

    def test_feature_count_matches_span_count(self, wide_trace):
        """Number of feature rows should match span count."""
        extractor = FeatureExtractor()
        features = extractor.extract(wide_trace)

        assert features.shape[0] == len(wide_trace.spans)

    def test_features_indexed_by_span_order(self, simple_trace):
        """Features should be in same order as trace.spans."""
        extractor = FeatureExtractor()
        features = extractor.extract(simple_trace)

        # Root is first in trace.spans, so first feature row should be root
        # Root has is_root = 1.0
        assert features[0, 4] == 1.0

        # Others have is_root = 0.0
        assert features[1, 4] == 0.0
        assert features[2, 4] == 0.0

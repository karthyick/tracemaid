"""
Tests for the tracemaid.core.plantuml module.

This module provides comprehensive tests for PlantUMLGenerator including:
- Basic diagram generation
- Node generation with special characters
- Edge generation for parent-child relationships
- Styling support for error and slow spans
- Edge cases and error handling
"""

import pytest
from tracemaid.core.parser import Span, Trace
from tracemaid.core.plantuml import PlantUMLGenerator, PlantUMLStyle

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_span() -> Span:
    """Create a simple root span for testing."""
    return Span(
        spanId="span_001",
        parentSpanId=None,
        service="service-a",
        operation="get_user",
        duration=1000,
        status="OK",
        depth=0,
        children=[],
    )


@pytest.fixture
def child_span() -> Span:
    """Create a child span for testing."""
    return Span(
        spanId="span_002",
        parentSpanId="span_001",
        service="service-b",
        operation="query_db",
        duration=500,
        status="OK",
        depth=1,
        children=[],
    )


@pytest.fixture
def error_span() -> Span:
    """Create an error span for testing."""
    return Span(
        spanId="span_003",
        parentSpanId="span_001",
        service="service-c",
        operation="external_call",
        duration=2000,
        status="ERROR",
        depth=1,
        children=[],
    )


@pytest.fixture
def simple_trace(simple_span: Span, child_span: Span) -> Trace:
    """Create a simple trace with two spans."""
    simple_span.children = [child_span]
    return Trace(traceId="trace_001", spans=[simple_span, child_span], total_duration=1000)


@pytest.fixture
def trace_with_error(simple_span: Span, child_span: Span, error_span: Span) -> Trace:
    """Create a trace that includes an error span."""
    simple_span.children = [child_span, error_span]
    return Trace(
        traceId="trace_002", spans=[simple_span, child_span, error_span], total_duration=2000
    )


@pytest.fixture
def generator() -> PlantUMLGenerator:
    """Create a PlantUMLGenerator instance."""
    return PlantUMLGenerator()


@pytest.fixture
def multi_level_trace() -> Trace:
    """Create a multi-level trace for complex edge testing."""
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="gateway",
        operation="handle_request",
        duration=5000,
        status="OK",
        depth=0,
        children=[],
    )
    level1_a = Span(
        spanId="level1_a",
        parentSpanId="root",
        service="auth",
        operation="validate_token",
        duration=1000,
        status="OK",
        depth=1,
        children=[],
    )
    level1_b = Span(
        spanId="level1_b",
        parentSpanId="root",
        service="user-service",
        operation="get_profile",
        duration=2000,
        status="OK",
        depth=1,
        children=[],
    )
    level2 = Span(
        spanId="level2",
        parentSpanId="level1_b",
        service="database",
        operation="query",
        duration=500,
        status="OK",
        depth=2,
        children=[],
    )
    level3 = Span(
        spanId="level3",
        parentSpanId="level2",
        service="cache",
        operation="get",
        duration=100,
        status="OK",
        depth=3,
        children=[],
    )

    root.children = [level1_a, level1_b]
    level1_b.children = [level2]
    level2.children = [level3]

    return Trace(
        traceId="multi_level_trace",
        spans=[root, level1_a, level1_b, level2, level3],
        total_duration=5000,
    )


# =============================================================================
# Basic Generation Tests
# =============================================================================


class TestBasicGeneration:
    """Tests for basic PlantUML diagram generation."""

    def test_generate_empty_spans(self, generator: PlantUMLGenerator):
        """Test generation with empty span list."""
        trace = Trace(traceId="empty", spans=[], total_duration=0)
        result = generator.generate([], trace)

        assert "@startuml" in result
        assert "@enduml" in result
        assert "No spans to display" in result

    def test_generate_single_span(self, generator: PlantUMLGenerator, simple_span: Span):
        """Test generation with a single span."""
        trace = Trace(traceId="single", spans=[simple_span], total_duration=1000)
        result = generator.generate([simple_span], trace)

        assert "@startuml activity" in result
        assert "@enduml" in result
        assert ":service-a: get_user; as span_001" in result

    def test_generate_header(self, generator: PlantUMLGenerator, simple_span: Span):
        """Test that @startuml/@enduml headers are present."""
        trace = Trace(traceId="test", spans=[simple_span], total_duration=1000)
        result = generator.generate([simple_span], trace)

        lines = result.split("\n")
        assert lines[0] == "@startuml activity"
        assert lines[-1] == "@enduml"

    def test_generate_with_title(self, generator: PlantUMLGenerator, simple_span: Span):
        """Test generation with optional title."""
        trace = Trace(traceId="test", spans=[simple_span], total_duration=1000)
        result = generator.generate([simple_span], trace, title="My Trace")

        assert "title My Trace" in result


# =============================================================================
# Node Generation Tests
# =============================================================================


class TestNodeGeneration:
    """Tests for PlantUML node generation."""

    def test_node_format(self, generator: PlantUMLGenerator, simple_span: Span):
        """Test that nodes have correct format: :service: operation; as alias."""
        trace = Trace(traceId="test", spans=[simple_span], total_duration=1000)
        result = generator.generate([simple_span], trace, enable_styling=False)

        assert ":service-a: get_user; as span_001" in result

    def test_node_special_char_escaping(self, generator: PlantUMLGenerator):
        """Test that special characters are escaped in labels."""
        span = Span(
            spanId="special",
            parentSpanId=None,
            service="service<test>",
            operation=r"do[something]\"here\" and|more;",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=100)
        result = generator.generate([span], trace, enable_styling=False)

        # Expect backslash and other special characters to be escaped correctly
        assert r":service\<test\>: do\[something\]\\\"here\\\" and\|more\;; as special" in result

    def test_multiple_nodes(
        self, generator: PlantUMLGenerator, simple_span: Span, child_span: Span
    ):
        """Test generation of multiple nodes."""
        trace = Trace(traceId="test", spans=[simple_span, child_span], total_duration=1000)
        result = generator.generate([simple_span, child_span], trace, enable_styling=False)

        assert ":service-a: get_user; as span_001" in result
        assert ":service-b: query_db; as span_002" in result


# =============================================================================
# Edge Generation Tests
# =============================================================================


class TestEdgeGeneration:
    """Tests for PlantUML edge generation."""

    def test_edge_format(self, generator: PlantUMLGenerator, simple_span: Span, child_span: Span):
        """Test that edges have correct format: alias --> alias."""
        simple_span.children = [child_span]
        trace = Trace(traceId="test", spans=[simple_span, child_span], total_duration=1000)
        result = generator.generate([simple_span, child_span], trace, enable_styling=False)

        assert "span_001 --> span_002" in result

    def test_edges_only_for_selected_spans(
        self, generator: PlantUMLGenerator, multi_level_trace: Trace
    ):
        """Test that edges only connect spans in the selection."""
        # Select only root and level2 (skip level1_b)
        spans = multi_level_trace.spans
        selected = [spans[0], spans[3]]  # root and level2

        result = generator.generate(selected, multi_level_trace, enable_styling=False)

        # Should have relation for indirect relationship
        assert "root ..> level2" in result

    def test_no_edges_for_single_span(self, generator: PlantUMLGenerator, simple_span: Span):
        """Test that single span produces no edges."""
        trace = Trace(traceId="test", spans=[simple_span], total_duration=1000)
        result = generator.generate([simple_span], trace, enable_styling=False)

        assert "-->" not in result

    def test_multiple_edges(self, generator: PlantUMLGenerator):
        """Test generation of multiple edges."""
        root = Span(
            spanId="root",
            parentSpanId=None,
            service="gateway",
            operation="handle",
            duration=1000,
            status="OK",
            depth=0,
            children=[],
        )
        child1 = Span(
            spanId="child1",
            parentSpanId="root",
            service="svc1",
            operation="op1",
            duration=500,
            status="OK",
            depth=1,
            children=[],
        )
        child2 = Span(
            spanId="child2",
            parentSpanId="root",
            service="svc2",
            operation="op2",
            duration=500,
            status="OK",
            depth=1,
            children=[],
        )

        root.children = [child1, child2]
        trace = Trace(traceId="test", spans=[root, child1, child2], total_duration=1000)

        result = generator.generate([root, child1, child2], trace, enable_styling=False)

        assert "root --> child1" in result
        assert "root --> child2" in result

    def test_indirect_edge_when_parent_missing(
        self, generator: PlantUMLGenerator, multi_level_trace: Trace
    ):
        """Test edge when direct parent is not in selection."""
        spans = multi_level_trace.spans
        # Select root and level3 (skip level1_b and level2)
        selected = [spans[0], spans[4]]  # root and level3

        result = generator.generate(selected, multi_level_trace, enable_styling=False)

        # PlantUML activity diagram should still show a path
        assert "root ..> level3" in result


# =============================================================================
# Styling Tests
# =============================================================================


class TestStyling:
    """Tests for PlantUML styling features."""

    def test_error_span_styling(self, generator: PlantUMLGenerator, trace_with_error: Trace):
        """Test that error spans get error styling."""
        spans = trace_with_error.spans
        result = generator.generate(spans, trace_with_error, enable_styling=True)

        assert "skinparam activityspan_003BackgroundColor Red" in result

    def test_slow_span_styling(self, generator: PlantUMLGenerator):
        """Test that slow spans get slow styling."""
        fast_span = Span(
            spanId="fast",
            parentSpanId=None,
            service="svc",
            operation="fast_op",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        slow_span = Span(
            spanId="slow",
            parentSpanId="fast",
            service="svc",
            operation="slow_op",
            duration=10000,  # Much slower
            status="OK",
            depth=1,
            children=[],
        )
        fast_span.children = [slow_span]
        trace = Trace(traceId="test", spans=[fast_span, slow_span], total_duration=10000)

        result = generator.generate([fast_span, slow_span], trace, enable_styling=True)

        assert "skinparam activityslowBackgroundColor Orange" in result

    def test_styling_disabled(self, generator: PlantUMLGenerator, trace_with_error: Trace):
        """Test that styling can be disabled."""
        spans = trace_with_error.spans
        result = generator.generate(spans, trace_with_error, enable_styling=False)

        assert "skinparam" not in result

    def test_custom_style_colors(self):
        """Test custom style colors."""
        custom_style = PlantUMLStyle(error_color="DarkRed", slow_color="DarkOrange")
        generator = PlantUMLGenerator(style=custom_style)

        error_span = Span(
            spanId="error",
            parentSpanId=None,
            service="svc",
            operation="fail",
            duration=100,
            status="ERROR",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[error_span], total_duration=100)

        result = generator.generate([error_span], trace, enable_styling=True)

        assert "skinparam activityerrorBackgroundColor DarkRed" in result

    def test_normal_span_styling(self, generator: PlantUMLGenerator):
        """Test that normal spans get normal styling."""
        # Create spans with varying durations so we can identify a "normal" one
        fast_span = Span(
            spanId="fast",
            parentSpanId=None,
            service="svc",
            operation="fast_op",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        medium_span = Span(
            spanId="medium",
            parentSpanId="fast",
            service="svc",
            operation="medium_op",
            duration=500,  # Below 90th percentile
            status="OK",
            depth=1,
            children=[],
        )
        slow_span = Span(
            spanId="slow",
            parentSpanId="fast",
            service="svc",
            operation="slow_op",
            duration=10000,  # Will be marked as slow (90th percentile)
            status="OK",
            depth=1,
            children=[],
        )
        fast_span.children = [medium_span, slow_span]
        trace = Trace(
            traceId="test", spans=[fast_span, medium_span, slow_span], total_duration=10000
        )
        result = generator.generate([fast_span, medium_span, slow_span], trace, enable_styling=True)

        # The fast and medium spans should get normal styling
        assert "skinparam activitymediumBackgroundColor LightBlue" in result


# =============================================================================
# Generate with Metadata Tests
# =============================================================================


class TestGenerateWithMetadata:
    """Tests for generate_with_metadata method."""

    def test_include_duration(self, generator: PlantUMLGenerator, simple_span: Span):
        """Test including duration in labels."""
        trace = Trace(traceId="test", spans=[simple_span], total_duration=1000)
        result = generator.generate_with_metadata([simple_span], trace, include_duration=True)

        # Duration should be formatted (1000μs = 1ms)
        assert "(1.0ms)" in result

    def test_include_depth(self, generator: PlantUMLGenerator, child_span: Span):
        """Test including depth in labels."""
        trace = Trace(traceId="test", spans=[child_span], total_duration=500)
        result = generator.generate_with_metadata([child_span], trace, include_depth=True)

        assert r"[depth: 1]" in result

    def test_duration_formatting_microseconds(self, generator: PlantUMLGenerator):
        """Test duration formatting for microseconds."""
        span = Span(
            spanId="fast",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=500,  # 500μs
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=500)
        result = generator.generate_with_metadata([span], trace)

        assert "(500μs)" in result

    def test_duration_formatting_milliseconds(self, generator: PlantUMLGenerator):
        """Test duration formatting for milliseconds."""
        span = Span(
            spanId="medium",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=50000,  # 50ms
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=50000)
        result = generator.generate_with_metadata([span], trace)

        assert "(50.0ms)" in result

    def test_duration_formatting_seconds(self, generator: PlantUMLGenerator):
        """Test duration formatting for seconds."""
        span = Span(
            spanId="slow",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=2500000,  # 2.5s
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=2500000)
        result = generator.generate_with_metadata([span], trace)

        assert "(2.50s)" in result


# =============================================================================
# PlantUMLStyle Tests
# =============================================================================


class TestPlantUMLStyle:
    """Tests for PlantUMLStyle dataclass."""

    def test_default_values(self):
        """Test default style values."""
        style = PlantUMLStyle()

        assert style.error_color == "Red"
        assert style.slow_color == "Orange"
        assert style.normal_color == "LightBlue"
        assert style.slow_threshold_percentile == 90.0
        assert style.box_type == ""  # For activity diagram, no explicit box type needed for default
        assert style.diagram_type == "activity"

    def test_custom_values(self):
        """Test custom style values."""
        style = PlantUMLStyle(
            error_color="DarkRed", slow_threshold_percentile=75.0, box_type="component"
        )

        assert style.error_color == "DarkRed"
        assert style.slow_threshold_percentile == 75.0
        assert style.box_type == "component"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_span_id(self):
        """Test handling of empty service/operation."""
        span = Span(
            spanId="empty_labels",
            parentSpanId=None,
            service="",
            operation="",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=100)
        generator = PlantUMLGenerator()

        result = generator.generate([span], trace)

        assert ":: ; as empty_labels" in result  # empty service and operation is valid in PlantUML
        assert "@startuml" in result

    def test_very_long_label(self, generator: PlantUMLGenerator):
        """Test handling of very long labels."""
        long_operation = "x" * 200
        span = Span(
            spanId="long",
            parentSpanId=None,
            service="svc",
            operation=long_operation,
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=100)

        result = generator.generate([span], trace)

        # Should still generate valid diagram
        assert "@startuml" in result
        assert long_operation in result

    def test_unicode_characters(self, generator: PlantUMLGenerator):
        """Test handling of unicode characters."""
        span = Span(
            spanId="unicode",
            parentSpanId=None,
            service="服务",
            operation="操作",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=100)

        result = generator.generate([span], trace)

        assert "服务" in result
        assert "操作" in result

    def test_all_spans_have_same_duration(self, generator: PlantUMLGenerator):
        """Test styling when all spans have the same duration."""
        spans = [
            Span(
                spanId=f"span{i}",
                parentSpanId=None if i == 0 else "span0",
                service="svc",
                operation=f"op{i}",
                duration=1000,  # Same duration
                status="OK",
                depth=0 if i == 0 else 1,
                children=[],
            )
            for i in range(5)
        ]
        spans[0].children = spans[1:]
        trace = Trace(traceId="test", spans=spans, total_duration=1000)

        result = generator.generate(spans, trace, enable_styling=True)

        # Should not crash and should produce valid output
        assert "@startuml" in result

    def test_circular_reference_protection(self, generator: PlantUMLGenerator):
        """Test that circular references don't cause infinite loops."""
        span1 = Span(
            spanId="span1",
            parentSpanId="span2",  # Points to span2
            service="svc",
            operation="op1",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        span2 = Span(
            spanId="span2",
            parentSpanId="span1",  # Points back to span1
            service="svc",
            operation="op2",
            duration=100,
            status="OK",
            depth=1,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span1, span2], total_duration=100)

        # Should not hang or crash
        result = generator.generate([span1, span2], trace)

        assert "@startuml" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for PlantUMLGenerator."""

    def test_complete_diagram_structure(
        self, generator: PlantUMLGenerator, multi_level_trace: Trace
    ):
        """Test that complete diagram has all expected parts."""
        result = generator.generate(multi_level_trace.spans, multi_level_trace, enable_styling=True)

        lines = result.split("\n")

        # Should start with header
        assert lines[0] == "@startuml activity"

        # Should have nodes
        assert any(":gateway: handle_request; as root" in line for line in lines)
        assert any(":auth: validate_token; as level1_a" in line for line in lines)
        assert any(":user-service: get_profile; as level1_b" in line for line in lines)
        assert any(":database: query; as level2" in line for line in lines)
        assert any(":cache: get; as level3" in line for line in lines)

        # Should have edges
        assert any("root --> level1_a" in line for line in lines)

        # Should have styles for the root span which is slow
        assert any("skinparam activityrootBackgroundColor Orange" in line for line in lines)

    def test_diagram_is_valid_plantuml_syntax(
        self, generator: PlantUMLGenerator, simple_trace: Trace
    ):
        """Test that output follows PlantUML syntax rules."""
        result = generator.generate(simple_trace.spans, simple_trace, enable_styling=True)

        lines = result.split("\n")

        # First line must be @startuml activity
        assert lines[0] == "@startuml activity"

        # Last line must be @enduml
        assert lines[-1] == "@enduml"

    def test_reproducible_output(self, generator: PlantUMLGenerator, simple_trace: Trace):
        """Test that same input produces same output."""
        result1 = generator.generate(simple_trace.spans, simple_trace, enable_styling=True)
        result2 = generator.generate(simple_trace.spans, simple_trace, enable_styling=True)

        assert result1 == result2


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestAdditionalCoverage:
    """Additional tests for edge cases and improved coverage."""

    def test_generate_with_metadata_empty_spans(self, generator: PlantUMLGenerator):
        """Test generate_with_metadata with empty span list."""
        trace = Trace(traceId="empty", spans=[], total_duration=0)
        result = generator.generate_with_metadata([], trace)

        assert "@startuml" in result
        assert "@enduml" in result
        assert "No spans to display" in result

    def test_generate_with_metadata_both_options(
        self, generator: PlantUMLGenerator, child_span: Span
    ):
        """Test generate_with_metadata with both duration and depth."""
        trace = Trace(traceId="test", spans=[child_span], total_duration=500)
        result = generator.generate_with_metadata(
            [child_span], trace, include_duration=True, include_depth=True
        )

        assert "(500μs)" in result
        assert r"[depth: 1]" in result

    def test_slow_threshold_at_boundary(self, generator: PlantUMLGenerator):
        """Test slow threshold calculation at exact boundary."""
        # Create exactly 10 spans where 90th percentile is clear
        spans = []
        for i in range(10):
            span = Span(
                spanId=f"span{i}",
                parentSpanId=None if i == 0 else "span0",
                service="svc",
                operation=f"op{i}",
                duration=(i + 1) * 100,  # 100, 200, ..., 1000
                status="OK",
                depth=0 if i == 0 else 1,
                children=[],
            )
            spans.append(span)

        spans[0].children = spans[1:]
        trace = Trace(traceId="test", spans=spans, total_duration=1000)

        result = generator.generate(spans, trace, enable_styling=True)

        assert "skinparam activityspan9BackgroundColor Orange" in result  # The 90th percentile span
        assert "skinparam activityspan0BackgroundColor LightBlue" in result  # A normal span

    def test_all_error_spans(self, generator: PlantUMLGenerator):
        """Test styling when all spans have errors."""
        spans = [
            Span(
                spanId=f"error{i}",
                parentSpanId=None if i == 0 else "error0",
                service="svc",
                operation=f"fail{i}",
                duration=100,
                status="ERROR",
                depth=0 if i == 0 else 1,
                children=[],
            )
            for i in range(3)
        ]

        spans[0].children = spans[1:]
        trace = Trace(traceId="test", spans=spans, total_duration=100)

        result = generator.generate(spans, trace, enable_styling=True)

        assert "skinparam activityerror0BackgroundColor Red" in result
        assert "skinparam activityerror1BackgroundColor Red" in result
        assert "skinparam activityerror2BackgroundColor Red" in result

    def test_subgraph_with_special_chars_in_title(
        self, generator: PlantUMLGenerator, simple_span: Span
    ):
        """Test title escaping with special characters."""
        trace = Trace(traceId="test", spans=[simple_span], total_duration=1000)
        result = generator.generate([simple_span], trace, title="My <Test> Title")

        assert r"title My \<Test\> Title" in result

    def test_backslash_in_operation(self, generator: PlantUMLGenerator):
        """Test handling of backslash in operation name."""
        span = Span(
            spanId="backslash",
            parentSpanId=None,
            service="svc",
            operation="path\\to\\file",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=100)
        result = generator.generate([span], trace, enable_styling=False)

        assert r":svc: path\\\\to\\\\file; as backslash" in result

    def test_forward_slash_in_operation(self, generator: PlantUMLGenerator):
        """Test handling of forward slash in operation name."""
        span = Span(
            spanId="fslash",
            parentSpanId=None,
            service="svc",
            operation="path/to/file",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=100)
        result = generator.generate([span], trace, enable_styling=False)

        assert ":svc: path/to/file; as fslash" in result

    def test_duration_edge_case_exactly_1ms(self, generator: PlantUMLGenerator):
        """Test duration formatting at exactly 1ms boundary."""
        span = Span(
            spanId="boundary",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=1000,  # Exactly 1ms
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=1000)
        result = generator.generate_with_metadata([span], trace)

        assert "(1.0ms)" in result

    def test_duration_edge_case_exactly_1s(self, generator: PlantUMLGenerator):
        """Test duration formatting at exactly 1s boundary."""
        span = Span(
            spanId="boundary",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=1_000_000,  # Exactly 1s
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=1_000_000)
        result = generator.generate_with_metadata([span], trace)

        assert "(1.00s)" in result

    def test_no_slow_threshold_when_single_span(self, generator: PlantUMLGenerator):
        """Test that single span doesn't get slow styling incorrectly."""
        span = Span(
            spanId="single",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=10000,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="test", spans=[span], total_duration=10000)
        result = generator.generate([span], trace, enable_styling=True)

        assert (
            "skinparam activitysingleBackgroundColor LightBlue" in result
        )  # Single span should be normal by default

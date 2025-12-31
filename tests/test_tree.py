"""
Tests for tracemaid.utils.tree module.

Tests cover tree traversal functions and statistics functions
with various tree structures: single span, deep trees, wide trees,
and complex multi-service traces.
"""

import pytest
from tracemaid.core.parser import Span, Trace
from tracemaid.utils.tree import (
    flatten_tree,
    get_depth,
    get_ancestors,
    get_descendants,
    get_max_depth,
    count_spans,
    get_spans_by_service,
    get_span_by_id,
    get_leaf_spans,
    get_sibling_spans,
    calculate_subtree_duration,
)


# Fixtures for creating test span trees


@pytest.fixture
def single_span() -> Span:
    """Create a single root span with no children."""
    return Span(
        spanId="root",
        parentSpanId=None,
        service="test-service",
        operation="test-op",
        duration=1000,
        status="OK",
        depth=0,
        children=[],
    )


@pytest.fixture
def single_span_trace(single_span: Span) -> Trace:
    """Create a trace with a single span."""
    return Trace(
        traceId="trace-1",
        spans=[single_span],
        total_duration=1000,
    )


@pytest.fixture
def deep_tree() -> Span:
    """Create a deep tree: root -> child1 -> child2 -> child3 -> leaf."""
    leaf = Span(
        spanId="leaf",
        parentSpanId="child3",
        service="service-d",
        operation="leaf-op",
        duration=100,
        status="OK",
        depth=4,
        children=[],
    )
    child3 = Span(
        spanId="child3",
        parentSpanId="child2",
        service="service-c",
        operation="op3",
        duration=200,
        status="OK",
        depth=3,
        children=[leaf],
    )
    child2 = Span(
        spanId="child2",
        parentSpanId="child1",
        service="service-b",
        operation="op2",
        duration=300,
        status="OK",
        depth=2,
        children=[child3],
    )
    child1 = Span(
        spanId="child1",
        parentSpanId="root",
        service="service-a",
        operation="op1",
        duration=400,
        status="OK",
        depth=1,
        children=[child2],
    )
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="service-root",
        operation="root-op",
        duration=500,
        status="OK",
        depth=0,
        children=[child1],
    )
    return root


@pytest.fixture
def deep_tree_trace(deep_tree: Span) -> Trace:
    """Create a trace from deep tree."""
    # Flatten to get all spans
    all_spans = flatten_tree(deep_tree)
    total_duration = sum(s.duration for s in all_spans)
    return Trace(
        traceId="trace-deep",
        spans=all_spans,
        total_duration=total_duration,
    )


@pytest.fixture
def wide_tree() -> Span:
    """Create a wide tree: root with 5 direct children (no grandchildren)."""
    children = [
        Span(
            spanId=f"child-{i}",
            parentSpanId="root",
            service=f"service-{i}",
            operation=f"op-{i}",
            duration=100 * (i + 1),
            status="OK" if i != 2 else "ERROR",
            depth=1,
            children=[],
        )
        for i in range(5)
    ]
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="root-service",
        operation="root-op",
        duration=2000,
        status="OK",
        depth=0,
        children=children,
    )
    return root


@pytest.fixture
def wide_tree_trace(wide_tree: Span) -> Trace:
    """Create a trace from wide tree."""
    all_spans = flatten_tree(wide_tree)
    return Trace(
        traceId="trace-wide",
        spans=all_spans,
        total_duration=2000,
    )


@pytest.fixture
def complex_tree() -> Span:
    """Create a complex tree with mixed depth and width.

    Structure:
        root
        ├── a1
        │   ├── b1
        │   │   └── c1
        │   └── b2
        └── a2
            └── b3
    """
    c1 = Span(
        spanId="c1", parentSpanId="b1", service="svc-c",
        operation="op-c1", duration=50, status="OK", depth=3, children=[]
    )
    b1 = Span(
        spanId="b1", parentSpanId="a1", service="svc-b",
        operation="op-b1", duration=100, status="OK", depth=2, children=[c1]
    )
    b2 = Span(
        spanId="b2", parentSpanId="a1", service="svc-b",
        operation="op-b2", duration=80, status="ERROR", depth=2, children=[]
    )
    b3 = Span(
        spanId="b3", parentSpanId="a2", service="svc-b",
        operation="op-b3", duration=120, status="OK", depth=2, children=[]
    )
    a1 = Span(
        spanId="a1", parentSpanId="root", service="svc-a",
        operation="op-a1", duration=300, status="OK", depth=1, children=[b1, b2]
    )
    a2 = Span(
        spanId="a2", parentSpanId="root", service="svc-a",
        operation="op-a2", duration=200, status="OK", depth=1, children=[b3]
    )
    root = Span(
        spanId="root", parentSpanId=None, service="svc-root",
        operation="main", duration=1000, status="OK", depth=0, children=[a1, a2]
    )
    return root


@pytest.fixture
def complex_tree_trace(complex_tree: Span) -> Trace:
    """Create a trace from complex tree."""
    all_spans = flatten_tree(complex_tree)
    return Trace(
        traceId="trace-complex",
        spans=all_spans,
        total_duration=1000,
    )


# Tests for flatten_tree


class TestFlattenTree:
    """Tests for flatten_tree function."""

    def test_single_span(self, single_span: Span) -> None:
        """Test flattening a single span returns list with one element."""
        result = flatten_tree(single_span)
        assert len(result) == 1
        assert result[0] == single_span

    def test_deep_tree(self, deep_tree: Span) -> None:
        """Test flattening a deep tree returns all 5 spans."""
        result = flatten_tree(deep_tree)
        assert len(result) == 5
        # Root should be first
        assert result[0].spanId == "root"
        # Verify all span IDs are present
        span_ids = {s.spanId for s in result}
        assert span_ids == {"root", "child1", "child2", "child3", "leaf"}

    def test_wide_tree(self, wide_tree: Span) -> None:
        """Test flattening a wide tree returns all 6 spans."""
        result = flatten_tree(wide_tree)
        assert len(result) == 6
        assert result[0].spanId == "root"

    def test_complex_tree(self, complex_tree: Span) -> None:
        """Test flattening a complex tree returns all 7 spans."""
        result = flatten_tree(complex_tree)
        assert len(result) == 7
        expected_ids = {"root", "a1", "a2", "b1", "b2", "b3", "c1"}
        actual_ids = {s.spanId for s in result}
        assert actual_ids == expected_ids

    def test_preserves_order(self, complex_tree: Span) -> None:
        """Test that flatten_tree uses pre-order traversal."""
        result = flatten_tree(complex_tree)
        # Pre-order: root, a1, b1, c1, b2, a2, b3
        span_ids = [s.spanId for s in result]
        assert span_ids[0] == "root"
        # a1 comes before a2
        assert span_ids.index("a1") < span_ids.index("a2")
        # b1 comes right after a1
        assert span_ids.index("b1") == span_ids.index("a1") + 1


# Tests for get_depth


class TestGetDepth:
    """Tests for get_depth function."""

    def test_root_span_depth(self, single_span: Span) -> None:
        """Test root span has depth 0."""
        assert get_depth(single_span) == 0

    def test_deep_tree_depths(self, deep_tree: Span) -> None:
        """Test depths in deep tree."""
        spans = flatten_tree(deep_tree)
        depths = {s.spanId: get_depth(s) for s in spans}
        assert depths["root"] == 0
        assert depths["child1"] == 1
        assert depths["child2"] == 2
        assert depths["child3"] == 3
        assert depths["leaf"] == 4

    def test_wide_tree_depths(self, wide_tree: Span) -> None:
        """Test all children of wide tree have depth 1."""
        spans = flatten_tree(wide_tree)
        for span in spans:
            if span.spanId == "root":
                assert get_depth(span) == 0
            else:
                assert get_depth(span) == 1


# Tests for get_ancestors


class TestGetAncestors:
    """Tests for get_ancestors function."""

    def test_root_has_no_ancestors(self, single_span: Span, single_span_trace: Trace) -> None:
        """Test root span returns empty ancestors list."""
        ancestors = get_ancestors(single_span, single_span_trace.spans)
        assert ancestors == []

    def test_deep_tree_ancestors(self, deep_tree: Span) -> None:
        """Test ancestors in deep tree."""
        all_spans = flatten_tree(deep_tree)
        leaf = next(s for s in all_spans if s.spanId == "leaf")

        ancestors = get_ancestors(leaf, all_spans)
        ancestor_ids = [a.spanId for a in ancestors]

        # Should be ordered from immediate parent to root
        assert ancestor_ids == ["child3", "child2", "child1", "root"]

    def test_mid_level_ancestors(self, complex_tree: Span) -> None:
        """Test ancestors of mid-level span."""
        all_spans = flatten_tree(complex_tree)
        b1 = next(s for s in all_spans if s.spanId == "b1")

        ancestors = get_ancestors(b1, all_spans)
        ancestor_ids = [a.spanId for a in ancestors]

        assert ancestor_ids == ["a1", "root"]


# Tests for get_descendants


class TestGetDescendants:
    """Tests for get_descendants function."""

    def test_leaf_has_no_descendants(self, single_span: Span) -> None:
        """Test leaf span returns empty descendants list."""
        descendants = get_descendants(single_span)
        assert descendants == []

    def test_deep_tree_root_descendants(self, deep_tree: Span) -> None:
        """Test root of deep tree has 4 descendants."""
        descendants = get_descendants(deep_tree)
        assert len(descendants) == 4
        descendant_ids = {d.spanId for d in descendants}
        assert descendant_ids == {"child1", "child2", "child3", "leaf"}

    def test_mid_level_descendants(self, complex_tree: Span) -> None:
        """Test descendants of mid-level span."""
        all_spans = flatten_tree(complex_tree)
        a1 = next(s for s in all_spans if s.spanId == "a1")

        descendants = get_descendants(a1)
        descendant_ids = {d.spanId for d in descendants}

        assert descendant_ids == {"b1", "b2", "c1"}


# Tests for get_max_depth


class TestGetMaxDepth:
    """Tests for get_max_depth function."""

    def test_single_span_max_depth(self, single_span_trace: Trace) -> None:
        """Test single span trace has max depth 0."""
        assert get_max_depth(single_span_trace) == 0

    def test_deep_tree_max_depth(self, deep_tree_trace: Trace) -> None:
        """Test deep tree has max depth 4."""
        assert get_max_depth(deep_tree_trace) == 4

    def test_wide_tree_max_depth(self, wide_tree_trace: Trace) -> None:
        """Test wide tree has max depth 1."""
        assert get_max_depth(wide_tree_trace) == 1

    def test_complex_tree_max_depth(self, complex_tree_trace: Trace) -> None:
        """Test complex tree has max depth 3."""
        assert get_max_depth(complex_tree_trace) == 3

    def test_empty_trace(self) -> None:
        """Test empty trace returns 0."""
        empty_trace = Trace(traceId="empty", spans=[], total_duration=0)
        assert get_max_depth(empty_trace) == 0


# Tests for count_spans


class TestCountSpans:
    """Tests for count_spans function."""

    def test_single_span_count(self, single_span_trace: Trace) -> None:
        """Test single span trace has count 1."""
        assert count_spans(single_span_trace) == 1

    def test_deep_tree_count(self, deep_tree_trace: Trace) -> None:
        """Test deep tree has 5 spans."""
        assert count_spans(deep_tree_trace) == 5

    def test_wide_tree_count(self, wide_tree_trace: Trace) -> None:
        """Test wide tree has 6 spans."""
        assert count_spans(wide_tree_trace) == 6

    def test_complex_tree_count(self, complex_tree_trace: Trace) -> None:
        """Test complex tree has 7 spans."""
        assert count_spans(complex_tree_trace) == 7

    def test_empty_trace_count(self) -> None:
        """Test empty trace has count 0."""
        empty_trace = Trace(traceId="empty", spans=[], total_duration=0)
        assert count_spans(empty_trace) == 0


# Tests for get_spans_by_service


class TestGetSpansByService:
    """Tests for get_spans_by_service function."""

    def test_single_service(self, single_span_trace: Trace) -> None:
        """Test trace with single service."""
        by_service = get_spans_by_service(single_span_trace)
        assert len(by_service) == 1
        assert "test-service" in by_service
        assert len(by_service["test-service"]) == 1

    def test_multiple_services(self, complex_tree_trace: Trace) -> None:
        """Test trace with multiple services."""
        by_service = get_spans_by_service(complex_tree_trace)

        # Check service count
        assert "svc-root" in by_service
        assert "svc-a" in by_service
        assert "svc-b" in by_service
        assert "svc-c" in by_service

        # Check span counts per service
        assert len(by_service["svc-root"]) == 1
        assert len(by_service["svc-a"]) == 2
        assert len(by_service["svc-b"]) == 3
        assert len(by_service["svc-c"]) == 1

    def test_empty_trace_by_service(self) -> None:
        """Test empty trace returns empty dict."""
        empty_trace = Trace(traceId="empty", spans=[], total_duration=0)
        by_service = get_spans_by_service(empty_trace)
        assert by_service == {}


# Tests for get_span_by_id


class TestGetSpanById:
    """Tests for get_span_by_id function."""

    def test_find_existing_span(self, complex_tree: Span) -> None:
        """Test finding an existing span."""
        all_spans = flatten_tree(complex_tree)
        result = get_span_by_id("b2", all_spans)
        assert result is not None
        assert result.spanId == "b2"
        assert result.operation == "op-b2"

    def test_span_not_found(self, complex_tree: Span) -> None:
        """Test returns None for non-existent span."""
        all_spans = flatten_tree(complex_tree)
        result = get_span_by_id("nonexistent", all_spans)
        assert result is None

    def test_find_root_span(self, complex_tree: Span) -> None:
        """Test finding root span."""
        all_spans = flatten_tree(complex_tree)
        result = get_span_by_id("root", all_spans)
        assert result is not None
        assert result.parentSpanId is None


# Tests for get_leaf_spans


class TestGetLeafSpans:
    """Tests for get_leaf_spans function."""

    def test_single_span_is_leaf(self, single_span: Span) -> None:
        """Test single span is itself a leaf."""
        leaves = get_leaf_spans(single_span)
        assert len(leaves) == 1
        assert leaves[0] == single_span

    def test_deep_tree_one_leaf(self, deep_tree: Span) -> None:
        """Test deep tree has exactly one leaf."""
        leaves = get_leaf_spans(deep_tree)
        assert len(leaves) == 1
        assert leaves[0].spanId == "leaf"

    def test_wide_tree_all_children_are_leaves(self, wide_tree: Span) -> None:
        """Test wide tree's children are all leaves."""
        leaves = get_leaf_spans(wide_tree)
        assert len(leaves) == 5
        leaf_ids = {l.spanId for l in leaves}
        expected = {f"child-{i}" for i in range(5)}
        assert leaf_ids == expected

    def test_complex_tree_leaves(self, complex_tree: Span) -> None:
        """Test complex tree has correct leaves."""
        leaves = get_leaf_spans(complex_tree)
        leaf_ids = {l.spanId for l in leaves}
        assert leaf_ids == {"c1", "b2", "b3"}


# Tests for get_sibling_spans


class TestGetSiblingSpans:
    """Tests for get_sibling_spans function."""

    def test_root_has_no_siblings(self, complex_tree: Span) -> None:
        """Test root span has no siblings."""
        all_spans = flatten_tree(complex_tree)
        siblings = get_sibling_spans(complex_tree, all_spans)
        assert siblings == []

    def test_find_siblings(self, complex_tree: Span) -> None:
        """Test finding siblings of a span."""
        all_spans = flatten_tree(complex_tree)
        a1 = next(s for s in all_spans if s.spanId == "a1")

        siblings = get_sibling_spans(a1, all_spans)
        sibling_ids = {s.spanId for s in siblings}

        assert sibling_ids == {"a2"}

    def test_multiple_siblings(self, wide_tree: Span) -> None:
        """Test span with multiple siblings."""
        all_spans = flatten_tree(wide_tree)
        child0 = next(s for s in all_spans if s.spanId == "child-0")

        siblings = get_sibling_spans(child0, all_spans)
        sibling_ids = {s.spanId for s in siblings}

        expected = {f"child-{i}" for i in range(1, 5)}
        assert sibling_ids == expected


# Tests for calculate_subtree_duration


class TestCalculateSubtreeDuration:
    """Tests for calculate_subtree_duration function."""

    def test_single_span_duration(self, single_span: Span) -> None:
        """Test single span returns its own duration."""
        assert calculate_subtree_duration(single_span) == 1000

    def test_deep_tree_total(self, deep_tree: Span) -> None:
        """Test deep tree sums all durations."""
        total = calculate_subtree_duration(deep_tree)
        expected = 500 + 400 + 300 + 200 + 100  # All span durations
        assert total == expected

    def test_subtree_duration(self, complex_tree: Span) -> None:
        """Test subtree duration calculation."""
        all_spans = flatten_tree(complex_tree)
        a1 = next(s for s in all_spans if s.spanId == "a1")

        # a1 (300) + b1 (100) + c1 (50) + b2 (80)
        expected = 300 + 100 + 50 + 80
        assert calculate_subtree_duration(a1) == expected

    def test_leaf_subtree_duration(self, deep_tree: Span) -> None:
        """Test leaf span subtree is just its own duration."""
        all_spans = flatten_tree(deep_tree)
        leaf = next(s for s in all_spans if s.spanId == "leaf")

        assert calculate_subtree_duration(leaf) == 100


# Edge case tests


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_span_with_orphan_parent(self) -> None:
        """Test get_ancestors handles orphan parent gracefully."""
        # Span with parent ID that doesn't exist in span list
        orphan = Span(
            spanId="orphan",
            parentSpanId="nonexistent",
            service="svc",
            operation="op",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        ancestors = get_ancestors(orphan, [orphan])
        assert ancestors == []

    def test_empty_children_list(self) -> None:
        """Test functions handle span with empty children list."""
        span = Span(
            spanId="empty-children",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        assert get_descendants(span) == []
        assert get_leaf_spans(span) == [span]
        assert flatten_tree(span) == [span]

    def test_large_depth_value(self) -> None:
        """Test get_depth handles large depth values."""
        span = Span(
            spanId="deep",
            parentSpanId="parent",
            service="svc",
            operation="op",
            duration=100,
            status="OK",
            depth=1000,
            children=[],
        )
        assert get_depth(span) == 1000

    def test_service_with_special_characters(self) -> None:
        """Test get_spans_by_service handles special characters in names."""
        span = Span(
            spanId="special",
            parentSpanId=None,
            service="service-with-dashes_and_underscores.and.dots",
            operation="op",
            duration=100,
            status="OK",
            depth=0,
            children=[],
        )
        trace = Trace(traceId="t", spans=[span], total_duration=100)
        by_service = get_spans_by_service(trace)
        assert "service-with-dashes_and_underscores.and.dots" in by_service

    def test_get_span_by_id_empty_list(self) -> None:
        """Test get_span_by_id with empty span list."""
        result = get_span_by_id("any-id", [])
        assert result is None

    def test_get_sibling_spans_no_siblings(self, deep_tree: Span) -> None:
        """Test span with no siblings (only child)."""
        all_spans = flatten_tree(deep_tree)
        child1 = next(s for s in all_spans if s.spanId == "child1")
        siblings = get_sibling_spans(child1, all_spans)
        assert siblings == []

    def test_get_ancestors_partial_chain(self) -> None:
        """Test get_ancestors when parent chain breaks mid-way."""
        # Create span whose parent exists but grandparent doesn't
        parent = Span(
            spanId="parent",
            parentSpanId="missing-grandparent",  # This ID won't exist
            service="svc",
            operation="op",
            duration=100,
            status="OK",
            depth=1,
            children=[],
        )
        child = Span(
            spanId="child",
            parentSpanId="parent",
            service="svc",
            operation="op",
            duration=50,
            status="OK",
            depth=2,
            children=[],
        )
        parent.children = [child]

        all_spans = [parent, child]
        ancestors = get_ancestors(child, all_spans)

        # Should only find parent, not the missing grandparent
        assert len(ancestors) == 1
        assert ancestors[0].spanId == "parent"

    def test_calculate_subtree_duration_zero_duration(self) -> None:
        """Test subtree duration calculation with zero-duration spans."""
        root = Span(
            spanId="root",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=0,
            status="OK",
            depth=0,
            children=[],
        )
        child = Span(
            spanId="child",
            parentSpanId="root",
            service="svc",
            operation="op",
            duration=0,
            status="OK",
            depth=1,
            children=[],
        )
        root.children = [child]

        assert calculate_subtree_duration(root) == 0

    def test_get_leaf_spans_deeply_nested(self) -> None:
        """Test get_leaf_spans works with deeply nested single-path tree."""
        # Create a chain: root -> a -> b -> c -> d (leaf)
        d = Span(spanId="d", parentSpanId="c", service="s", operation="o", duration=10, status="OK", depth=4, children=[])
        c = Span(spanId="c", parentSpanId="b", service="s", operation="o", duration=10, status="OK", depth=3, children=[d])
        b = Span(spanId="b", parentSpanId="a", service="s", operation="o", duration=10, status="OK", depth=2, children=[c])
        a = Span(spanId="a", parentSpanId="root", service="s", operation="o", duration=10, status="OK", depth=1, children=[b])
        root = Span(spanId="root", parentSpanId=None, service="s", operation="o", duration=10, status="OK", depth=0, children=[a])

        leaves = get_leaf_spans(root)
        assert len(leaves) == 1
        assert leaves[0].spanId == "d"

    def test_get_descendants_deeply_nested(self) -> None:
        """Test get_descendants counts all nodes in deeply nested tree."""
        d = Span(spanId="d", parentSpanId="c", service="s", operation="o", duration=10, status="OK", depth=4, children=[])
        c = Span(spanId="c", parentSpanId="b", service="s", operation="o", duration=10, status="OK", depth=3, children=[d])
        b = Span(spanId="b", parentSpanId="a", service="s", operation="o", duration=10, status="OK", depth=2, children=[c])
        a = Span(spanId="a", parentSpanId="root", service="s", operation="o", duration=10, status="OK", depth=1, children=[b])
        root = Span(spanId="root", parentSpanId=None, service="s", operation="o", duration=10, status="OK", depth=0, children=[a])

        descendants = get_descendants(root)
        assert len(descendants) == 4
        descendant_ids = {s.spanId for s in descendants}
        assert descendant_ids == {"a", "b", "c", "d"}

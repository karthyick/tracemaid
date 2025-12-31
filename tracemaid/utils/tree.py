"""
tracemaid.utils.tree - Tree traversal and manipulation utilities for span trees.

This module provides helper functions for traversing, analyzing, and
manipulating span trees created from OpenTelemetry traces.

Functions:
    flatten_tree: Flatten a span tree into a list of spans
    get_depth: Get the depth of a span in the tree
    get_ancestors: Get all ancestor spans of a given span
    get_descendants: Get all descendant spans of a given span
    get_max_depth: Get the maximum depth in a trace
    count_spans: Count total spans in a trace
    get_spans_by_service: Group spans by their service name
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from tracemaid.core.parser import Span, Trace


def flatten_tree(root_span: "Span") -> List["Span"]:
    """Flatten a span tree into a list using pre-order traversal.

    Traverses the span tree starting from the root and collects all spans
    into a flat list. The root span is always first, followed by its
    descendants in depth-first pre-order.

    Args:
        root_span: The root span of the tree to flatten

    Returns:
        List of all spans in the tree, starting with root_span

    Example:
        >>> root = Span(spanId="root", ...)
        >>> root.children = [child1, child2]
        >>> spans = flatten_tree(root)
        >>> len(spans)  # Returns total count including root
    """
    result: List["Span"] = []
    _flatten_recursive(root_span, result)
    return result


def _flatten_recursive(span: "Span", result: List["Span"]) -> None:
    """Recursively flatten span tree into result list.

    Args:
        span: Current span to process
        result: List to append spans to
    """
    result.append(span)
    for child in span.children:
        _flatten_recursive(child, result)


def get_depth(span: "Span") -> int:
    """Get the depth of a span in the tree.

    The depth is the number of edges from the root to this span.
    Root spans have depth 0.

    Note: This returns the pre-computed depth stored on the span object.
    For accurate results, ensure the span tree has been built with
    depths calculated by the parser.

    Args:
        span: The span to get depth for

    Returns:
        Integer depth value (0 for root, 1 for direct children, etc.)

    Example:
        >>> depth = get_depth(some_span)
        >>> print(f"Span is {depth} levels deep")
    """
    return span.depth


def get_ancestors(span: "Span", all_spans: List["Span"]) -> List["Span"]:
    """Get all ancestor spans of a given span.

    Traverses up the tree from the given span to the root,
    collecting all parent spans along the way. The result is
    ordered from immediate parent to root (bottom-up).

    Args:
        span: The span to find ancestors for
        all_spans: List of all spans in the trace (used for parent lookup)

    Returns:
        List of ancestor spans, ordered from immediate parent to root.
        Empty list if span is a root span.

    Example:
        >>> ancestors = get_ancestors(leaf_span, trace.spans)
        >>> for ancestor in ancestors:
        ...     print(ancestor.operation)
    """
    ancestors: List["Span"] = []

    # Build a map for efficient parent lookup
    span_map: Dict[str, "Span"] = {s.spanId: s for s in all_spans}

    current_parent_id: Optional[str] = span.parentSpanId
    while current_parent_id is not None:
        parent = span_map.get(current_parent_id)
        if parent is None:
            # Parent not found in span list, stop traversal
            break
        ancestors.append(parent)
        current_parent_id = parent.parentSpanId

    return ancestors


def get_descendants(span: "Span") -> List["Span"]:
    """Get all descendant spans of a given span.

    Collects all children, grandchildren, and further descendants
    of the given span using depth-first traversal.

    Args:
        span: The span to find descendants for

    Returns:
        List of all descendant spans, not including the span itself.
        Empty list if span has no children.

    Example:
        >>> descendants = get_descendants(root_span)
        >>> print(f"Root has {len(descendants)} descendants")
    """
    descendants: List["Span"] = []
    _collect_descendants(span, descendants)
    return descendants


def _collect_descendants(span: "Span", result: List["Span"]) -> None:
    """Recursively collect all descendants of a span.

    Args:
        span: Current span whose children to process
        result: List to append descendants to
    """
    for child in span.children:
        result.append(child)
        _collect_descendants(child, result)


def get_max_depth(trace: "Trace") -> int:
    """Get the maximum depth of any span in the trace.

    Scans all spans in the trace to find the one with the
    greatest depth value.

    Args:
        trace: The trace to analyze

    Returns:
        Maximum depth value found. Returns 0 if trace has no spans
        or only root-level spans.

    Example:
        >>> max_depth = get_max_depth(trace)
        >>> print(f"Trace has {max_depth + 1} levels")
    """
    if not trace.spans:
        return 0

    return max(span.depth for span in trace.spans)


def count_spans(trace: "Trace") -> int:
    """Count the total number of spans in a trace.

    Args:
        trace: The trace to count spans in

    Returns:
        Total number of spans in the trace

    Example:
        >>> total = count_spans(trace)
        >>> print(f"Trace contains {total} spans")
    """
    return len(trace.spans)


def get_spans_by_service(trace: "Trace") -> Dict[str, List["Span"]]:
    """Group spans by their service name.

    Creates a dictionary mapping service names to lists of spans
    that belong to each service.

    Args:
        trace: The trace to analyze

    Returns:
        Dictionary mapping service names to lists of spans.
        Each span appears exactly once in the appropriate service list.

    Example:
        >>> by_service = get_spans_by_service(trace)
        >>> for service, spans in by_service.items():
        ...     print(f"{service}: {len(spans)} spans")
    """
    service_map: Dict[str, List["Span"]] = {}

    for span in trace.spans:
        service_name = span.service
        if service_name not in service_map:
            service_map[service_name] = []
        service_map[service_name].append(span)

    return service_map


def get_span_by_id(span_id: str, all_spans: List["Span"]) -> Optional["Span"]:
    """Find a span by its ID.

    Searches through the span list to find a span with the matching ID.

    Args:
        span_id: The span ID to search for
        all_spans: List of spans to search in

    Returns:
        The matching Span object, or None if not found

    Example:
        >>> span = get_span_by_id("abc123", trace.spans)
        >>> if span:
        ...     print(span.operation)
    """
    for span in all_spans:
        if span.spanId == span_id:
            return span
    return None


def get_leaf_spans(span: "Span") -> List["Span"]:
    """Get all leaf spans (spans with no children) under a given span.

    Args:
        span: The span to find leaf descendants for

    Returns:
        List of leaf spans. If the span itself is a leaf, returns [span].

    Example:
        >>> leaves = get_leaf_spans(root_span)
        >>> print(f"Found {len(leaves)} terminal operations")
    """
    leaves: List["Span"] = []
    _collect_leaves(span, leaves)
    return leaves


def _collect_leaves(span: "Span", result: List["Span"]) -> None:
    """Recursively collect leaf spans.

    Args:
        span: Current span to check
        result: List to append leaf spans to
    """
    if not span.children:
        result.append(span)
    else:
        for child in span.children:
            _collect_leaves(child, result)


def get_sibling_spans(span: "Span", all_spans: List["Span"]) -> List["Span"]:
    """Get sibling spans (spans with the same parent).

    Args:
        span: The span to find siblings for
        all_spans: List of all spans in the trace

    Returns:
        List of sibling spans, not including the span itself.
        Empty list if span is root or has no siblings.

    Example:
        >>> siblings = get_sibling_spans(some_span, trace.spans)
        >>> print(f"Span has {len(siblings)} siblings")
    """
    if span.parentSpanId is None:
        # Root spans don't have siblings in the traditional sense
        return []

    siblings: List["Span"] = []
    for s in all_spans:
        if s.parentSpanId == span.parentSpanId and s.spanId != span.spanId:
            siblings.append(s)

    return siblings


def calculate_subtree_duration(span: "Span") -> int:
    """Calculate the total duration of a span and all its descendants.

    This sums the duration of the span itself plus all its descendants.
    Note: This is the sum of individual durations, not wall-clock time.

    Args:
        span: The root of the subtree to calculate duration for

    Returns:
        Total duration in microseconds

    Example:
        >>> subtree_time = calculate_subtree_duration(span)
        >>> print(f"Subtree total: {subtree_time} µs")
    """
    total = span.duration
    for child in span.children:
        total += calculate_subtree_duration(child)
    return total

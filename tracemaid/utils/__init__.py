"""
tracemaid.utils - Utility functions for tree operations and helpers.

This subpackage contains utility functions:
- tree: Functions for building and traversing span trees
"""

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

__all__ = [
    "flatten_tree",
    "get_depth",
    "get_ancestors",
    "get_descendants",
    "get_max_depth",
    "count_spans",
    "get_spans_by_service",
    "get_span_by_id",
    "get_leaf_spans",
    "get_sibling_spans",
    "calculate_subtree_duration",
]

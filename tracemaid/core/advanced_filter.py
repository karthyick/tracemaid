from __future__ import annotations
from enum import Enum
from typing import List, Any, TypedDict


# Sentinel object to distinguish between an attribute not found and an attribute explicitly set to None
class _MissingAttribute:
    def __repr__(self):
        return "<MISSING_ATTRIBUTE>"


MISSING_ATTRIBUTE = _MissingAttribute()


class FilterOperator(Enum):
    """Enumeration for supported filter operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    LT = "lt"
    GE = "ge"
    LE = "le"
    CONTAINS = "contains"
    NCONTAINS = "ncontains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


class FilterCriterion(TypedDict):
    """
    Represents a single criterion for advanced filtering.

    Attributes:
        attribute: The dot-separated path to the attribute on the span (e.g., "service", "attributes.http.method").
        operator: The comparison operator to use (e.g., FilterOperator.EQ).
        value: The value to compare against the span's attribute.
    """

    attribute: str
    operator: FilterOperator
    value: Any


def _get_attribute_value(span: Span, attribute_path: str) -> Any | _MissingAttribute:
    """
    Safely retrieves a nested attribute value from a Span object.

    This function supports direct attribute access (e.g., "service", "duration")
    and nested attributes. For paths starting with "attributes.", it first tries
    to retrieve the full dotted key directly from `span.attributes`. If not found,
    it then attempts to traverse it as a nested dictionary.

    Args:
        span: The Span object from which to retrieve the attribute.
        attribute_path: A dot-separated string representing the path to the attribute
                        (e.g., "service", "attributes.http.method").

    Returns:
        The value of the requested attribute, MISSING_ATTRIBUTE if the attribute path is invalid
        or the attribute is not found.
    """
    current_value: Any = span
    parts = attribute_path.split(".")

    if parts[0] == "attributes" and len(parts) > 0:
        if span.attributes is None:
            return MISSING_ATTRIBUTE

        # Try to get the full dotted key directly from span.attributes first
        full_attr_key_in_attrs = ".".join(parts[1:])
        if full_attr_key_in_attrs in span.attributes:
            return span.attributes.get(full_attr_key_in_attrs)

        # If not found as a direct dotted key, then try to traverse it as nested dicts
        current_value = span.attributes
        parts = parts[1:]  # Update parts to only include the keys within attributes

    # Generic traversal for remaining parts (either from span object or nested attributes dict)
    for part in parts:
        if current_value is None:
            return MISSING_ATTRIBUTE

        if isinstance(current_value, dict):
            current_value = current_value.get(part, MISSING_ATTRIBUTE)
            if (
                current_value is MISSING_ATTRIBUTE
            ):  # if .get(part) returns MISSING_ATTRIBUTE, it means key was not found.
                return MISSING_ATTRIBUTE
        elif hasattr(current_value, part):
            current_value = getattr(current_value, part)
        else:
            return MISSING_ATTRIBUTE  # Attribute not found or path is invalid
    return current_value


def _evaluate_condition(
    span: Span, attribute_path: str, operator: FilterOperator, value: Any
) -> bool:
    """
    Evaluates a single filter condition against a span.

    Compares the span's attribute value (retrieved using `attribute_path`)
    against a given `value` using the specified `operator`.

    Args:
        span: The Span object to evaluate.
        attribute_path: The dot-separated path to the attribute on the span.
        operator: The comparison operator (e.g., FilterOperator.EQ).
        value: The value to compare against the span's attribute.

    Returns:
        True if the condition is met, False otherwise. Returns False if the attribute
        is not found, types are incompatible for comparison, or the operator is unsupported.
    """
    span_value = _get_attribute_value(span, attribute_path)

    if span_value is MISSING_ATTRIBUTE:
        return False  # A missing attribute cannot match any filter criterion, except perhaps NE with MISSING_ATTRIBUTE itself, which is not supported

    # If span_value is None, it only matches if filter value is also None or
    # if operator is NE with non-None value, or NCONTAINS with non-None value
    if span_value is None:
        if operator == FilterOperator.EQ:
            return value is None
        elif operator == FilterOperator.NE:
            return value is not None
        elif (
            operator == FilterOperator.NCONTAINS
        ):  # if span_value is None, it cannot contain anything, so if filter value is not None, it satisfies NCONTAINS
            return value is not None
        return False  # Cannot evaluate other operators with None

    # Handle numeric comparisons with type conversion for filter value
    if operator in [FilterOperator.GT, FilterOperator.LT, FilterOperator.GE, FilterOperator.LE]:
        if not isinstance(span_value, (int, float)):
            return False  # Span value must be numeric
        try:
            value = float(value) if isinstance(span_value, float) else int(value)
        except (ValueError, TypeError):
            return False  # Filter value not convertible to numeric

    # Handle string comparisons
    if operator in [
        FilterOperator.CONTAINS,
        FilterOperator.NCONTAINS,
        FilterOperator.STARTS_WITH,
        FilterOperator.ENDS_WITH,
    ]:
        if not isinstance(span_value, str):
            return False  # Span value must be string
        if not isinstance(value, str):
            return False  # Filter value must be string

    # Perform comparisons
    if operator == FilterOperator.EQ:
        return span_value == value
    elif operator == FilterOperator.NE:
        return span_value != value
    elif operator == FilterOperator.CONTAINS:
        return value in span_value
    elif operator == FilterOperator.NCONTAINS:
        return value not in span_value
    elif operator == FilterOperator.STARTS_WITH:
        return span_value.startswith(value)
    elif operator == FilterOperator.ENDS_WITH:
        return span_value.endswith(value)
    elif operator == FilterOperator.GT:
        return span_value > value
    elif operator == FilterOperator.LT:
        return span_value < value
    elif operator == FilterOperator.GE:
        return span_value >= value
    elif operator == FilterOperator.LE:
        return span_value <= value
    else:
        return False  # Should not happen if all operators are covered


def apply_advanced_filter(spans: List[Span], filter_criteria: List[FilterCriterion]) -> List[Span]:
    """
    Filters a list of spans based on advanced criteria.

    Args:
        spans: A list of Span objects to filter.
        filter_criteria: A list of FilterCriterion objects defining the filter conditions.
                         Each criterion specifies an attribute path, an operator, and a value.

    Returns:
        A new list of Span objects that satisfy all filter criteria (AND logic).
    """
    if not filter_criteria:
        return spans  # No criteria, return all spans

    filtered_spans: List[Span] = []
    for span in spans:
        match_all_criteria = True
        for criterion in filter_criteria:
            attribute_path = criterion["attribute"]
            operator = criterion["operator"]
            value = criterion["value"]

            if not _evaluate_condition(span, attribute_path, operator, value):
                match_all_criteria = False
                break

        if match_all_criteria:
            filtered_spans.append(span)

    return filtered_spans

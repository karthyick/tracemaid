from typing import List, Any

import pytest

from tracemaid.core.advanced_filter import (
    apply_advanced_filter,
    _get_attribute_value,
    FilterOperator,
    FilterCriterion,
    MISSING_ATTRIBUTE,
)
from tracemaid.core.parser import Span


# Mock Span class for testing
class MockSpan(Span):
    def __init__(
        self, span_id, parent_span_id, service, operation, duration, status, depth, attributes=None
    ):
        super().__init__(span_id, parent_span_id, service, operation, duration, status, depth)
        self.attributes = attributes if attributes is not None else {}

    def __repr__(self):
        return f"MockSpan(id={self.spanId}, service={self.service}, op={self.operation}, attributes={self.attributes})"


# Fixture for sample spans
@pytest.fixture
def sample_spans() -> List[MockSpan]:
    return [
        MockSpan(
            span_id="1",
            parent_span_id="0",
            service="auth-service",
            operation="/login",
            duration=100,
            status="OK",
            depth=0,
            attributes={"http.status_code": 200, "user.id": "user123", "region": "us-east-1"},
        ),
        MockSpan(
            span_id="2",
            parent_span_id="1",
            service="user-service",
            operation="get-profile",
            duration=50,
            status="OK",
            depth=1,
            attributes={"http.method": "GET", "user.id": "user123", "region": "us-east-1"},
        ),
        MockSpan(
            span_id="3",
            parent_span_id="1",
            service="product-service",
            operation="get-item",
            duration=120,
            status="ERROR",
            depth=1,
            attributes={"http.status_code": 500, "item.id": "item456", "region": "us-west-2"},
        ),
        MockSpan(
            span_id="4",
            parent_span_id="2",
            service="db-service",
            operation="query-db",
            duration=30,
            status="OK",
            depth=2,
            attributes={"db.type": "postgres", "table": "users"},
        ),
        MockSpan(
            span_id="5",
            parent_span_id="0",
            service="payment-service",
            operation="process-payment",
            duration=200,
            status="OK",
            depth=0,
            attributes={"payment.method": "credit_card", "amount": 100.0, "region": "us-east-1"},
        ),
    ]


# Test cases for _get_attribute_value
@pytest.mark.parametrize(
    "span, attribute_path, expected_value",
    [
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {"k1": "v1"}),
            "service",
            "svc",
        ),
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {"k1": "v1"}),
            "attributes.k1",
            "v1",
        ),
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {"k1": {"nested": "v2"}}),
            "attributes.k1.nested",
            "v2",
        ),
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {"k1": "v1"}),
            "non_existent",
            MISSING_ATTRIBUTE,
        ),
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {"k1": "v1"}),
            "attributes.non_existent",
            MISSING_ATTRIBUTE,
        ),
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {}),
            "attributes.k1",
            MISSING_ATTRIBUTE,
        ),
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {"num": 123}),
            "attributes.num",
            123,
        ),
        (
            MockSpan("1", "0", "svc", "op", 100, "OK", 0, {"bool": True}),
            "attributes.bool",
            True,
        ),
    ],
)
def test_get_attribute_value(span: MockSpan, attribute_path: str, expected_value: Any):
    assert _get_attribute_value(span, attribute_path) == expected_value


# Test cases for apply_advanced_filter
def test_filter_by_service_eq(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(attribute="service", operator=FilterOperator.EQ, value="auth-service")
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].service == "auth-service"


def test_filter_by_status_not_eq(sample_spans: List[MockSpan]):
    criteria = [FilterCriterion(attribute="status", operator=FilterOperator.NE, value="OK")]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].status == "ERROR"


def test_filter_by_duration_gt(sample_spans: List[MockSpan]):
    criteria = [FilterCriterion(attribute="duration", operator=FilterOperator.GT, value=100)]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 2
    assert {s.spanId for s in filtered} == {"3", "5"}


def test_filter_by_depth_lt_and_service_eq(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(attribute="depth", operator=FilterOperator.LT, value=1),
        FilterCriterion(attribute="service", operator=FilterOperator.EQ, value="payment-service"),
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "5"


def test_filter_by_attribute_contains(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(
            attribute="attributes.user.id", operator=FilterOperator.CONTAINS, value="user"
        )
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 2
    assert {s.spanId for s in filtered} == {"1", "2"}


def test_filter_by_attribute_starts_with(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(
            attribute="attributes.http.method", operator=FilterOperator.STARTS_WITH, value="GE"
        )
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "2"


def test_filter_by_attribute_ends_with(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(
            attribute="attributes.db.type", operator=FilterOperator.ENDS_WITH, value="gres"
        )
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "4"


def test_filter_by_non_existent_attribute(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(attribute="non_existent_attribute", operator=FilterOperator.EQ, value="any")
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 0


def test_filter_by_nested_attribute_equals(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(
            attribute="attributes.http.status_code",
            operator=FilterOperator.EQ,
            value=200,
        )
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "1"


def test_filter_by_nested_attribute_different_type(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(
            attribute="attributes.http.status_code",
            operator=FilterOperator.EQ,
            value="200",  # String instead of int
        )
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 0


def test_empty_spans_list(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(attribute="service", operator=FilterOperator.EQ, value="auth-service")
    ]
    filtered = apply_advanced_filter([], criteria)
    assert len(filtered) == 0


def test_no_criteria(sample_spans: List[MockSpan]):
    filtered = apply_advanced_filter(sample_spans, [])
    assert len(filtered) == len(sample_spans)
    assert filtered == sample_spans


def test_filter_by_amount_greater_than_or_equal(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(attribute="attributes.amount", operator=FilterOperator.GE, value=100.0)
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "5"


def test_filter_by_amount_less_than_or_equal(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(attribute="attributes.amount", operator=FilterOperator.LE, value=100.0)
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "5"


def test_filter_by_duration_less_than_or_equal(sample_spans: List[MockSpan]):
    criteria = [FilterCriterion(attribute="duration", operator=FilterOperator.LE, value=100)]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 3
    assert {s.spanId for s in filtered} == {"1", "2", "4"}


def test_filter_by_region_not_contains(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(
            attribute="attributes.region", operator=FilterOperator.NCONTAINS, value="us-west-2"
        )
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 3
    assert {s.spanId for s in filtered} == {"1", "2", "5"}


def test_filter_by_region_not_equals(sample_spans: List[MockSpan]):
    criteria = [
        FilterCriterion(
            attribute="attributes.region", operator=FilterOperator.NE, value="us-east-1"
        )
    ]
    filtered = apply_advanced_filter(sample_spans, criteria)
    assert len(filtered) == 1
    assert {s.spanId for s in filtered} == {"3"}


def test_filter_by_attribute_eq_none(sample_spans: List[MockSpan]):
    spans_with_none = sample_spans + [
        MockSpan(
            span_id="6",
            parent_span_id="0",
            service="none-service",
            operation="none-op",
            duration=10,
            status="OK",
            depth=0,
            attributes={"nullable_key": None, "another_key": "some_value"},
        ),
        MockSpan(
            span_id="7",
            parent_span_id="0",
            service="not-none-service",
            operation="not-none-op",
            duration=20,
            status="OK",
            depth=0,
            attributes={"nullable_key": "not_none"},
        ),
    ]
    criteria = [
        FilterCriterion(attribute="attributes.nullable_key", operator=FilterOperator.EQ, value=None)
    ]
    filtered = apply_advanced_filter(spans_with_none, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "6"


def test_filter_by_attribute_ne_none(sample_spans: List[MockSpan]):
    spans_with_none = sample_spans + [
        MockSpan(
            span_id="6",
            parent_span_id="0",
            service="none-service",
            operation="none-op",
            duration=10,
            status="OK",
            depth=0,
            attributes={"nullable_key": None, "another_key": "some_value"},
        ),
        MockSpan(
            span_id="7",
            parent_span_id="0",
            service="not-none-service",
            operation="not-none-op",
            duration=20,
            status="OK",
            depth=0,
            attributes={"nullable_key": "not_none"},
        ),
    ]
    criteria = [
        FilterCriterion(attribute="attributes.nullable_key", operator=FilterOperator.NE, value=None)
    ]
    filtered = apply_advanced_filter(spans_with_none, criteria)
    assert len(filtered) == 1
    assert filtered[0].spanId == "7"

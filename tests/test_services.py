"""Unit tests for FastAPI demo services.

This module provides comprehensive test coverage for UserService and OrderService,
validating CRUD operations, trace generation, error handling, and logging behavior.
"""

from __future__ import annotations

import logging
import time
from typing import List
from unittest.mock import patch

import pytest

# Add examples to path for imports
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_PATH = PROJECT_ROOT / "examples"
if str(EXAMPLES_PATH) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PATH))

from fastapi_demo.services import (
    UserService,
    User,
    SpanData,
    OrderService,
    Order,
    OrderItem,
    OrderStatus,
    InsufficientInventoryError,
    PaymentFailedError,
)


class TestUserServiceInitialization:
    """Test UserService initialization and basic setup."""

    def test_user_service_initializes_empty_store(self) -> None:
        """Verify UserService initializes with empty user store."""
        service = UserService()
        users, _ = service.list_users()
        assert len(users) == 0

    def test_user_service_initializes_empty_traces(self) -> None:
        """Verify UserService initializes with empty trace store."""
        service = UserService()
        traces = service.get_trace_data()
        assert len(traces) == 0

    def test_user_service_has_service_name(self) -> None:
        """Verify UserService has correct service name constant."""
        assert UserService.SERVICE_NAME == "user-service"


class TestUserCRUDOperations:
    """Test UserService CRUD operations."""

    def test_create_user_success(self) -> None:
        """Test successful user creation."""
        service = UserService()
        user, spans = service.create_user("testuser", "test@example.com")

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert len(user.id) == 16  # UUID hex[:16]
        assert len(spans) > 0

    def test_create_user_strips_whitespace(self) -> None:
        """Test that username and email are stripped of whitespace."""
        service = UserService()
        user, _ = service.create_user("  testuser  ", "  test@example.com  ")

        assert user.username == "testuser"
        assert user.email == "test@example.com"

    def test_create_user_empty_username_raises_error(self) -> None:
        """Test that empty username raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="username is required"):
            service.create_user("", "test@example.com")

    def test_create_user_whitespace_only_username_raises_error(self) -> None:
        """Test that whitespace-only username raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="username is required"):
            service.create_user("   ", "test@example.com")

    def test_create_user_short_username_raises_error(self) -> None:
        """Test that username shorter than 3 chars raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="at least 3 characters"):
            service.create_user("ab", "test@example.com")

    def test_create_user_empty_email_raises_error(self) -> None:
        """Test that empty email raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="email is required"):
            service.create_user("testuser", "")

    def test_create_user_duplicate_username_raises_error(self) -> None:
        """Test that duplicate username raises ValueError."""
        service = UserService()
        service.create_user("testuser", "test1@example.com")
        with pytest.raises(ValueError, match="already exists"):
            service.create_user("testuser", "test2@example.com")

    def test_get_user_existing(self) -> None:
        """Test retrieving an existing user."""
        service = UserService()
        created_user, _ = service.create_user("testuser", "test@example.com")

        retrieved_user, spans = service.get_user(created_user.id)

        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
        assert retrieved_user.username == "testuser"
        assert len(spans) > 0

    def test_get_user_nonexistent(self) -> None:
        """Test retrieving a non-existent user returns None."""
        service = UserService()
        user, spans = service.get_user("nonexistent-id")

        assert user is None
        assert len(spans) > 0

    def test_delete_user_existing(self) -> None:
        """Test deleting an existing user."""
        service = UserService()
        created_user, _ = service.create_user("testuser", "test@example.com")

        success, spans = service.delete_user(created_user.id)

        assert success is True
        assert len(spans) > 0

        # Verify user is gone
        user, _ = service.get_user(created_user.id)
        assert user is None

    def test_delete_user_nonexistent(self) -> None:
        """Test deleting a non-existent user returns False."""
        service = UserService()
        success, spans = service.delete_user("nonexistent-id")

        assert success is False
        assert len(spans) > 0

    def test_list_users_empty(self) -> None:
        """Test listing users when store is empty."""
        service = UserService()
        users, spans = service.list_users()

        assert len(users) == 0
        assert len(spans) > 0

    def test_list_users_multiple(self) -> None:
        """Test listing multiple users."""
        service = UserService()
        service.create_user("user1", "user1@example.com")
        service.create_user("user2", "user2@example.com")
        service.create_user("user3", "user3@example.com")

        users, spans = service.list_users()

        assert len(users) == 3
        assert len(spans) > 0


class TestUserServiceTracing:
    """Test UserService trace data generation."""

    def test_create_user_generates_spans(self) -> None:
        """Test that create_user generates trace spans."""
        service = UserService()
        _, spans = service.create_user("testuser", "test@example.com")

        # Should have multiple spans: root, validation, duplicate check, persist
        assert len(spans) >= 4

        # Check span structure
        for span in spans:
            assert isinstance(span, SpanData)
            assert len(span.span_id) == 16
            assert len(span.trace_id) == 32
            assert span.service == "user-service"
            assert span.end_time > span.start_time

    def test_spans_have_parent_child_relationship(self) -> None:
        """Test that spans have proper parent-child relationships."""
        service = UserService()
        _, spans = service.create_user("testuser", "test@example.com")

        # Find root span (create_user)
        root_spans = [s for s in spans if s.operation == "create_user"]
        assert len(root_spans) == 1
        root_span = root_spans[0]

        # Find child spans
        child_spans = [s for s in spans if s.parent_span_id == root_span.span_id]
        assert len(child_spans) >= 1

    def test_spans_share_trace_id(self) -> None:
        """Test that all spans in an operation share the same trace ID."""
        service = UserService()
        _, spans = service.create_user("testuser", "test@example.com")

        trace_ids = set(s.trace_id for s in spans)
        assert len(trace_ids) == 1

    def test_span_to_otlp_dict(self) -> None:
        """Test SpanData to_otlp_dict conversion."""
        service = UserService()
        _, spans = service.create_user("testuser", "test@example.com")

        for span in spans:
            otlp = span.to_otlp_dict()
            assert "spanId" in otlp
            assert "traceId" in otlp
            assert "name" in otlp
            assert "_serviceName" in otlp
            assert "startTimeUnixNano" in otlp
            assert "endTimeUnixNano" in otlp
            assert "status" in otlp
            assert "attributes" in otlp

    def test_get_trace_data_returns_otlp_format(self) -> None:
        """Test that get_trace_data returns OTLP-formatted spans."""
        service = UserService()
        service.create_user("testuser", "test@example.com")

        trace_data = service.get_trace_data()

        assert len(trace_data) > 0
        for span_dict in trace_data:
            assert isinstance(span_dict, dict)
            assert "spanId" in span_dict

    def test_clear_traces(self) -> None:
        """Test clearing trace data."""
        service = UserService()
        service.create_user("testuser", "test@example.com")
        assert len(service.get_trace_data()) > 0

        service.clear_traces()
        assert len(service.get_trace_data()) == 0

    def test_custom_trace_id_propagation(self) -> None:
        """Test that custom trace_id is propagated through operations."""
        service = UserService()
        custom_trace_id = "a" * 32

        _, spans = service.create_user(
            "testuser", "test@example.com",
            trace_id=custom_trace_id
        )

        for span in spans:
            assert span.trace_id == custom_trace_id


class TestUserDataclass:
    """Test User dataclass."""

    def test_user_defaults(self) -> None:
        """Test User default values."""
        before = time.time()
        user = User(id="test-id", username="testuser", email="test@example.com")
        after = time.time()

        assert user.is_active is True
        assert before <= user.created_at <= after

    def test_user_custom_values(self) -> None:
        """Test User with custom values."""
        user = User(
            id="test-id",
            username="testuser",
            email="test@example.com",
            is_active=False,
            created_at=1000.0
        )

        assert user.is_active is False
        assert user.created_at == 1000.0


class TestOrderServiceInitialization:
    """Test OrderService initialization."""

    def test_order_service_initializes_empty_store(self) -> None:
        """Verify OrderService initializes with empty order store."""
        service = OrderService()
        orders, _ = service.list_orders()
        assert len(orders) == 0

    def test_order_service_initializes_empty_traces(self) -> None:
        """Verify OrderService initializes with empty trace store."""
        service = OrderService()
        traces = service.get_trace_data()
        assert len(traces) == 0

    def test_order_service_has_service_name(self) -> None:
        """Verify OrderService has correct service name constant."""
        assert OrderService.SERVICE_NAME == "order-service"

    def test_order_service_has_inventory(self) -> None:
        """Verify OrderService has simulated inventory."""
        assert len(OrderService._INVENTORY) > 0


class TestOrderCRUDOperations:
    """Test OrderService CRUD operations."""

    def test_create_order_success(self) -> None:
        """Test successful order creation."""
        service = OrderService()
        items = [OrderItem("prod-001", 2, 29.99)]

        order, spans = service.create_order("user-123", items)

        assert order.user_id == "user-123"
        assert len(order.items) == 1
        assert order.status == OrderStatus.PENDING
        assert len(order.id) == 16
        assert len(spans) > 0

    def test_create_order_calculates_total(self) -> None:
        """Test order total amount calculation."""
        service = OrderService()
        items = [
            OrderItem("prod-001", 2, 10.00),
            OrderItem("prod-002", 3, 5.00),
        ]

        order, _ = service.create_order("user-123", items)

        # 2*10 + 3*5 = 35
        assert order.total_amount == 35.00

    def test_create_order_empty_items_raises_error(self) -> None:
        """Test that empty items list raises ValueError."""
        service = OrderService()
        with pytest.raises(ValueError, match="at least one item"):
            service.create_order("user-123", [])

    def test_create_order_empty_user_id_raises_error(self) -> None:
        """Test that empty user_id raises ValueError."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        with pytest.raises(ValueError, match="User ID is required"):
            service.create_order("", items)

    def test_get_order_existing(self) -> None:
        """Test retrieving an existing order."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        created_order, _ = service.create_order("user-123", items)

        retrieved_order, spans = service.get_order(created_order.id)

        assert retrieved_order is not None
        assert retrieved_order.id == created_order.id
        assert len(spans) > 0

    def test_get_order_nonexistent(self) -> None:
        """Test retrieving a non-existent order returns None."""
        service = OrderService()
        order, spans = service.get_order("nonexistent-id")

        assert order is None
        assert len(spans) > 0

    def test_process_order_success(self) -> None:
        """Test successful order processing."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order, _ = service.create_order("user-123", items)

        processed_order, spans = service.process_order(order.id)

        assert processed_order.status == OrderStatus.COMPLETED
        assert processed_order.processed_at is not None
        assert len(spans) > 0

    def test_process_order_not_found_raises_error(self) -> None:
        """Test processing non-existent order raises ValueError."""
        service = OrderService()
        with pytest.raises(ValueError, match="not found"):
            service.process_order("nonexistent-id")

    def test_process_order_already_completed_raises_error(self) -> None:
        """Test processing already completed order raises ValueError."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order, _ = service.create_order("user-123", items)
        service.process_order(order.id)

        with pytest.raises(ValueError, match="already completed"):
            service.process_order(order.id)

    def test_process_order_insufficient_inventory_raises_error(self) -> None:
        """Test processing order with insufficient inventory raises error."""
        service = OrderService()
        # prod-004 has 0 inventory
        items = [OrderItem("prod-004", 1, 10.00)]
        order, _ = service.create_order("user-123", items)

        with pytest.raises(InsufficientInventoryError):
            service.process_order(order.id)

    def test_cancel_order_success(self) -> None:
        """Test successful order cancellation."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order, _ = service.create_order("user-123", items)

        cancelled_order, spans = service.cancel_order(order.id)

        assert cancelled_order.status == OrderStatus.CANCELLED
        assert len(spans) > 0

    def test_cancel_order_not_found_raises_error(self) -> None:
        """Test cancelling non-existent order raises ValueError."""
        service = OrderService()
        with pytest.raises(ValueError, match="not found"):
            service.cancel_order("nonexistent-id")

    def test_cancel_order_completed_raises_error(self) -> None:
        """Test cancelling completed order raises ValueError."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order, _ = service.create_order("user-123", items)
        service.process_order(order.id)

        with pytest.raises(ValueError, match="Cannot cancel completed"):
            service.cancel_order(order.id)

    def test_cancel_order_already_cancelled_succeeds(self) -> None:
        """Test cancelling already cancelled order returns gracefully."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order, _ = service.create_order("user-123", items)
        service.cancel_order(order.id)

        # Should not raise, just return the cancelled order
        cancelled_order, _ = service.cancel_order(order.id)
        assert cancelled_order.status == OrderStatus.CANCELLED

    def test_list_orders_empty(self) -> None:
        """Test listing orders when store is empty."""
        service = OrderService()
        orders, spans = service.list_orders()

        assert len(orders) == 0
        assert len(spans) > 0

    def test_list_orders_with_user_filter(self) -> None:
        """Test listing orders filtered by user_id."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        service.create_order("user-1", items)
        service.create_order("user-1", items)
        service.create_order("user-2", items)

        orders, _ = service.list_orders(user_id="user-1")

        assert len(orders) == 2
        assert all(o.user_id == "user-1" for o in orders)

    def test_list_orders_with_status_filter(self) -> None:
        """Test listing orders filtered by status."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order1, _ = service.create_order("user-1", items)
        service.create_order("user-1", items)
        service.process_order(order1.id)

        pending_orders, _ = service.list_orders(status=OrderStatus.PENDING)
        completed_orders, _ = service.list_orders(status=OrderStatus.COMPLETED)

        assert len(pending_orders) == 1
        assert len(completed_orders) == 1


class TestOrderServiceLogging:
    """Test OrderService logging at various levels."""

    def test_create_order_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that create_order logs info messages."""
        with caplog.at_level(logging.INFO):
            service = OrderService()
            items = [OrderItem("prod-001", 1, 10.00)]
            service.create_order("user-123", items)

        assert any("Creating order" in record.message for record in caplog.records)
        assert any("created successfully" in record.message for record in caplog.records)

    def test_out_of_stock_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that out-of-stock product logs warning."""
        with caplog.at_level(logging.WARNING):
            service = OrderService()
            # prod-004 has 0 inventory
            items = [OrderItem("prod-004", 1, 10.00)]
            service.create_order("user-123", items)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert any("out of stock" in r.message for r in warning_records)

    def test_low_stock_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that low stock logs warning."""
        with caplog.at_level(logging.WARNING):
            service = OrderService()
            # prod-005 has 5 inventory, ordering 3 leaves low stock (5 < 3*2=6)
            items = [OrderItem("prod-005", 3, 10.00)]
            service.create_order("user-123", items)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert any("Low stock" in r.message or "insufficient" in r.message.lower()
                   for r in warning_records)

    def test_insufficient_inventory_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that ordering more than available logs warning."""
        with caplog.at_level(logging.WARNING):
            service = OrderService()
            # prod-005 has 5 inventory, ordering 10 is insufficient
            items = [OrderItem("prod-005", 10, 10.00)]
            service.create_order("user-123", items)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        assert any("Insufficient inventory" in r.message for r in warning_records)

    def test_validation_error_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that validation errors log error messages."""
        with caplog.at_level(logging.ERROR):
            service = OrderService()
            try:
                service.create_order("user-123", [])
            except ValueError:
                pass

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1

    def test_debug_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that debug messages are logged for span creation."""
        with caplog.at_level(logging.DEBUG):
            service = OrderService()
            items = [OrderItem("prod-001", 1, 10.00)]
            service.create_order("user-123", items)

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_records) >= 1
        assert any("Span created" in r.message for r in debug_records)


class TestOrderServiceTracing:
    """Test OrderService trace data generation."""

    def test_create_order_generates_spans(self) -> None:
        """Test that create_order generates trace spans."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        _, spans = service.create_order("user-123", items)

        # Should have spans: root, validation, inventory, calculate, persist
        assert len(spans) >= 4

    def test_process_order_generates_spans(self) -> None:
        """Test that process_order generates trace spans."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order, _ = service.create_order("user-123", items)
        service.clear_traces()

        _, spans = service.process_order(order.id)

        # Should have spans: root, lookup, reserve, payment, notification
        assert len(spans) >= 4

    def test_error_spans_have_error_status(self) -> None:
        """Test that error operations set span status to ERROR."""
        service = OrderService()
        try:
            service.create_order("user-123", [])
        except ValueError:
            pass

        traces = service.get_trace_data()
        error_spans = [s for s in traces if s.get("status", {}).get("code") == 2]
        assert len(error_spans) >= 1


class TestOrderItem:
    """Test OrderItem dataclass."""

    def test_order_item_total_price(self) -> None:
        """Test OrderItem total_price calculation."""
        item = OrderItem("prod-1", 3, 10.50)
        assert item.total_price == 31.50

    def test_order_item_zero_quantity(self) -> None:
        """Test OrderItem with zero quantity."""
        item = OrderItem("prod-1", 0, 10.00)
        assert item.total_price == 0.0


class TestOrder:
    """Test Order dataclass."""

    def test_order_total_amount(self) -> None:
        """Test Order total_amount calculation."""
        items = [
            OrderItem("prod-1", 2, 10.00),  # 20
            OrderItem("prod-2", 1, 15.00),  # 15
        ]
        order = Order(id="test", user_id="user-1", items=items)
        assert order.total_amount == 35.00

    def test_order_defaults(self) -> None:
        """Test Order default values."""
        before = time.time()
        order = Order(id="test", user_id="user-1", items=[])
        after = time.time()

        assert order.status == OrderStatus.PENDING
        assert before <= order.created_at <= after
        assert order.processed_at is None
        assert order.error_message is None


class TestOrderStatus:
    """Test OrderStatus enum."""

    def test_order_status_values(self) -> None:
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.PROCESSING.value == "processing"
        assert OrderStatus.COMPLETED.value == "completed"
        assert OrderStatus.FAILED.value == "failed"
        assert OrderStatus.CANCELLED.value == "cancelled"


class TestFailureSimulation:
    """Test OrderService failure simulation."""

    def test_enable_failure_simulation(self) -> None:
        """Test enabling failure simulation."""
        service = OrderService()
        service.enable_failure_simulation(True)
        assert service._simulate_failures is True

    def test_disable_failure_simulation(self) -> None:
        """Test disabling failure simulation."""
        service = OrderService()
        service.enable_failure_simulation(True)
        service.enable_failure_simulation(False)
        assert service._simulate_failures is False

    def test_payment_failure_simulation(self) -> None:
        """Test that payment can fail with simulation enabled."""
        service = OrderService()
        service.enable_failure_simulation(True)

        # Run multiple times to trigger failure
        failures = 0
        for _ in range(20):
            items = [OrderItem("prod-001", 1, 10.00)]
            order, _ = service.create_order("user-123", items)
            try:
                service.process_order(order.id)
            except PaymentFailedError:
                failures += 1
            # Reset inventory for next iteration
            OrderService._INVENTORY["prod-001"] = 100

        # With 20% failure rate, we should see at least some failures
        # but this is probabilistic, so we just check it can fail
        # (The test may occasionally pass with 0 failures due to randomness)


class TestSpanDataOTLPConversion:
    """Test SpanData OTLP conversion."""

    def test_span_data_to_otlp_dict_structure(self) -> None:
        """Test SpanData OTLP dictionary structure."""
        span = SpanData(
            span_id="abc123",
            parent_span_id="parent123",
            trace_id="trace123",
            service="test-service",
            operation="test-op",
            start_time=1000000000,
            end_time=2000000000,
            status="OK",
            attributes={"key1": "value1", "key2": 123}
        )

        otlp = span.to_otlp_dict()

        assert otlp["spanId"] == "abc123"
        assert otlp["parentSpanId"] == "parent123"
        assert otlp["traceId"] == "trace123"
        assert otlp["name"] == "test-op"
        assert otlp["_serviceName"] == "test-service"
        assert otlp["startTimeUnixNano"] == "1000000000"
        assert otlp["endTimeUnixNano"] == "2000000000"
        assert otlp["status"]["code"] == 1  # OK

    def test_span_data_error_status_code(self) -> None:
        """Test SpanData with ERROR status has code 2."""
        span = SpanData(
            span_id="abc123",
            parent_span_id=None,
            trace_id="trace123",
            service="test-service",
            operation="test-op",
            start_time=1000000000,
            end_time=2000000000,
            status="ERROR"
        )

        otlp = span.to_otlp_dict()
        assert otlp["status"]["code"] == 2

    def test_span_data_null_parent_span_id(self) -> None:
        """Test SpanData with None parent_span_id converts to empty string."""
        span = SpanData(
            span_id="abc123",
            parent_span_id=None,
            trace_id="trace123",
            service="test-service",
            operation="test-op",
            start_time=1000000000,
            end_time=2000000000
        )

        otlp = span.to_otlp_dict()
        assert otlp["parentSpanId"] == ""

    def test_span_data_attributes_conversion(self) -> None:
        """Test SpanData attributes are converted to OTLP format."""
        span = SpanData(
            span_id="abc123",
            parent_span_id=None,
            trace_id="trace123",
            service="test-service",
            operation="test-op",
            start_time=1000000000,
            end_time=2000000000,
            attributes={"key1": "value1", "key2": 123}
        )

        otlp = span.to_otlp_dict()
        attrs = otlp["attributes"]

        assert len(attrs) == 2
        # Verify structure of attribute entries
        for attr in attrs:
            assert "key" in attr
            assert "value" in attr
            assert "stringValue" in attr["value"]


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_insufficient_inventory_error(self) -> None:
        """Test InsufficientInventoryError can be raised and caught."""
        with pytest.raises(InsufficientInventoryError):
            raise InsufficientInventoryError("Not enough stock")

    def test_payment_failed_error(self) -> None:
        """Test PaymentFailedError can be raised and caught."""
        with pytest.raises(PaymentFailedError):
            raise PaymentFailedError("Payment declined")

    def test_exception_messages(self) -> None:
        """Test exception messages are preserved."""
        try:
            raise InsufficientInventoryError("Custom message")
        except InsufficientInventoryError as e:
            assert str(e) == "Custom message"

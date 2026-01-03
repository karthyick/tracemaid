"""Unit tests for FastAPI demo services.

This module provides comprehensive test coverage for UserService and OrderService,
validating CRUD operations, error handling, and logging behavior.

Note: Services now use standard OpenTelemetry tracing. Traces are automatically
exported by TracemaidExporter - no manual trace retrieval needed.
"""

from __future__ import annotations

import logging
import time
from typing import List

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
        users = service.list_users()
        assert len(users) == 0


class TestUserCRUDOperations:
    """Test UserService CRUD operations."""

    def test_create_user_success(self) -> None:
        """Test successful user creation."""
        service = UserService()
        user = service.create_user("testuser", "test@example.com")

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert len(user.id) == 16  # UUID hex[:16]

    def test_create_user_strips_whitespace(self) -> None:
        """Test that username and email are stripped of whitespace."""
        service = UserService()
        user = service.create_user("  testuser  ", "  test@example.com  ")

        assert user.username == "testuser"
        assert user.email == "test@example.com"

    def test_create_user_empty_username_raises_error(self) -> None:
        """Test that empty username raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="Username is required"):
            service.create_user("", "test@example.com")

    def test_create_user_whitespace_only_username_raises_error(self) -> None:
        """Test that whitespace-only username raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="Username is required"):
            service.create_user("   ", "test@example.com")

    def test_create_user_short_username_raises_error(self) -> None:
        """Test that username shorter than 3 chars raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="at least 3 characters"):
            service.create_user("ab", "test@example.com")

    def test_create_user_empty_email_raises_error(self) -> None:
        """Test that empty email raises ValueError."""
        service = UserService()
        with pytest.raises(ValueError, match="Email is required"):
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
        created_user = service.create_user("testuser", "test@example.com")

        retrieved_user = service.get_user(created_user.id)

        assert retrieved_user is not None
        assert retrieved_user.id == created_user.id
        assert retrieved_user.username == "testuser"

    def test_get_user_nonexistent(self) -> None:
        """Test retrieving a non-existent user returns None."""
        service = UserService()
        user = service.get_user("nonexistent-id")

        assert user is None

    def test_delete_user_existing(self) -> None:
        """Test deleting an existing user."""
        service = UserService()
        created_user = service.create_user("testuser", "test@example.com")

        success = service.delete_user(created_user.id)

        assert success is True

        # Verify user is gone
        user = service.get_user(created_user.id)
        assert user is None

    def test_delete_user_nonexistent(self) -> None:
        """Test deleting a non-existent user returns False."""
        service = UserService()
        success = service.delete_user("nonexistent-id")

        assert success is False

    def test_list_users_empty(self) -> None:
        """Test listing users when store is empty."""
        service = UserService()
        users = service.list_users()

        assert len(users) == 0

    def test_list_users_multiple(self) -> None:
        """Test listing multiple users."""
        service = UserService()
        service.create_user("user1", "user1@example.com")
        service.create_user("user2", "user2@example.com")
        service.create_user("user3", "user3@example.com")

        users = service.list_users()

        assert len(users) == 3


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
        orders = service.list_orders()
        assert len(orders) == 0

    def test_order_service_has_inventory(self) -> None:
        """Verify OrderService has simulated inventory."""
        assert len(OrderService._INVENTORY) > 0


class TestOrderCRUDOperations:
    """Test OrderService CRUD operations."""

    def test_create_order_success(self) -> None:
        """Test successful order creation."""
        service = OrderService()
        items = [OrderItem("prod-001", 2, 29.99)]

        order = service.create_order("user-123", items)

        assert order.user_id == "user-123"
        assert len(order.items) == 1
        assert order.status == OrderStatus.PENDING
        assert len(order.id) == 16

    def test_create_order_calculates_total(self) -> None:
        """Test order total amount calculation."""
        service = OrderService()
        items = [
            OrderItem("prod-001", 2, 10.00),
            OrderItem("prod-002", 3, 5.00),
        ]

        order = service.create_order("user-123", items)

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
        created_order = service.create_order("user-123", items)

        retrieved_order = service.get_order(created_order.id)

        assert retrieved_order is not None
        assert retrieved_order.id == created_order.id

    def test_get_order_nonexistent(self) -> None:
        """Test retrieving a non-existent order returns None."""
        service = OrderService()
        order = service.get_order("nonexistent-id")

        assert order is None

    def test_process_order_success(self) -> None:
        """Test successful order processing."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order = service.create_order("user-123", items)

        processed_order = service.process_order(order.id)

        assert processed_order.status == OrderStatus.COMPLETED
        assert processed_order.processed_at is not None

    def test_process_order_not_found_raises_error(self) -> None:
        """Test processing non-existent order raises ValueError."""
        service = OrderService()
        with pytest.raises(ValueError, match="not found"):
            service.process_order("nonexistent-id")

    def test_process_order_already_completed_raises_error(self) -> None:
        """Test processing already completed order raises ValueError."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order = service.create_order("user-123", items)
        service.process_order(order.id)

        with pytest.raises(ValueError, match="already completed"):
            service.process_order(order.id)

    def test_process_order_insufficient_inventory_raises_error(self) -> None:
        """Test processing order with insufficient inventory raises error."""
        service = OrderService()
        # prod-004 has 0 inventory
        items = [OrderItem("prod-004", 1, 10.00)]
        order = service.create_order("user-123", items)

        with pytest.raises(InsufficientInventoryError):
            service.process_order(order.id)

    def test_cancel_order_success(self) -> None:
        """Test successful order cancellation."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order = service.create_order("user-123", items)

        cancelled_order = service.cancel_order(order.id)

        assert cancelled_order.status == OrderStatus.CANCELLED

    def test_cancel_order_not_found_raises_error(self) -> None:
        """Test cancelling non-existent order raises ValueError."""
        service = OrderService()
        with pytest.raises(ValueError, match="not found"):
            service.cancel_order("nonexistent-id")

    def test_cancel_order_completed_raises_error(self) -> None:
        """Test cancelling completed order raises ValueError."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order = service.create_order("user-123", items)
        service.process_order(order.id)

        with pytest.raises(ValueError, match="Cannot cancel completed"):
            service.cancel_order(order.id)

    def test_cancel_order_already_cancelled_succeeds(self) -> None:
        """Test cancelling already cancelled order returns gracefully."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        order = service.create_order("user-123", items)
        service.cancel_order(order.id)

        # Should not raise, just return the cancelled order
        cancelled_order = service.cancel_order(order.id)
        assert cancelled_order.status == OrderStatus.CANCELLED

    def test_list_orders_empty(self) -> None:
        """Test listing orders when store is empty."""
        service = OrderService()
        orders = service.list_orders()

        assert len(orders) == 0

    def test_list_orders_with_user_filter(self) -> None:
        """Test listing orders filtered by user_id."""
        service = OrderService()
        items = [OrderItem("prod-001", 1, 10.00)]
        service.create_order("user-1", items)
        service.create_order("user-1", items)
        service.create_order("user-2", items)

        orders = service.list_orders(user_id="user-1")

        assert len(orders) == 2
        assert all(o.user_id == "user-1" for o in orders)


class TestOrderServiceLogging:
    """Test OrderService logging at various levels."""

    def test_create_order_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that create_order logs info messages."""
        with caplog.at_level(logging.INFO):
            service = OrderService()
            items = [OrderItem("prod-001", 1, 10.00)]
            service.create_order("user-123", items)

        assert any("Creating order" in record.message for record in caplog.records)
        assert any("Order created:" in record.message for record in caplog.records)

    def test_process_order_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that process_order logs info messages."""
        with caplog.at_level(logging.INFO):
            service = OrderService()
            items = [OrderItem("prod-001", 1, 10.00)]
            order = service.create_order("user-123", items)
            service.process_order(order.id)

        assert any("Processing order" in record.message for record in caplog.records)
        assert any("processed successfully" in record.message for record in caplog.records)

    def test_cancel_order_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that cancel_order logs info messages."""
        with caplog.at_level(logging.INFO):
            service = OrderService()
            items = [OrderItem("prod-001", 1, 10.00)]
            order = service.create_order("user-123", items)
            service.cancel_order(order.id)

        assert any("cancelled" in record.message for record in caplog.records)


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

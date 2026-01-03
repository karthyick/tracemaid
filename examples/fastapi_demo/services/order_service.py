"""
Order service module with manual tracing for comprehensive span generation.

This module demonstrates creating nested spans for complex business operations,
resulting in traces with 10+ spans to test Tracemaid's span selection algorithm.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OrderItem:
    """Represents an item in an order."""
    product_id: str
    quantity: int
    unit_price: float

    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price


@dataclass
class Order:
    """Represents an order entity."""
    id: str
    user_id: str
    items: List[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def total_amount(self) -> float:
        return sum(item.total_price for item in self.items)


class InsufficientInventoryError(Exception):
    """Raised when there's insufficient inventory."""
    pass


class PaymentFailedError(Exception):
    """Raised when payment processing fails."""
    pass


class OrderService:
    """Service for order operations - no manual tracing code."""

    _INVENTORY: Dict[str, int] = {
        "prod-001": 100,
        "prod-002": 50,
        "prod-003": 25,
        "prod-004": 0,
        "prod-005": 5,
    }

    def __init__(self) -> None:
        self._orders: Dict[str, Order] = {}
        logger.info("OrderService initialized")

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def create_order(self, user_id: str, items: List[OrderItem]) -> Order:
        """Create a new order."""
        logger.info("Creating order for user %s", user_id)

        if not items:
            raise ValueError("Order must contain at least one item")
        if not user_id or not user_id.strip():
            raise ValueError("User ID is required")

        order_id = self._generate_id()
        order = Order(
            id=order_id,
            user_id=user_id.strip(),
            items=items,
            status=OrderStatus.PENDING
        )
        self._orders[order_id] = order

        logger.info("Order created: %s (total: $%.2f)", order_id, order.total_amount)
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by ID."""
        return self._orders.get(order_id)

    def process_order(self, order_id: str) -> Order:
        """Process an order with comprehensive tracing."""
        with tracer.start_as_current_span("process_order") as span:
            span.set_attribute("order.id", order_id)
            logger.info("Processing order %s", order_id)

            # Step 1: Validate order
            order = self._validate_order(order_id)
            span.set_attribute("order.total", order.total_amount)
            span.set_attribute("order.items_count", len(order.items))

            # Step 2: Check inventory for all items
            self._check_inventory(order)

            # Step 3: Reserve inventory
            self._reserve_inventory(order)

            # Step 4: Process payment
            self._process_payment(order)

            # Step 5: Update order status
            self._update_order_status(order)

            # Step 6: Send notifications
            self._send_notifications(order)

            logger.info("Order %s processed successfully", order_id)
            return order

    def _validate_order(self, order_id: str) -> Order:
        """Validate order exists and is processable."""
        with tracer.start_as_current_span("validate_order") as span:
            span.set_attribute("order.id", order_id)
            order = self._orders.get(order_id)
            if not order:
                span.set_attribute("error", True)
                raise ValueError(f"Order '{order_id}' not found")

            if order.status in (OrderStatus.COMPLETED, OrderStatus.FAILED, OrderStatus.CANCELLED):
                span.set_attribute("error", True)
                raise ValueError(f"Order '{order_id}' is already {order.status.value}")

            span.set_attribute("validation.passed", True)
            return order

    def _check_inventory(self, order: Order) -> None:
        """Check inventory for all items."""
        with tracer.start_as_current_span("check_inventory") as span:
            span.set_attribute("items.count", len(order.items))
            for item in order.items:
                self._check_item_inventory(item)

    def _check_item_inventory(self, item: OrderItem) -> None:
        """Check inventory for a single item."""
        with tracer.start_as_current_span("check_item_inventory") as span:
            span.set_attribute("product.id", item.product_id)
            span.set_attribute("quantity.requested", item.quantity)
            available = self._INVENTORY.get(item.product_id, 0)
            span.set_attribute("quantity.available", available)

            if available < item.quantity:
                span.set_attribute("error", True)
                raise InsufficientInventoryError(f"Insufficient inventory: {item.product_id}")

    def _reserve_inventory(self, order: Order) -> None:
        """Reserve inventory for order items."""
        with tracer.start_as_current_span("reserve_inventory") as span:
            span.set_attribute("order.id", order.id)
            for item in order.items:
                self._reserve_item(item)

    def _reserve_item(self, item: OrderItem) -> None:
        """Reserve inventory for a single item."""
        with tracer.start_as_current_span("reserve_item") as span:
            span.set_attribute("product.id", item.product_id)
            span.set_attribute("quantity", item.quantity)
            self._INVENTORY[item.product_id] -= item.quantity
            span.set_attribute("inventory.remaining", self._INVENTORY[item.product_id])

    def _process_payment(self, order: Order) -> None:
        """Process payment for order."""
        with tracer.start_as_current_span("process_payment") as span:
            span.set_attribute("order.id", order.id)
            span.set_attribute("amount", order.total_amount)
            # Simulate payment processing
            time.sleep(0.01)
            span.set_attribute("payment.status", "approved")

    def _update_order_status(self, order: Order) -> None:
        """Update order status to completed."""
        with tracer.start_as_current_span("update_order_status") as span:
            span.set_attribute("order.id", order.id)
            span.set_attribute("status.old", order.status.value)
            order.status = OrderStatus.COMPLETED
            order.processed_at = time.time()
            span.set_attribute("status.new", order.status.value)

    def _send_notifications(self, order: Order) -> None:
        """Send order notifications."""
        with tracer.start_as_current_span("send_notifications") as span:
            span.set_attribute("order.id", order.id)
            self._send_email_notification(order)
            self._send_sms_notification(order)

    def _send_email_notification(self, order: Order) -> None:
        """Send email notification."""
        with tracer.start_as_current_span("send_email") as span:
            span.set_attribute("order.id", order.id)
            span.set_attribute("channel", "email")
            time.sleep(0.005)

    def _send_sms_notification(self, order: Order) -> None:
        """Send SMS notification."""
        with tracer.start_as_current_span("send_sms") as span:
            span.set_attribute("order.id", order.id)
            span.set_attribute("channel", "sms")
            time.sleep(0.005)

    def cancel_order(self, order_id: str, reason: str = "customer_request") -> Order:
        """Cancel an order."""
        order = self._orders.get(order_id)
        if not order:
            raise ValueError(f"Order '{order_id}' not found")

        if order.status == OrderStatus.COMPLETED:
            raise ValueError(f"Cannot cancel completed order '{order_id}'")

        order.status = OrderStatus.CANCELLED
        logger.info("Order %s cancelled", order_id)
        return order

    def list_orders(self, user_id: Optional[str] = None) -> List[Order]:
        """List orders."""
        orders = list(self._orders.values())
        if user_id:
            orders = [o for o in orders if o.user_id == user_id]
        return orders

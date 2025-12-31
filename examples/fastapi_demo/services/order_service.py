"""
Order service module demonstrating tracemaid integration with various log levels.

This module provides an OrderService class that simulates order operations
and generates OpenTelemetry-compatible trace data. It specifically demonstrates
warning and error level logs for different scenarios.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from examples.fastapi_demo.services.tracing import SpanData

# Configure module logger
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OrderItem:
    """Represents an item in an order.

    Attributes:
        product_id: Unique identifier for the product
        quantity: Number of items ordered
        unit_price: Price per unit
    """
    product_id: str
    quantity: int
    unit_price: float

    @property
    def total_price(self) -> float:
        """Calculate total price for this item."""
        return self.quantity * self.unit_price


@dataclass
class Order:
    """Represents an order entity.

    Attributes:
        id: Unique identifier for the order
        user_id: ID of the user who placed the order
        items: List of order items
        status: Current order status
        created_at: Timestamp of creation
        processed_at: Timestamp when order was processed
        error_message: Error message if order failed
    """
    id: str
    user_id: str
    items: List[OrderItem]
    status: OrderStatus = OrderStatus.PENDING
    created_at: float = field(default_factory=time.time)
    processed_at: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def total_amount(self) -> float:
        """Calculate total order amount."""
        return sum(item.total_price for item in self.items)


class InsufficientInventoryError(Exception):
    """Raised when there's insufficient inventory for an order."""
    pass


class PaymentFailedError(Exception):
    """Raised when payment processing fails."""
    pass


class OrderProcessingError(Exception):
    """Raised when order processing fails."""
    pass


class OrderService:
    """Service for order operations with comprehensive trace data generation.

    This service simulates order operations and generates OpenTelemetry-compatible
    trace data. It demonstrates various log levels including debug, info, warning,
    and error scenarios.

    Example:
        >>> service = OrderService()
        >>> items = [OrderItem("prod-1", 2, 29.99)]
        >>> order, traces = service.create_order("user-123", items)
        >>> # traces can be parsed by tracemaid.OTelParser
        >>> from tracemaid import OTelParser
        >>> parser = OTelParser()
        >>> trace = parser.parse_otlp({"spans": [t.to_otlp_dict() for t in traces]})
    """

    SERVICE_NAME = "order-service"

    # Simulated inventory (product_id -> available quantity)
    _INVENTORY: Dict[str, int] = {
        "prod-001": 100,
        "prod-002": 50,
        "prod-003": 25,
        "prod-004": 0,  # Out of stock - will trigger warnings/errors
        "prod-005": 5,  # Low stock - will trigger warnings
    }

    def __init__(self) -> None:
        """Initialize the OrderService with an in-memory order store."""
        self._orders: Dict[str, Order] = {}
        self._traces: List[SpanData] = []
        self._simulate_failures: bool = False
        logger.info(
            "OrderService initialized",
            extra={"service": self.SERVICE_NAME}
        )

    def enable_failure_simulation(self, enabled: bool = True) -> None:
        """Enable or disable random failure simulation.

        Args:
            enabled: Whether to enable failure simulation
        """
        self._simulate_failures = enabled
        logger.info(
            "Failure simulation %s",
            "enabled" if enabled else "disabled"
        )

    def _generate_span_id(self) -> str:
        """Generate a unique span ID.

        Returns:
            16-character hex string span ID
        """
        return uuid.uuid4().hex[:16]

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID.

        Returns:
            32-character hex string trace ID
        """
        return uuid.uuid4().hex

    def _get_current_time_nanos(self) -> int:
        """Get current time in nanoseconds.

        Returns:
            Current Unix timestamp in nanoseconds
        """
        return int(time.time() * 1_000_000_000)

    def _create_span(
        self,
        operation: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> SpanData:
        """Create a new span for tracing.

        Args:
            operation: Name of the operation being traced
            parent_span_id: ID of the parent span
            trace_id: Trace ID (generated if not provided)
            attributes: Additional attributes for the span

        Returns:
            New SpanData instance
        """
        span = SpanData(
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id,
            trace_id=trace_id or self._generate_trace_id(),
            service=self.SERVICE_NAME,
            operation=operation,
            start_time=self._get_current_time_nanos(),
            end_time=0,
            attributes=attributes or {}
        )
        logger.debug(
            "Span created: %s",
            operation,
            extra={
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "operation": operation
            }
        )
        return span

    def _complete_span(
        self,
        span: SpanData,
        status: str = "OK",
        additional_attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete a span with end time and status.

        Args:
            span: The span to complete
            status: Final status (OK or ERROR)
            additional_attributes: Additional attributes to add
        """
        span.end_time = self._get_current_time_nanos()
        span.status = status
        if additional_attributes:
            span.attributes.update(additional_attributes)

        self._traces.append(span)

        duration_ms = (span.end_time - span.start_time) / 1_000_000
        log_extra = {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "status": status,
            "duration_ms": duration_ms
        }

        if status == "ERROR":
            logger.error(
                "Span completed with error: %s (%.2fms)",
                span.operation,
                duration_ms,
                extra=log_extra
            )
        else:
            logger.info(
                "Span completed: %s (%.2fms)",
                span.operation,
                duration_ms,
                extra=log_extra
            )

    def create_order(
        self,
        user_id: str,
        items: List[OrderItem],
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[Order, List[SpanData]]:
        """Create a new order.

        Args:
            user_id: ID of the user placing the order
            items: List of order items
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (created Order, list of trace spans)

        Raises:
            ValueError: If items list is empty
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        root_span = self._create_span(
            operation="create_order",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={
                "user.id": user_id,
                "order.items_count": str(len(items)),
                "operation.type": "write"
            }
        )

        logger.info(
            "Creating order for user %s with %d items",
            user_id,
            len(items),
            extra={
                "user_id": user_id,
                "items_count": len(items),
                "trace_id": trace_id
            }
        )

        # Validation span
        validation_span = self._create_span(
            operation="validate_order",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"validation.type": "order_input"}
        )

        if not items:
            self._complete_span(
                validation_span,
                status="ERROR",
                additional_attributes={"validation.error": "empty_items"}
            )
            spans.append(validation_span)

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={"error.type": "validation_error"}
            )
            spans.append(root_span)

            logger.error(
                "Order validation failed: empty items list",
                extra={"user_id": user_id}
            )
            raise ValueError("Order must contain at least one item")

        if not user_id or not user_id.strip():
            self._complete_span(
                validation_span,
                status="ERROR",
                additional_attributes={"validation.error": "invalid_user_id"}
            )
            spans.append(validation_span)

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={"error.type": "validation_error"}
            )
            spans.append(root_span)

            logger.error("Order validation failed: invalid user ID")
            raise ValueError("User ID is required")

        time.sleep(0.001)
        self._complete_span(validation_span, status="OK")
        spans.append(validation_span)

        # Check inventory span
        inventory_span = self._create_span(
            operation="check_inventory",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"check.type": "availability"}
        )

        inventory_warnings: List[str] = []
        for item in items:
            available = self._INVENTORY.get(item.product_id, 0)
            if available == 0:
                # Warning for out of stock
                logger.warning(
                    "Product %s is out of stock",
                    item.product_id,
                    extra={"product_id": item.product_id, "requested": item.quantity}
                )
                inventory_warnings.append(f"{item.product_id}: out of stock")
            elif available < item.quantity:
                # Warning for insufficient quantity
                logger.warning(
                    "Insufficient inventory for product %s: requested %d, available %d",
                    item.product_id,
                    item.quantity,
                    available,
                    extra={
                        "product_id": item.product_id,
                        "requested": item.quantity,
                        "available": available
                    }
                )
                inventory_warnings.append(
                    f"{item.product_id}: insufficient (need {item.quantity}, have {available})"
                )
            elif available < item.quantity * 2:
                # Warning for low stock
                logger.warning(
                    "Low stock warning for product %s: only %d remaining after order",
                    item.product_id,
                    available - item.quantity,
                    extra={
                        "product_id": item.product_id,
                        "remaining": available - item.quantity
                    }
                )

        time.sleep(0.003)  # Simulate inventory check

        if inventory_warnings:
            self._complete_span(
                inventory_span,
                status="OK",  # Still OK, but with warnings
                additional_attributes={
                    "inventory.warnings": str(len(inventory_warnings)),
                    "inventory.warning_details": "; ".join(inventory_warnings)
                }
            )
        else:
            self._complete_span(
                inventory_span,
                status="OK",
                additional_attributes={"inventory.available": "true"}
            )
        spans.append(inventory_span)

        # Calculate total span
        calculate_span = self._create_span(
            operation="calculate_order_total",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"calculation.type": "order_total"}
        )

        total_amount = sum(item.total_price for item in items)
        time.sleep(0.001)

        self._complete_span(
            calculate_span,
            status="OK",
            additional_attributes={"order.total": f"{total_amount:.2f}"}
        )
        spans.append(calculate_span)

        # Create order
        persist_span = self._create_span(
            operation="persist_order",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"persistence.type": "memory"}
        )

        order_id = self._generate_span_id()
        order = Order(
            id=order_id,
            user_id=user_id.strip(),
            items=items,
            status=OrderStatus.PENDING
        )

        self._orders[order_id] = order
        time.sleep(0.002)

        self._complete_span(
            persist_span,
            status="OK",
            additional_attributes={"order.id": order_id}
        )
        spans.append(persist_span)

        # Complete root span
        self._complete_span(
            root_span,
            status="OK",
            additional_attributes={
                "order.id": order_id,
                "order.total": f"{total_amount:.2f}",
                "order.status": OrderStatus.PENDING.value
            }
        )
        spans.append(root_span)

        logger.info(
            "Order created successfully: %s (total: $%.2f)",
            order_id,
            total_amount,
            extra={
                "order_id": order_id,
                "total_amount": total_amount,
                "user_id": user_id
            }
        )

        return order, spans

    def get_order(
        self,
        order_id: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[Optional[Order], List[SpanData]]:
        """Retrieve an order by ID.

        Args:
            order_id: The ID of the order to retrieve
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (Order or None if not found, list of trace spans)
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        root_span = self._create_span(
            operation="get_order",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={"order.id": order_id, "operation.type": "read"}
        )

        logger.info(
            "Getting order %s",
            order_id,
            extra={"order_id": order_id, "trace_id": trace_id}
        )

        # Lookup span
        lookup_span = self._create_span(
            operation="lookup_order",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"lookup.type": "by_id"}
        )

        time.sleep(0.001)
        order = self._orders.get(order_id)

        self._complete_span(
            lookup_span,
            status="OK",
            additional_attributes={"order.found": str(order is not None).lower()}
        )
        spans.append(lookup_span)

        if order:
            self._complete_span(
                root_span,
                status="OK",
                additional_attributes={
                    "order.found": "true",
                    "order.status": order.status.value
                }
            )
            logger.info(
                "Order found: %s (status: %s)",
                order_id,
                order.status.value
            )
        else:
            self._complete_span(
                root_span,
                status="OK",
                additional_attributes={"order.found": "false"}
            )
            logger.warning(
                "Order not found: %s",
                order_id,
                extra={"order_id": order_id}
            )

        spans.append(root_span)
        return order, spans

    def process_order(
        self,
        order_id: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[Order, List[SpanData]]:
        """Process an order (payment, inventory deduction, etc.).

        This method demonstrates various error scenarios:
        - Order not found
        - Order already processed
        - Insufficient inventory
        - Payment failures
        - General processing errors

        Args:
            order_id: The ID of the order to process
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (processed Order, list of trace spans)

        Raises:
            ValueError: If order not found or already processed
            InsufficientInventoryError: If inventory is insufficient
            PaymentFailedError: If payment processing fails
            OrderProcessingError: For general processing failures
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        root_span = self._create_span(
            operation="process_order",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={"order.id": order_id, "operation.type": "process"}
        )

        logger.info(
            "Processing order %s",
            order_id,
            extra={"order_id": order_id, "trace_id": trace_id}
        )

        # Lookup order
        lookup_span = self._create_span(
            operation="lookup_order_for_processing",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"lookup.purpose": "processing"}
        )

        time.sleep(0.001)
        order = self._orders.get(order_id)

        if not order:
            self._complete_span(
                lookup_span,
                status="ERROR",
                additional_attributes={"error": "order_not_found"}
            )
            spans.append(lookup_span)

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={"error.type": "not_found"}
            )
            spans.append(root_span)

            logger.error(
                "Order not found for processing: %s",
                order_id,
                extra={"order_id": order_id}
            )
            raise ValueError(f"Order '{order_id}' not found")

        self._complete_span(
            lookup_span,
            status="OK",
            additional_attributes={"order.status": order.status.value}
        )
        spans.append(lookup_span)

        # Check if already processed
        if order.status in (OrderStatus.COMPLETED, OrderStatus.FAILED, OrderStatus.CANCELLED):
            logger.warning(
                "Order %s is already in terminal state: %s",
                order_id,
                order.status.value,
                extra={"order_id": order_id, "status": order.status.value}
            )

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={
                    "error.type": "already_processed",
                    "order.status": order.status.value
                }
            )
            spans.append(root_span)

            raise ValueError(f"Order '{order_id}' is already {order.status.value}")

        # Update status to processing
        order.status = OrderStatus.PROCESSING

        # Reserve inventory
        reserve_span = self._create_span(
            operation="reserve_inventory",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"reservation.type": "order_items"}
        )

        time.sleep(0.005)  # Simulate inventory reservation

        insufficient_items: List[str] = []
        for item in order.items:
            available = self._INVENTORY.get(item.product_id, 0)
            if available < item.quantity:
                insufficient_items.append(
                    f"{item.product_id} (need {item.quantity}, have {available})"
                )

        if insufficient_items:
            self._complete_span(
                reserve_span,
                status="ERROR",
                additional_attributes={
                    "error": "insufficient_inventory",
                    "insufficient_items": "; ".join(insufficient_items)
                }
            )
            spans.append(reserve_span)

            order.status = OrderStatus.FAILED
            order.error_message = f"Insufficient inventory: {', '.join(insufficient_items)}"

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={
                    "error.type": "inventory_error",
                    "order.status": OrderStatus.FAILED.value
                }
            )
            spans.append(root_span)

            logger.error(
                "Order %s failed: insufficient inventory for %s",
                order_id,
                ", ".join(insufficient_items),
                extra={
                    "order_id": order_id,
                    "insufficient_items": insufficient_items
                }
            )
            raise InsufficientInventoryError(
                f"Insufficient inventory: {', '.join(insufficient_items)}"
            )

        # Deduct inventory
        for item in order.items:
            self._INVENTORY[item.product_id] = (
                self._INVENTORY.get(item.product_id, 0) - item.quantity
            )

        self._complete_span(
            reserve_span,
            status="OK",
            additional_attributes={"inventory.reserved": "true"}
        )
        spans.append(reserve_span)

        # Process payment
        payment_span = self._create_span(
            operation="process_payment",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={
                "payment.amount": f"{order.total_amount:.2f}",
                "payment.currency": "USD"
            }
        )

        time.sleep(0.010)  # Simulate payment processing (slow operation)

        # Simulate random payment failures if enabled
        if self._simulate_failures and random.random() < 0.2:
            self._complete_span(
                payment_span,
                status="ERROR",
                additional_attributes={
                    "error": "payment_declined",
                    "error.code": "INSUFFICIENT_FUNDS"
                }
            )
            spans.append(payment_span)

            # Rollback inventory
            rollback_span = self._create_span(
                operation="rollback_inventory",
                parent_span_id=root_span.span_id,
                trace_id=trace_id,
                attributes={"rollback.reason": "payment_failed"}
            )

            for item in order.items:
                self._INVENTORY[item.product_id] = (
                    self._INVENTORY.get(item.product_id, 0) + item.quantity
                )

            time.sleep(0.002)
            self._complete_span(rollback_span, status="OK")
            spans.append(rollback_span)

            order.status = OrderStatus.FAILED
            order.error_message = "Payment declined"

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={
                    "error.type": "payment_failed",
                    "order.status": OrderStatus.FAILED.value
                }
            )
            spans.append(root_span)

            logger.error(
                "Order %s payment failed",
                order_id,
                extra={"order_id": order_id, "amount": order.total_amount}
            )
            raise PaymentFailedError("Payment declined: INSUFFICIENT_FUNDS")

        self._complete_span(
            payment_span,
            status="OK",
            additional_attributes={"payment.status": "approved"}
        )
        spans.append(payment_span)

        # Send confirmation
        notification_span = self._create_span(
            operation="send_confirmation",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"notification.type": "email"}
        )

        time.sleep(0.003)  # Simulate sending notification

        # Simulate occasional notification failures (warning, not error)
        if self._simulate_failures and random.random() < 0.1:
            logger.warning(
                "Failed to send order confirmation for %s, will retry",
                order_id,
                extra={"order_id": order_id, "retry_scheduled": True}
            )
            self._complete_span(
                notification_span,
                status="OK",  # Still OK because order succeeded
                additional_attributes={
                    "notification.status": "failed",
                    "notification.retry_scheduled": "true"
                }
            )
        else:
            self._complete_span(
                notification_span,
                status="OK",
                additional_attributes={"notification.status": "sent"}
            )
        spans.append(notification_span)

        # Complete order
        order.status = OrderStatus.COMPLETED
        order.processed_at = time.time()

        self._complete_span(
            root_span,
            status="OK",
            additional_attributes={
                "order.status": OrderStatus.COMPLETED.value,
                "order.processed": "true"
            }
        )
        spans.append(root_span)

        logger.info(
            "Order %s processed successfully (total: $%.2f)",
            order_id,
            order.total_amount,
            extra={
                "order_id": order_id,
                "total_amount": order.total_amount
            }
        )

        return order, spans

    def cancel_order(
        self,
        order_id: str,
        reason: str = "customer_request",
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[Order, List[SpanData]]:
        """Cancel an order.

        Args:
            order_id: The ID of the order to cancel
            reason: Reason for cancellation
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (cancelled Order, list of trace spans)

        Raises:
            ValueError: If order not found or cannot be cancelled
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        root_span = self._create_span(
            operation="cancel_order",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={
                "order.id": order_id,
                "cancellation.reason": reason,
                "operation.type": "cancel"
            }
        )

        logger.info(
            "Cancelling order %s (reason: %s)",
            order_id,
            reason,
            extra={"order_id": order_id, "reason": reason}
        )

        order = self._orders.get(order_id)
        if not order:
            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={"error.type": "not_found"}
            )
            spans.append(root_span)

            logger.error("Order not found for cancellation: %s", order_id)
            raise ValueError(f"Order '{order_id}' not found")

        if order.status == OrderStatus.COMPLETED:
            logger.warning(
                "Cannot cancel completed order %s",
                order_id,
                extra={"order_id": order_id, "status": order.status.value}
            )

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={
                    "error.type": "cannot_cancel",
                    "order.status": order.status.value
                }
            )
            spans.append(root_span)

            raise ValueError(f"Cannot cancel completed order '{order_id}'")

        if order.status == OrderStatus.CANCELLED:
            logger.warning(
                "Order %s is already cancelled",
                order_id,
                extra={"order_id": order_id}
            )

            self._complete_span(
                root_span,
                status="OK",
                additional_attributes={"already_cancelled": "true"}
            )
            spans.append(root_span)

            return order, spans

        order.status = OrderStatus.CANCELLED
        time.sleep(0.002)

        self._complete_span(
            root_span,
            status="OK",
            additional_attributes={
                "order.status": OrderStatus.CANCELLED.value,
                "order.cancelled": "true"
            }
        )
        spans.append(root_span)

        logger.info(
            "Order %s cancelled successfully",
            order_id,
            extra={"order_id": order_id, "reason": reason}
        )

        return order, spans

    def get_trace_data(self) -> List[Dict[str, Any]]:
        """Get all collected trace data in OTLP format.

        Returns:
            List of span dictionaries in OTLP format
        """
        return [span.to_otlp_dict() for span in self._traces]

    def clear_traces(self) -> None:
        """Clear all collected trace data."""
        self._traces.clear()
        logger.debug("Order service trace data cleared")

    def list_orders(
        self,
        user_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[List[Order], List[SpanData]]:
        """List orders with optional filtering.

        Args:
            user_id: Optional user ID to filter by
            status: Optional status to filter by
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (list of Orders, list of trace spans)
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        root_span = self._create_span(
            operation="list_orders",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={
                "filter.user_id": user_id or "none",
                "filter.status": status.value if status else "none",
                "operation.type": "read"
            }
        )

        logger.info(
            "Listing orders",
            extra={
                "user_id_filter": user_id,
                "status_filter": status.value if status else None
            }
        )

        time.sleep(0.003)  # Simulate query

        orders = list(self._orders.values())
        if user_id:
            orders = [o for o in orders if o.user_id == user_id]
        if status:
            orders = [o for o in orders if o.status == status]

        self._complete_span(
            root_span,
            status="OK",
            additional_attributes={"orders.count": str(len(orders))}
        )
        spans.append(root_span)

        logger.info("Listed %d orders", len(orders))

        return orders, spans

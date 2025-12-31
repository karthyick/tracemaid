"""
Order routes module for FastAPI demo application.

This module provides API endpoints for order operations with tracemaid
request tracing, correlation ID propagation, and comprehensive error handling
that demonstrates tracemaid error logging capabilities.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from examples.fastapi_demo.services import (
    OrderService,
    Order,
    OrderItem,
    OrderStatus,
    SpanData,
    InsufficientInventoryError,
    PaymentFailedError,
    OrderProcessingError,
)

# Configure module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/orders", tags=["orders"])

# Service instance (in production, use dependency injection)
_order_service = OrderService()


class OrderItemRequest(BaseModel):
    """Request model for an order item.

    Attributes:
        product_id: Unique identifier for the product
        quantity: Number of items to order (minimum 1)
        unit_price: Price per unit
    """
    product_id: str = Field(..., description="Product identifier")
    quantity: int = Field(..., ge=1, description="Quantity to order")
    unit_price: float = Field(..., gt=0, description="Price per unit")


class OrderCreateRequest(BaseModel):
    """Request model for creating an order.

    Attributes:
        user_id: ID of the user placing the order
        items: List of order items (at least one required)
    """
    user_id: str = Field(..., min_length=1, description="User ID")
    items: List[OrderItemRequest] = Field(
        ...,
        min_length=1,
        description="List of items to order"
    )


class OrderItemResponse(BaseModel):
    """Response model for an order item.

    Attributes:
        product_id: Product identifier
        quantity: Quantity ordered
        unit_price: Price per unit
        total_price: Total price for this item
    """
    product_id: str
    quantity: int
    unit_price: float
    total_price: float


class OrderResponse(BaseModel):
    """Response model for order data.

    Attributes:
        id: Unique order identifier
        user_id: ID of the user who placed the order
        items: List of order items
        status: Current order status
        total_amount: Total order amount
        created_at: Unix timestamp of creation
        processed_at: Unix timestamp when processed (if applicable)
        error_message: Error message if order failed
    """
    id: str
    user_id: str
    items: List[OrderItemResponse]
    status: str
    total_amount: float
    created_at: float
    processed_at: Optional[float] = None
    error_message: Optional[str] = None


class OrderListResponse(BaseModel):
    """Response model for listing orders.

    Attributes:
        orders: List of order objects
        total: Total count of orders
    """
    orders: List[OrderResponse]
    total: int


class TraceResponse(BaseModel):
    """Response model including trace data.

    Attributes:
        data: The response data
        trace_id: The trace ID for this request
        spans_count: Number of spans generated
    """
    data: Any
    trace_id: str
    spans_count: int


class ProcessOrderRequest(BaseModel):
    """Request model for processing an order.

    Attributes:
        simulate_failures: Whether to enable random failure simulation
    """
    simulate_failures: bool = Field(
        default=False,
        description="Enable random failure simulation for testing"
    )


class CancelOrderRequest(BaseModel):
    """Request model for cancelling an order.

    Attributes:
        reason: Reason for cancellation
    """
    reason: str = Field(
        default="customer_request",
        description="Reason for cancellation"
    )


def _get_trace_context(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Extract trace context from request headers.

    Args:
        request: The incoming FastAPI request

    Returns:
        Tuple of (trace_id, parent_span_id)
    """
    trace_id = request.headers.get("X-Trace-Id")
    parent_span_id = request.headers.get("X-Parent-Span-Id")

    if not trace_id:
        trace_id = uuid.uuid4().hex
        logger.debug("Generated new trace_id: %s", trace_id)

    return trace_id, parent_span_id


def _add_trace_headers(response: Response, trace_id: str, spans: List[SpanData]) -> None:
    """Add trace headers to the response.

    Args:
        response: The FastAPI response object
        trace_id: The trace ID for this request
        spans: List of spans generated during the request
    """
    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Spans-Count"] = str(len(spans))


def _order_item_to_response(item: OrderItem) -> OrderItemResponse:
    """Convert OrderItem dataclass to OrderItemResponse model.

    Args:
        item: OrderItem dataclass instance

    Returns:
        OrderItemResponse model instance
    """
    return OrderItemResponse(
        product_id=item.product_id,
        quantity=item.quantity,
        unit_price=item.unit_price,
        total_price=item.total_price
    )


def _order_to_response(order: Order) -> OrderResponse:
    """Convert Order dataclass to OrderResponse model.

    Args:
        order: Order dataclass instance

    Returns:
        OrderResponse model instance
    """
    return OrderResponse(
        id=order.id,
        user_id=order.user_id,
        items=[_order_item_to_response(item) for item in order.items],
        status=order.status.value,
        total_amount=order.total_amount,
        created_at=order.created_at,
        processed_at=order.processed_at,
        error_message=order.error_message
    )


@router.post("", response_model=OrderResponse, status_code=201)
async def create_order(
    order_data: OrderCreateRequest,
    request: Request,
    response: Response,
    include_trace: bool = False
) -> OrderResponse | TraceResponse:
    """Create a new order.

    Creates an order with validation, inventory checking, and full tracemaid
    tracing. This endpoint demonstrates INFO and WARNING level logs.

    Args:
        order_data: The order creation request data
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        include_trace: Whether to include trace data in response

    Returns:
        OrderResponse with created order data

    Raises:
        HTTPException: 400 for validation errors
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "POST /orders - user_id: %s, items: %d, trace_id: %s",
        order_data.user_id,
        len(order_data.items),
        trace_id,
        extra={
            "user_id": order_data.user_id,
            "items_count": len(order_data.items),
            "trace_id": trace_id
        }
    )

    # Convert request items to service items
    order_items = [
        OrderItem(
            product_id=item.product_id,
            quantity=item.quantity,
            unit_price=item.unit_price
        )
        for item in order_data.items
    ]

    try:
        order, spans = _order_service.create_order(
            user_id=order_data.user_id,
            items=order_items,
            trace_id=trace_id,
            parent_span_id=parent_span_id
        )
    except ValueError as e:
        logger.error(
            "Order creation failed: %s",
            str(e),
            extra={"user_id": order_data.user_id, "trace_id": trace_id}
        )
        raise HTTPException(status_code=400, detail=str(e))

    _add_trace_headers(response, trace_id, spans)

    order_response = _order_to_response(order)

    if include_trace:
        return TraceResponse(
            data=order_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return order_response


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    request: Request,
    response: Response,
    include_trace: bool = False
) -> OrderResponse | TraceResponse:
    """Get an order by ID.

    Retrieves an order with full tracemaid tracing. This endpoint
    demonstrates WARNING level logs when order is not found.

    Args:
        order_id: The unique identifier of the order
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        include_trace: Whether to include trace data in response

    Returns:
        OrderResponse with order data

    Raises:
        HTTPException: 404 if order not found
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "GET /orders/%s - trace_id: %s",
        order_id,
        trace_id,
        extra={"order_id": order_id, "trace_id": trace_id}
    )

    order, spans = _order_service.get_order(
        order_id=order_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id
    )

    _add_trace_headers(response, trace_id, spans)

    if not order:
        logger.warning(
            "Order not found: %s",
            order_id,
            extra={"order_id": order_id, "trace_id": trace_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"Order with ID '{order_id}' not found"
        )

    order_response = _order_to_response(order)

    if include_trace:
        return TraceResponse(
            data=order_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return order_response


@router.post("/{order_id}/process", response_model=OrderResponse)
async def process_order(
    order_id: str,
    request: Request,
    response: Response,
    process_request: Optional[ProcessOrderRequest] = None,
    include_trace: bool = False
) -> OrderResponse | TraceResponse:
    """Process an order.

    Processes an order including inventory reservation, payment processing,
    and confirmation. This endpoint demonstrates ERROR level logs for various
    failure scenarios.

    Error Scenarios Demonstrated:
    - Order not found (404)
    - Order already processed (400)
    - Insufficient inventory (409)
    - Payment failure (402)
    - General processing error (500)

    Args:
        order_id: The unique identifier of the order to process
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        process_request: Optional processing options
        include_trace: Whether to include trace data in response

    Returns:
        OrderResponse with processed order data

    Raises:
        HTTPException: Various status codes for different error conditions
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "POST /orders/%s/process - trace_id: %s",
        order_id,
        trace_id,
        extra={"order_id": order_id, "trace_id": trace_id}
    )

    # Enable failure simulation if requested
    if process_request and process_request.simulate_failures:
        _order_service.enable_failure_simulation(True)
        logger.warning(
            "Failure simulation enabled for order processing",
            extra={"order_id": order_id, "trace_id": trace_id}
        )

    try:
        order, spans = _order_service.process_order(
            order_id=order_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id
        )
    except ValueError as e:
        error_message = str(e)
        logger.error(
            "Order processing failed (ValueError): %s",
            error_message,
            extra={"order_id": order_id, "trace_id": trace_id}
        )

        if "not found" in error_message.lower():
            raise HTTPException(status_code=404, detail=error_message)
        elif "already" in error_message.lower():
            raise HTTPException(status_code=400, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    except InsufficientInventoryError as e:
        logger.error(
            "Order processing failed (Inventory): %s",
            str(e),
            extra={"order_id": order_id, "trace_id": trace_id}
        )
        raise HTTPException(
            status_code=409,
            detail=str(e)
        )

    except PaymentFailedError as e:
        logger.error(
            "Order processing failed (Payment): %s",
            str(e),
            extra={"order_id": order_id, "trace_id": trace_id}
        )
        raise HTTPException(
            status_code=402,
            detail=str(e)
        )

    except OrderProcessingError as e:
        logger.error(
            "Order processing failed (General): %s",
            str(e),
            extra={"order_id": order_id, "trace_id": trace_id}
        )
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        # Disable failure simulation after processing
        if process_request and process_request.simulate_failures:
            _order_service.enable_failure_simulation(False)

    _add_trace_headers(response, trace_id, spans)

    order_response = _order_to_response(order)

    if include_trace:
        return TraceResponse(
            data=order_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return order_response


@router.post("/{order_id}/cancel", response_model=OrderResponse)
async def cancel_order(
    order_id: str,
    request: Request,
    response: Response,
    cancel_request: Optional[CancelOrderRequest] = None,
    include_trace: bool = False
) -> OrderResponse | TraceResponse:
    """Cancel an order.

    Cancels a pending or processing order. This endpoint demonstrates
    WARNING level logs for edge cases like already cancelled orders.

    Args:
        order_id: The unique identifier of the order to cancel
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        cancel_request: Optional cancellation options
        include_trace: Whether to include trace data in response

    Returns:
        OrderResponse with cancelled order data

    Raises:
        HTTPException: 404 if not found, 400 if cannot be cancelled
    """
    trace_id, parent_span_id = _get_trace_context(request)
    reason = cancel_request.reason if cancel_request else "customer_request"

    logger.info(
        "POST /orders/%s/cancel - reason: %s, trace_id: %s",
        order_id,
        reason,
        trace_id,
        extra={"order_id": order_id, "reason": reason, "trace_id": trace_id}
    )

    try:
        order, spans = _order_service.cancel_order(
            order_id=order_id,
            reason=reason,
            trace_id=trace_id,
            parent_span_id=parent_span_id
        )
    except ValueError as e:
        error_message = str(e)
        logger.error(
            "Order cancellation failed: %s",
            error_message,
            extra={"order_id": order_id, "trace_id": trace_id}
        )

        if "not found" in error_message.lower():
            raise HTTPException(status_code=404, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    _add_trace_headers(response, trace_id, spans)

    order_response = _order_to_response(order)

    if include_trace:
        return TraceResponse(
            data=order_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return order_response


@router.get("", response_model=OrderListResponse)
async def list_orders(
    request: Request,
    response: Response,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    include_trace: bool = False
) -> OrderListResponse | TraceResponse:
    """List orders with optional filtering.

    Retrieves orders with optional filtering by user_id and status.
    Full tracemaid tracing is applied.

    Args:
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        user_id: Optional user ID to filter by
        status: Optional status to filter by
        include_trace: Whether to include trace data in response

    Returns:
        OrderListResponse with list of orders and total count

    Raises:
        HTTPException: 400 if invalid status provided
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "GET /orders - user_id: %s, status: %s, trace_id: %s",
        user_id,
        status,
        trace_id,
        extra={"user_id": user_id, "status": status, "trace_id": trace_id}
    )

    # Parse status if provided
    order_status = None
    if status:
        try:
            order_status = OrderStatus(status)
        except ValueError:
            valid_statuses = [s.value for s in OrderStatus]
            logger.warning(
                "Invalid status provided: %s",
                status,
                extra={"provided": status, "valid": valid_statuses}
            )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Valid values: {valid_statuses}"
            )

    orders, spans = _order_service.list_orders(
        user_id=user_id,
        status=order_status,
        trace_id=trace_id,
        parent_span_id=parent_span_id
    )

    _add_trace_headers(response, trace_id, spans)

    order_responses = [_order_to_response(order) for order in orders]
    list_response = OrderListResponse(orders=order_responses, total=len(orders))

    if include_trace:
        return TraceResponse(
            data=list_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return list_response


@router.get("/trace-data", response_model=Dict[str, Any])
async def get_trace_data() -> Dict[str, Any]:
    """Get all collected trace data from the order service.

    This endpoint is useful for debugging and analyzing traces.

    Returns:
        Dictionary containing all collected trace spans in OTLP format
    """
    trace_data = _order_service.get_trace_data()
    return {
        "service": "order-service",
        "spans": trace_data,
        "total_spans": len(trace_data)
    }


@router.post("/trace-data/clear", status_code=204)
async def clear_trace_data() -> None:
    """Clear all collected trace data from the order service.

    This endpoint is useful for resetting trace collection between tests.
    """
    _order_service.clear_traces()
    logger.info("Order service trace data cleared")


@router.post("/simulation/enable-failures", status_code=200)
async def enable_failure_simulation(enabled: bool = True) -> Dict[str, Any]:
    """Enable or disable random failure simulation.

    When enabled, certain operations will randomly fail to demonstrate
    error logging capabilities.

    Args:
        enabled: Whether to enable failure simulation

    Returns:
        Status of failure simulation
    """
    _order_service.enable_failure_simulation(enabled)
    logger.info(
        "Failure simulation %s",
        "enabled" if enabled else "disabled"
    )
    return {
        "failure_simulation_enabled": enabled,
        "message": f"Failure simulation {'enabled' if enabled else 'disabled'}"
    }

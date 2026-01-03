"""
Order routes module for FastAPI demo application.

This module provides API endpoints for order operations with automatic
OpenTelemetry tracing. Traces are automatically exported as Mermaid
diagrams by TracemaidExporter - no manual trace retrieval needed!
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from examples.fastapi_demo.services import (
    OrderService,
    Order,
    OrderItem,
    OrderStatus,
    InsufficientInventoryError,
    PaymentFailedError,
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


class CancelOrderRequest(BaseModel):
    """Request model for cancelling an order.

    Attributes:
        reason: Reason for cancellation
    """
    reason: str = Field(
        default="customer_request",
        description="Reason for cancellation"
    )


def _order_item_to_response(item: OrderItem) -> OrderItemResponse:
    """Convert OrderItem dataclass to OrderItemResponse model."""
    return OrderItemResponse(
        product_id=item.product_id,
        quantity=item.quantity,
        unit_price=item.unit_price,
        total_price=item.total_price
    )


def _order_to_response(order: Order) -> OrderResponse:
    """Convert Order dataclass to OrderResponse model."""
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


@router.get("")
async def list_orders(user_id: Optional[str] = None) -> OrderListResponse:
    """List orders with optional filtering.

    Retrieves orders with optional filtering. Traces are automatically
    generated and exported as Mermaid diagrams.

    Args:
        user_id: Optional user ID to filter by

    Returns:
        OrderListResponse with list of orders and total count
    """
    logger.info("GET /orders - user_id: %s", user_id)

    orders = _order_service.list_orders(user_id=user_id)

    order_responses = [_order_to_response(order) for order in orders]
    return OrderListResponse(orders=order_responses, total=len(orders))


@router.post("", status_code=201)
async def create_order(order_data: OrderCreateRequest) -> OrderResponse:
    """Create a new order.

    Creates an order with validation and inventory checking. Traces are
    automatically generated and exported as Mermaid diagrams.

    Args:
        order_data: The order creation request data

    Returns:
        OrderResponse with created order data

    Raises:
        HTTPException: 400 for validation errors
    """
    logger.info(
        "POST /orders - user_id: %s, items: %d",
        order_data.user_id,
        len(order_data.items)
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
        order = _order_service.create_order(
            user_id=order_data.user_id,
            items=order_items
        )
    except ValueError as e:
        logger.error("Order creation failed: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

    return _order_to_response(order)


@router.get("/{order_id}")
async def get_order(order_id: str) -> OrderResponse:
    """Get an order by ID.

    Retrieves an order. Traces are automatically generated and
    exported as Mermaid diagrams.

    Args:
        order_id: The unique identifier of the order

    Returns:
        OrderResponse with order data

    Raises:
        HTTPException: 404 if order not found
    """
    logger.info("GET /orders/%s", order_id)

    order = _order_service.get_order(order_id=order_id)

    if not order:
        logger.warning("Order not found: %s", order_id)
        raise HTTPException(
            status_code=404,
            detail=f"Order with ID '{order_id}' not found"
        )

    return _order_to_response(order)


@router.post("/{order_id}/process")
async def process_order(order_id: str) -> OrderResponse:
    """Process an order.

    Processes an order including inventory reservation, payment processing,
    and confirmation. Traces are automatically generated and exported as
    Mermaid diagrams.

    Error Scenarios:
    - Order not found (404)
    - Order already processed (400)
    - Insufficient inventory (409)
    - Payment failure (402)

    Args:
        order_id: The unique identifier of the order to process

    Returns:
        OrderResponse with processed order data

    Raises:
        HTTPException: Various status codes for different error conditions
    """
    logger.info("POST /orders/%s/process", order_id)

    try:
        order = _order_service.process_order(order_id=order_id)
    except ValueError as e:
        error_message = str(e)
        logger.error("Order processing failed (ValueError): %s", error_message)

        if "not found" in error_message.lower():
            raise HTTPException(status_code=404, detail=error_message)
        elif "already" in error_message.lower():
            raise HTTPException(status_code=400, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    except InsufficientInventoryError as e:
        logger.error("Order processing failed (Inventory): %s", str(e))
        raise HTTPException(status_code=409, detail=str(e))

    except PaymentFailedError as e:
        logger.error("Order processing failed (Payment): %s", str(e))
        raise HTTPException(status_code=402, detail=str(e))

    return _order_to_response(order)


@router.post("/{order_id}/cancel")
async def cancel_order(
    order_id: str,
    cancel_request: Optional[CancelOrderRequest] = None
) -> OrderResponse:
    """Cancel an order.

    Cancels a pending or processing order. Traces are automatically
    generated and exported as Mermaid diagrams.

    Args:
        order_id: The unique identifier of the order to cancel
        cancel_request: Optional cancellation options

    Returns:
        OrderResponse with cancelled order data

    Raises:
        HTTPException: 404 if not found, 400 if cannot be cancelled
    """
    reason = cancel_request.reason if cancel_request else "customer_request"

    logger.info("POST /orders/%s/cancel - reason: %s", order_id, reason)

    try:
        order = _order_service.cancel_order(order_id=order_id, reason=reason)
    except ValueError as e:
        error_message = str(e)
        logger.error("Order cancellation failed: %s", error_message)

        if "not found" in error_message.lower():
            raise HTTPException(status_code=404, detail=error_message)
        else:
            raise HTTPException(status_code=400, detail=error_message)

    return _order_to_response(order)

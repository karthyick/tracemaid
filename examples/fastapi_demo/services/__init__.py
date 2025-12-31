"""Services package for FastAPI demo application.

Contains business logic services for user and order operations.
These services demonstrate tracemaid integration for trace data generation
and analysis.

Classes:
    UserService: Service for user CRUD operations with trace generation
    OrderService: Service for order operations with various log levels

Example:
    >>> from examples.fastapi_demo.services import UserService, OrderService
    >>> user_service = UserService()
    >>> order_service = OrderService()
"""

from examples.fastapi_demo.services.tracing import SpanData
from examples.fastapi_demo.services.user_service import UserService, User
from examples.fastapi_demo.services.order_service import (
    OrderService,
    Order,
    OrderItem,
    OrderStatus,
    InsufficientInventoryError,
    PaymentFailedError,
    OrderProcessingError,
)

__all__ = [
    # Shared tracing types
    "SpanData",
    # User service exports
    "UserService",
    "User",
    # Order service exports
    "OrderService",
    "Order",
    "OrderItem",
    "OrderStatus",
    "InsufficientInventoryError",
    "PaymentFailedError",
    "OrderProcessingError",
]

"""Services package for FastAPI demo application.

Contains business logic services for user and order operations.
These services use standard OpenTelemetry tracing which is automatically
exported as Mermaid diagrams by TracemaidExporter.

Classes:
    UserService: Service for user CRUD operations with automatic tracing
    OrderService: Service for order operations with automatic tracing

Example:
    >>> from tracemaid.integrations import setup_tracing
    >>> setup_tracing(service_name="demo-api", output_dir="./traces")
    >>>
    >>> from examples.fastapi_demo.services import UserService, OrderService
    >>> user_service = UserService()
    >>> order_service = OrderService()
    >>>
    >>> # All operations automatically generate Mermaid diagrams!
    >>> user = user_service.create_user("john", "john@example.com")
"""

from examples.fastapi_demo.services.user_service import UserService, User
from examples.fastapi_demo.services.order_service import (
    OrderService,
    Order,
    OrderItem,
    OrderStatus,
    InsufficientInventoryError,
    PaymentFailedError,
)

__all__ = [
    "UserService",
    "User",
    "OrderService",
    "Order",
    "OrderItem",
    "OrderStatus",
    "InsufficientInventoryError",
    "PaymentFailedError",
]

"""Routes package for FastAPI demo application.

Contains API route definitions for user and order endpoints with
tracemaid integration for request tracing and correlation ID propagation.

Routers:
    user_router: APIRouter for user CRUD operations
    order_router: APIRouter for order operations with error handling

Example:
    >>> from examples.fastapi_demo.routes import user_router, order_router
    >>> app = FastAPI()
    >>> app.include_router(user_router)
    >>> app.include_router(order_router)
"""

from examples.fastapi_demo.routes.user_routes import router as user_router
from examples.fastapi_demo.routes.order_routes import router as order_router

__all__ = [
    "user_router",
    "order_router",
]

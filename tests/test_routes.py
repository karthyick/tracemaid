"""Unit tests for FastAPI demo routes.

This module provides comprehensive test coverage for user_routes and order_routes,
validating API endpoints, request/response models, and error handling.

Note: Routes now use standard OpenTelemetry tracing. Traces are automatically
exported by TracemaidExporter - no manual trace retrieval needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# FastAPI test client requires httpx
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from examples.fastapi_demo.routes import user_router, order_router
from examples.fastapi_demo.routes.user_routes import (
    UserCreateRequest,
    UserResponse,
    UserListResponse,
    _user_service,
)
from examples.fastapi_demo.routes.order_routes import (
    OrderCreateRequest,
    OrderItemRequest,
    OrderResponse,
    OrderListResponse,
    OrderItemResponse,
    CancelOrderRequest,
    _order_service,
)
from examples.fastapi_demo.services import (
    User,
    Order,
    OrderItem,
    OrderStatus,
)


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with both routers for testing."""
    test_app = FastAPI()
    test_app.include_router(user_router)
    test_app.include_router(order_router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_services() -> None:
    """Reset service stores before each test."""
    _user_service._users.clear()
    _order_service._orders.clear()
    # Reset inventory to original values
    _order_service._INVENTORY.update({
        "prod-001": 100,
        "prod-002": 50,
        "prod-003": 25,
        "prod-004": 0,
        "prod-005": 5,
    })


# =============================================================================
# User Routes Tests
# =============================================================================


class TestUserRoutesListUsers:
    """Test GET /users endpoint."""

    def test_list_users_empty(self, client: TestClient) -> None:
        """Test listing users when none exist."""
        response = client.get("/users")
        assert response.status_code == 200
        data = response.json()
        assert data["users"] == []
        assert data["total"] == 0

    def test_list_users_multiple(self, client: TestClient) -> None:
        """Test listing multiple users."""
        # Create users
        client.post("/users", json={"username": "user1", "email": "u1@test.com"})
        client.post("/users", json={"username": "user2", "email": "u2@test.com"})

        response = client.get("/users")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["users"]) == 2


class TestUserRoutesCreateUser:
    """Test POST /users endpoint."""

    def test_create_user_success(self, client: TestClient) -> None:
        """Test successful user creation."""
        response = client.post(
            "/users",
            json={"username": "testuser", "email": "test@example.com"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data

    def test_create_user_empty_username(self, client: TestClient) -> None:
        """Test user creation with empty username fails."""
        response = client.post(
            "/users",
            json={"username": "", "email": "test@example.com"}
        )
        assert response.status_code == 422  # Pydantic validation

    def test_create_user_short_username(self, client: TestClient) -> None:
        """Test user creation with short username fails."""
        response = client.post(
            "/users",
            json={"username": "ab", "email": "test@example.com"}
        )
        assert response.status_code == 422

    def test_create_user_invalid_email(self, client: TestClient) -> None:
        """Test user creation with invalid email fails."""
        response = client.post(
            "/users",
            json={"username": "testuser", "email": "not-an-email"}
        )
        assert response.status_code == 422

    def test_create_user_duplicate_username(self, client: TestClient) -> None:
        """Test user creation with duplicate username fails."""
        client.post("/users", json={"username": "testuser", "email": "t1@test.com"})
        response = client.post(
            "/users",
            json={"username": "testuser", "email": "t2@test.com"}
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]


class TestUserRoutesGetUser:
    """Test GET /users/{user_id} endpoint."""

    def test_get_user_success(self, client: TestClient) -> None:
        """Test getting an existing user."""
        create_response = client.post(
            "/users",
            json={"username": "testuser", "email": "test@example.com"}
        )
        user_id = create_response.json()["id"]

        response = client.get(f"/users/{user_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == "testuser"

    def test_get_user_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent user."""
        response = client.get("/users/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestUserRoutesDeleteUser:
    """Test DELETE /users/{user_id} endpoint."""

    def test_delete_user_success(self, client: TestClient) -> None:
        """Test deleting an existing user."""
        create_response = client.post(
            "/users",
            json={"username": "testuser", "email": "test@example.com"}
        )
        user_id = create_response.json()["id"]

        response = client.delete(f"/users/{user_id}")
        assert response.status_code == 204

        # Verify user is gone
        get_response = client.get(f"/users/{user_id}")
        assert get_response.status_code == 404

    def test_delete_user_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent user."""
        response = client.delete("/users/nonexistent-id")
        assert response.status_code == 404


# =============================================================================
# Order Routes Tests
# =============================================================================


class TestOrderRoutesListOrders:
    """Test GET /orders endpoint."""

    def test_list_orders_empty(self, client: TestClient) -> None:
        """Test listing orders when none exist."""
        response = client.get("/orders")
        assert response.status_code == 200
        data = response.json()
        assert data["orders"] == []
        assert data["total"] == 0

    def test_list_orders_with_user_filter(self, client: TestClient) -> None:
        """Test listing orders filtered by user_id."""
        # Create orders
        client.post("/orders", json={
            "user_id": "user-1",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        client.post("/orders", json={
            "user_id": "user-2",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })

        response = client.get("/orders?user_id=user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["orders"][0]["user_id"] == "user-1"

class TestOrderRoutesCreateOrder:
    """Test POST /orders endpoint."""

    def test_create_order_success(self, client: TestClient) -> None:
        """Test successful order creation."""
        response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [
                {"product_id": "prod-001", "quantity": 2, "unit_price": 29.99}
            ]
        })
        assert response.status_code == 201
        data = response.json()
        assert data["user_id"] == "user-123"
        assert data["status"] == "pending"
        assert len(data["items"]) == 1
        assert "id" in data
        assert "total_amount" in data

    def test_create_order_calculates_total(self, client: TestClient) -> None:
        """Test order total calculation."""
        response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [
                {"product_id": "prod-001", "quantity": 2, "unit_price": 10.00},
                {"product_id": "prod-002", "quantity": 3, "unit_price": 5.00}
            ]
        })
        data = response.json()
        # 2*10 + 3*5 = 35
        assert data["total_amount"] == 35.00

    def test_create_order_empty_user_id(self, client: TestClient) -> None:
        """Test order creation with empty user_id fails."""
        response = client.post("/orders", json={
            "user_id": "",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        assert response.status_code == 422

    def test_create_order_empty_items(self, client: TestClient) -> None:
        """Test order creation with empty items list fails."""
        response = client.post("/orders", json={
            "user_id": "user-123",
            "items": []
        })
        assert response.status_code == 422

    def test_create_order_invalid_quantity(self, client: TestClient) -> None:
        """Test order creation with invalid quantity fails."""
        response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 0, "unit_price": 10.00}]
        })
        assert response.status_code == 422

    def test_create_order_invalid_price(self, client: TestClient) -> None:
        """Test order creation with invalid price fails."""
        response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 0}]
        })
        assert response.status_code == 422


class TestOrderRoutesGetOrder:
    """Test GET /orders/{order_id} endpoint."""

    def test_get_order_success(self, client: TestClient) -> None:
        """Test getting an existing order."""
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]

        response = client.get(f"/orders/{order_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == order_id
        assert data["user_id"] == "user-123"

    def test_get_order_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent order."""
        response = client.get("/orders/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestOrderRoutesProcessOrder:
    """Test POST /orders/{order_id}/process endpoint."""

    def test_process_order_success(self, client: TestClient) -> None:
        """Test successful order processing."""
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]

        response = client.post(f"/orders/{order_id}/process")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["processed_at"] is not None

    def test_process_order_not_found(self, client: TestClient) -> None:
        """Test processing a non-existent order."""
        response = client.post("/orders/nonexistent-id/process")
        assert response.status_code == 404

    def test_process_order_already_completed(self, client: TestClient) -> None:
        """Test processing an already completed order."""
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]
        client.post(f"/orders/{order_id}/process")

        response = client.post(f"/orders/{order_id}/process")
        assert response.status_code == 400
        assert "already" in response.json()["detail"]

    def test_process_order_insufficient_inventory(self, client: TestClient) -> None:
        """Test processing order with insufficient inventory."""
        # prod-004 has 0 inventory
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-004", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]

        response = client.post(f"/orders/{order_id}/process")
        assert response.status_code == 409


class TestOrderRoutesCancelOrder:
    """Test POST /orders/{order_id}/cancel endpoint."""

    def test_cancel_order_success(self, client: TestClient) -> None:
        """Test successful order cancellation."""
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]

        response = client.post(f"/orders/{order_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_cancel_order_with_reason(self, client: TestClient) -> None:
        """Test cancelling order with custom reason."""
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]

        response = client.post(
            f"/orders/{order_id}/cancel",
            json={"reason": "out_of_stock"}
        )
        assert response.status_code == 200

    def test_cancel_order_not_found(self, client: TestClient) -> None:
        """Test cancelling a non-existent order."""
        response = client.post("/orders/nonexistent-id/cancel")
        assert response.status_code == 404

    def test_cancel_order_completed_fails(self, client: TestClient) -> None:
        """Test cancelling a completed order fails."""
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]
        client.post(f"/orders/{order_id}/process")

        response = client.post(f"/orders/{order_id}/cancel")
        assert response.status_code == 400
        assert "Cannot cancel completed" in response.json()["detail"]

    def test_cancel_order_already_cancelled_succeeds(self, client: TestClient) -> None:
        """Test cancelling an already cancelled order succeeds."""
        create_response = client.post("/orders", json={
            "user_id": "user-123",
            "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
        })
        order_id = create_response.json()["id"]
        client.post(f"/orders/{order_id}/cancel")

        response = client.post(f"/orders/{order_id}/cancel")
        assert response.status_code == 200


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestPydanticModels:
    """Test Pydantic request/response models."""

    def test_user_create_request_validation(self) -> None:
        """Test UserCreateRequest validation."""
        # Valid request
        request = UserCreateRequest(username="testuser", email="test@example.com")
        assert request.username == "testuser"
        assert request.email == "test@example.com"

    def test_order_create_request_validation(self) -> None:
        """Test OrderCreateRequest validation."""
        items = [OrderItemRequest(product_id="prod-1", quantity=2, unit_price=10.00)]
        request = OrderCreateRequest(user_id="user-1", items=items)
        assert request.user_id == "user-1"
        assert len(request.items) == 1

    def test_order_item_request_validation(self) -> None:
        """Test OrderItemRequest validation."""
        item = OrderItemRequest(product_id="prod-1", quantity=2, unit_price=29.99)
        assert item.product_id == "prod-1"
        assert item.quantity == 2
        assert item.unit_price == 29.99

    def test_order_response_model(self) -> None:
        """Test OrderResponse model structure."""
        items = [OrderItemResponse(
            product_id="prod-1", quantity=2, unit_price=10.00, total_price=20.00
        )]
        response = OrderResponse(
            id="order-1",
            user_id="user-1",
            items=items,
            status="pending",
            total_amount=20.00,
            created_at=1234567890.0
        )
        assert response.id == "order-1"
        assert response.status == "pending"


# =============================================================================
# Router Configuration Tests
# =============================================================================


class TestRouterConfiguration:
    """Test router configuration and setup."""

    def test_user_router_has_correct_prefix(self) -> None:
        """Test user router has /users prefix."""
        assert user_router.prefix == "/users"

    def test_order_router_has_correct_prefix(self) -> None:
        """Test order router has /orders prefix."""
        assert order_router.prefix == "/orders"

    def test_user_router_has_correct_tags(self) -> None:
        """Test user router has correct tags."""
        assert "users" in user_router.tags

    def test_order_router_has_correct_tags(self) -> None:
        """Test order router has correct tags."""
        assert "orders" in order_router.tags


# =============================================================================
# Integration Tests - End-to-End Flow
# =============================================================================


class TestEndToEndFlow:
    """Test end-to-end request flow through routes and services."""

    def test_complete_user_workflow(self, client: TestClient) -> None:
        """Test complete user CRUD workflow."""
        # Create user
        create_response = client.post(
            "/users",
            json={"username": "integrationuser", "email": "int@test.com"}
        )
        assert create_response.status_code == 201
        user_id = create_response.json()["id"]

        # Get user
        get_response = client.get(f"/users/{user_id}")
        assert get_response.status_code == 200
        assert get_response.json()["username"] == "integrationuser"

        # List users
        list_response = client.get("/users")
        assert list_response.status_code == 200
        assert list_response.json()["total"] == 1

        # Delete user
        delete_response = client.delete(f"/users/{user_id}")
        assert delete_response.status_code == 204

        # Verify deleted
        verify_response = client.get(f"/users/{user_id}")
        assert verify_response.status_code == 404

    def test_complete_order_workflow(self, client: TestClient) -> None:
        """Test complete order workflow."""
        # Create order
        create_response = client.post(
            "/orders",
            json={
                "user_id": "test-user",
                "items": [
                    {"product_id": "prod-001", "quantity": 2, "unit_price": 25.00}
                ]
            }
        )
        assert create_response.status_code == 201
        order_id = create_response.json()["id"]

        # Get order
        get_response = client.get(f"/orders/{order_id}")
        assert get_response.status_code == 200
        assert get_response.json()["status"] == "pending"

        # Process order
        process_response = client.post(f"/orders/{order_id}/process")
        assert process_response.status_code == 200
        assert process_response.json()["status"] == "completed"

        # List orders
        list_response = client.get("/orders")
        assert list_response.status_code == 200
        assert list_response.json()["total"] == 1

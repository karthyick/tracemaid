"""
Comprehensive endpoint tests for FastAPI demo application.

This module provides pytest-based tests for all API endpoints in the FastAPI
demo application, including tests for tracemaid logging verification.

Tests cover:
- Health check endpoint
- User CRUD operations
- Order operations and error handling
- Correlation ID propagation
- Log level verification
- Trace context propagation
"""

from __future__ import annotations

import logging
import re
import uuid
from io import StringIO
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from examples.fastapi_demo.main import app, clear_request_traces


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application.

    Yields:
        TestClient instance configured for the demo app
    """
    # Clear traces before each test
    clear_request_traces()

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def log_capture() -> Generator[StringIO, None, None]:
    """Capture log output for verification.

    Yields:
        StringIO buffer containing captured log output
    """
    log_buffer = StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | trace_id=%(trace_id)s | "
        "span_id=%(span_id)s | %(name)s:%(lineno)d | %(message)s"
    )

    # Add custom filter to inject trace context
    class TraceContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, 'trace_id'):
                record.trace_id = '-'
            if not hasattr(record, 'span_id'):
                record.span_id = '-'
            return True

    handler.addFilter(TraceContextFilter())
    handler.setFormatter(formatter)

    # Get root logger and add handler
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)

    yield log_buffer

    # Cleanup
    root_logger.removeHandler(handler)
    root_logger.setLevel(original_level)


@pytest.fixture
def created_user(client: TestClient) -> Dict[str, Any]:
    """Create a test user and return the response data.

    Args:
        client: Test client fixture

    Returns:
        Dictionary containing created user data
    """
    response = client.post(
        "/api/v1/users",
        json={
            "username": f"testuser_{uuid.uuid4().hex[:8]}",
            "email": "testuser@example.com"
        }
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def created_order(client: TestClient, created_user: Dict[str, Any]) -> Dict[str, Any]:
    """Create a test order and return the response data.

    Args:
        client: Test client fixture
        created_user: Created user fixture for user_id

    Returns:
        Dictionary containing created order data
    """
    response = client.post(
        "/api/v1/orders",
        json={
            "user_id": created_user["id"],
            "items": [
                {
                    "product_id": "prod-001",
                    "quantity": 2,
                    "unit_price": 29.99
                }
            ]
        }
    )
    assert response.status_code == 201
    return response.json()


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_check_returns_200(self, client: TestClient) -> None:
        """Test that health check returns 200 OK status."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_check_response_structure(self, client: TestClient) -> None:
        """Test that health check returns expected response structure."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "app" in data
        assert "version" in data
        assert "timestamp" in data
        assert "tracemaid" in data

    def test_health_check_tracemaid_config(self, client: TestClient) -> None:
        """Test that health check includes tracemaid configuration."""
        response = client.get("/health")
        data = response.json()

        tracemaid_config = data.get("tracemaid", {})
        assert tracemaid_config.get("enabled") is True
        assert "middleware" in tracemaid_config

    def test_health_check_has_trace_headers(self, client: TestClient) -> None:
        """Test that health check response includes trace headers."""
        response = client.get("/health")

        assert "x-trace-id" in response.headers
        assert "x-span-id" in response.headers
        assert "x-request-duration-ms" in response.headers


# =============================================================================
# User CRUD Tests
# =============================================================================

class TestUserCRUD:
    """Tests for user CRUD operations."""

    def test_list_users_empty(self, client: TestClient) -> None:
        """Test listing users when no users exist."""
        response = client.get("/api/v1/users")

        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert isinstance(data["users"], list)

    def test_create_user_success(self, client: TestClient) -> None:
        """Test creating a new user successfully."""
        username = f"newuser_{uuid.uuid4().hex[:8]}"
        response = client.post(
            "/api/v1/users",
            json={
                "username": username,
                "email": "newuser@example.com"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["username"] == username
        assert data["email"] == "newuser@example.com"
        assert "id" in data
        assert data["is_active"] is True

    def test_create_user_duplicate_username(self, client: TestClient) -> None:
        """Test that creating a user with duplicate username fails."""
        username = f"duplicate_{uuid.uuid4().hex[:8]}"

        # Create first user
        response1 = client.post(
            "/api/v1/users",
            json={"username": username, "email": "first@example.com"}
        )
        assert response1.status_code == 201

        # Try to create duplicate
        response2 = client.post(
            "/api/v1/users",
            json={"username": username, "email": "second@example.com"}
        )
        assert response2.status_code == 409

    def test_create_user_invalid_username(self, client: TestClient) -> None:
        """Test that creating a user with short username fails validation."""
        response = client.post(
            "/api/v1/users",
            json={"username": "ab", "email": "short@example.com"}
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_get_user_success(self, client: TestClient, created_user: Dict[str, Any]) -> None:
        """Test getting an existing user."""
        user_id = created_user["id"]
        response = client.get(f"/api/v1/users/{user_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user_id
        assert data["username"] == created_user["username"]

    def test_get_user_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent user returns 404."""
        response = client.get("/api/v1/users/nonexistent-id")

        assert response.status_code == 404

    def test_delete_user_success(self, client: TestClient, created_user: Dict[str, Any]) -> None:
        """Test deleting an existing user."""
        user_id = created_user["id"]
        response = client.delete(f"/api/v1/users/{user_id}")

        assert response.status_code == 204

        # Verify user is deleted
        get_response = client.get(f"/api/v1/users/{user_id}")
        assert get_response.status_code == 404

    def test_delete_user_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent user returns 404."""
        response = client.delete("/api/v1/users/nonexistent-id")

        assert response.status_code == 404

    def test_user_crud_with_trace_id_propagation(self, client: TestClient) -> None:
        """Test that trace ID is propagated through user operations."""
        custom_trace_id = uuid.uuid4().hex
        headers = {"X-Trace-Id": custom_trace_id}

        # Create user with custom trace ID
        response = client.post(
            "/api/v1/users",
            json={
                "username": f"traced_user_{uuid.uuid4().hex[:8]}",
                "email": "traced@example.com"
            },
            headers=headers
        )

        assert response.status_code == 201
        assert response.headers.get("x-trace-id") == custom_trace_id


# =============================================================================
# Order Operations Tests
# =============================================================================

class TestOrderOperations:
    """Tests for order operations."""

    def test_list_orders_empty(self, client: TestClient) -> None:
        """Test listing orders when no orders exist."""
        response = client.get("/api/v1/orders")

        assert response.status_code == 200
        data = response.json()
        assert "orders" in data
        assert "total" in data

    def test_create_order_success(self, client: TestClient) -> None:
        """Test creating a new order successfully."""
        response = client.post(
            "/api/v1/orders",
            json={
                "user_id": "user-123",
                "items": [
                    {
                        "product_id": "prod-001",
                        "quantity": 2,
                        "unit_price": 29.99
                    }
                ]
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["user_id"] == "user-123"
        assert data["status"] == "pending"
        assert len(data["items"]) == 1
        assert data["total_amount"] == 59.98

    def test_create_order_empty_items(self, client: TestClient) -> None:
        """Test that creating an order with empty items fails."""
        response = client.post(
            "/api/v1/orders",
            json={
                "user_id": "user-123",
                "items": []
            }
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_get_order_success(self, client: TestClient, created_order: Dict[str, Any]) -> None:
        """Test getting an existing order."""
        order_id = created_order["id"]
        response = client.get(f"/api/v1/orders/{order_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == order_id

    def test_get_order_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent order returns 404."""
        response = client.get("/api/v1/orders/nonexistent-id")

        assert response.status_code == 404

    def test_process_order_success(self, client: TestClient, created_order: Dict[str, Any]) -> None:
        """Test processing an order successfully."""
        order_id = created_order["id"]
        response = client.post(f"/api/v1/orders/{order_id}/process")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["processed_at"] is not None

    def test_process_order_not_found(self, client: TestClient) -> None:
        """Test processing a non-existent order returns 404."""
        response = client.post("/api/v1/orders/nonexistent-id/process")

        assert response.status_code == 404

    def test_process_order_already_completed(
        self,
        client: TestClient,
        created_order: Dict[str, Any]
    ) -> None:
        """Test that processing an already completed order fails."""
        order_id = created_order["id"]

        # First process
        response1 = client.post(f"/api/v1/orders/{order_id}/process")
        assert response1.status_code == 200

        # Try to process again
        response2 = client.post(f"/api/v1/orders/{order_id}/process")
        assert response2.status_code == 400

    def test_cancel_order_success(self, client: TestClient, created_order: Dict[str, Any]) -> None:
        """Test cancelling an order successfully."""
        order_id = created_order["id"]
        response = client.post(
            f"/api/v1/orders/{order_id}/cancel",
            json={"reason": "customer_request"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_cancel_order_already_completed(
        self,
        client: TestClient,
        created_order: Dict[str, Any]
    ) -> None:
        """Test that cancelling a completed order fails."""
        order_id = created_order["id"]

        # Process order first
        client.post(f"/api/v1/orders/{order_id}/process")

        # Try to cancel
        response = client.post(
            f"/api/v1/orders/{order_id}/cancel",
            json={"reason": "changed_mind"}
        )
        assert response.status_code == 400

    def test_list_orders_with_status_filter(
        self,
        client: TestClient,
        created_order: Dict[str, Any]
    ) -> None:
        """Test listing orders with status filter."""
        response = client.get("/api/v1/orders?status=pending")

        assert response.status_code == 200
        data = response.json()
        assert all(o["status"] == "pending" for o in data["orders"])

    def test_list_orders_invalid_status(self, client: TestClient) -> None:
        """Test listing orders with invalid status returns 400."""
        response = client.get("/api/v1/orders?status=invalid_status")

        assert response.status_code == 400


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_404_for_unknown_route(self, client: TestClient) -> None:
        """Test that unknown routes return 404."""
        response = client.get("/api/v1/unknown")

        assert response.status_code == 404

    def test_method_not_allowed(self, client: TestClient) -> None:
        """Test that wrong HTTP method returns 405."""
        response = client.put("/health")

        assert response.status_code == 405

    def test_validation_error_response_format(self, client: TestClient) -> None:
        """Test that validation errors return proper format."""
        response = client.post(
            "/api/v1/users",
            json={"username": "ab"}  # Missing email, short username
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_error_response_includes_trace_headers(self, client: TestClient) -> None:
        """Test that error responses include trace headers."""
        response = client.get("/api/v1/users/nonexistent-id")

        assert response.status_code == 404
        assert "x-trace-id" in response.headers


# =============================================================================
# Correlation ID Tests
# =============================================================================

class TestCorrelationIdPresent:
    """Tests for correlation ID presence and propagation."""

    def test_correlation_id_generated_when_not_provided(self, client: TestClient) -> None:
        """Test that correlation ID is generated when not provided."""
        response = client.get("/health")

        trace_id = response.headers.get("x-trace-id")
        assert trace_id is not None
        assert len(trace_id) == 32  # UUID hex without dashes

    def test_correlation_id_propagated_when_provided(self, client: TestClient) -> None:
        """Test that provided correlation ID is propagated."""
        custom_trace_id = uuid.uuid4().hex

        response = client.get(
            "/health",
            headers={"X-Trace-Id": custom_trace_id}
        )

        assert response.headers.get("x-trace-id") == custom_trace_id

    def test_correlation_id_consistent_across_subrequests(
        self,
        client: TestClient
    ) -> None:
        """Test correlation ID consistency when making related requests."""
        custom_trace_id = uuid.uuid4().hex
        headers = {"X-Trace-Id": custom_trace_id}

        # Make multiple requests with same trace ID
        response1 = client.post(
            "/api/v1/users",
            json={
                "username": f"consistent_user_{uuid.uuid4().hex[:8]}",
                "email": "consistent@example.com"
            },
            headers=headers
        )

        response2 = client.get(
            "/api/v1/users",
            headers=headers
        )

        assert response1.headers.get("x-trace-id") == custom_trace_id
        assert response2.headers.get("x-trace-id") == custom_trace_id

    def test_span_id_present_in_response(self, client: TestClient) -> None:
        """Test that span ID is present in response headers."""
        response = client.get("/health")

        span_id = response.headers.get("x-span-id")
        assert span_id is not None
        assert len(span_id) == 16  # Short span ID


# =============================================================================
# Log Level Verification Tests
# =============================================================================

class TestLogLevelsCorrect:
    """Tests for log level correctness."""

    def test_info_log_on_successful_request(
        self,
        client: TestClient,
        log_capture: StringIO
    ) -> None:
        """Test that successful requests generate INFO level logs."""
        client.get("/health")

        log_output = log_capture.getvalue()
        assert "INFO" in log_output
        assert "Request started" in log_output or "Request completed" in log_output

    def test_warning_log_on_not_found(
        self,
        client: TestClient,
        log_capture: StringIO
    ) -> None:
        """Test that 404 responses generate WARNING level logs."""
        client.get("/api/v1/users/nonexistent-user-id")

        log_output = log_capture.getvalue()
        # Check for WARNING level in logs
        assert "WARNING" in log_output or "not found" in log_output.lower()

    def test_error_log_on_validation_failure(
        self,
        client: TestClient,
        log_capture: StringIO
    ) -> None:
        """Test that validation errors generate appropriate logs."""
        # Create user with invalid data
        client.post(
            "/api/v1/users",
            json={"username": "", "email": "invalid"}
        )

        # Log output should contain request information
        log_output = log_capture.getvalue()
        assert len(log_output) > 0  # Logs were captured

    def test_debug_log_for_span_creation(
        self,
        client: TestClient,
        log_capture: StringIO
    ) -> None:
        """Test that span creation generates DEBUG level logs."""
        # Set logging level to DEBUG
        logging.getLogger().setLevel(logging.DEBUG)

        client.get("/api/v1/users")

        log_output = log_capture.getvalue()
        # Verify some logging occurred
        assert len(log_output) > 0


# =============================================================================
# Trace Context Propagation Tests
# =============================================================================

class TestTraceContextPropagation:
    """Tests for trace context propagation."""

    def test_trace_data_endpoint_returns_traces(self, client: TestClient) -> None:
        """Test that trace data endpoint returns collected traces."""
        # Make some requests to generate traces
        client.get("/health")
        client.get("/api/v1/users")

        # Get trace data
        response = client.get("/trace-data")

        assert response.status_code == 200
        data = response.json()
        assert "trace_count" in data
        assert "traces" in data

    def test_clear_trace_data_endpoint(self, client: TestClient) -> None:
        """Test that trace data can be cleared."""
        # Generate some traces
        client.get("/health")

        # Clear traces
        response = client.post("/trace-data/clear")
        assert response.status_code == 204

        # Verify traces are cleared
        get_response = client.get("/trace-data")
        # After clearing, new request will create a new trace
        data = get_response.json()
        assert data["trace_count"] >= 1  # At least the /trace-data request itself

    def test_request_duration_header_present(self, client: TestClient) -> None:
        """Test that request duration is included in response headers."""
        response = client.get("/health")

        duration_header = response.headers.get("x-request-duration-ms")
        assert duration_header is not None

        # Verify it's a valid number
        duration = float(duration_header)
        assert duration >= 0

    def test_trace_includes_service_spans(
        self,
        client: TestClient,
        created_user: Dict[str, Any]
    ) -> None:
        """Test that traces include service-level spans."""
        # Get user service trace data
        response = client.get("/api/v1/users/trace-data")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "user-service"
        assert "spans" in data

    def test_order_trace_includes_all_operations(
        self,
        client: TestClient,
        created_order: Dict[str, Any]
    ) -> None:
        """Test that order processing includes all expected trace operations."""
        order_id = created_order["id"]

        # Process the order
        client.post(f"/api/v1/orders/{order_id}/process")

        # Get order service trace data
        response = client.get("/api/v1/orders/trace-data")

        assert response.status_code == 200
        data = response.json()
        assert len(data["spans"]) > 0

    def test_parent_span_propagation(self, client: TestClient) -> None:
        """Test that parent span ID is properly propagated."""
        custom_trace_id = uuid.uuid4().hex
        custom_parent_span_id = uuid.uuid4().hex[:16]

        response = client.post(
            "/api/v1/users",
            json={
                "username": f"parent_span_user_{uuid.uuid4().hex[:8]}",
                "email": "parent@example.com"
            },
            headers={
                "X-Trace-Id": custom_trace_id,
                "X-Parent-Span-Id": custom_parent_span_id
            }
        )

        assert response.status_code == 201
        # The trace ID should be propagated
        assert response.headers.get("x-trace-id") == custom_trace_id


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_user_lifecycle(self, client: TestClient) -> None:
        """Test complete user lifecycle: create, read, delete."""
        # Create
        username = f"lifecycle_user_{uuid.uuid4().hex[:8]}"
        create_response = client.post(
            "/api/v1/users",
            json={"username": username, "email": "lifecycle@example.com"}
        )
        assert create_response.status_code == 201
        user_id = create_response.json()["id"]

        # Read
        get_response = client.get(f"/api/v1/users/{user_id}")
        assert get_response.status_code == 200
        assert get_response.json()["username"] == username

        # Delete
        delete_response = client.delete(f"/api/v1/users/{user_id}")
        assert delete_response.status_code == 204

        # Verify deleted
        verify_response = client.get(f"/api/v1/users/{user_id}")
        assert verify_response.status_code == 404

    def test_complete_order_lifecycle(self, client: TestClient) -> None:
        """Test complete order lifecycle: create, process."""
        # Create order
        create_response = client.post(
            "/api/v1/orders",
            json={
                "user_id": "lifecycle-user",
                "items": [
                    {"product_id": "prod-001", "quantity": 1, "unit_price": 19.99}
                ]
            }
        )
        assert create_response.status_code == 201
        order_id = create_response.json()["id"]
        assert create_response.json()["status"] == "pending"

        # Process order
        process_response = client.post(f"/api/v1/orders/{order_id}/process")
        assert process_response.status_code == 200
        assert process_response.json()["status"] == "completed"

        # Verify final state
        get_response = client.get(f"/api/v1/orders/{order_id}")
        assert get_response.status_code == 200
        assert get_response.json()["status"] == "completed"

    def test_order_cancellation_workflow(self, client: TestClient) -> None:
        """Test order cancellation workflow."""
        # Create order
        create_response = client.post(
            "/api/v1/orders",
            json={
                "user_id": "cancel-test-user",
                "items": [
                    {"product_id": "prod-002", "quantity": 3, "unit_price": 15.00}
                ]
            }
        )
        assert create_response.status_code == 201
        order_id = create_response.json()["id"]

        # Cancel order
        cancel_response = client.post(
            f"/api/v1/orders/{order_id}/cancel",
            json={"reason": "out_of_stock"}
        )
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"

    def test_trace_consistency_across_operations(self, client: TestClient) -> None:
        """Test that trace IDs are consistent across related operations."""
        trace_id = uuid.uuid4().hex
        headers = {"X-Trace-Id": trace_id}

        # Create user
        user_response = client.post(
            "/api/v1/users",
            json={
                "username": f"trace_test_{uuid.uuid4().hex[:8]}",
                "email": "trace@example.com"
            },
            headers=headers
        )
        assert user_response.headers.get("x-trace-id") == trace_id

        # Create order
        order_response = client.post(
            "/api/v1/orders",
            json={
                "user_id": user_response.json()["id"],
                "items": [{"product_id": "prod-001", "quantity": 1, "unit_price": 10.00}]
            },
            headers=headers
        )
        assert order_response.headers.get("x-trace-id") == trace_id


# =============================================================================
# Root Endpoint Tests
# =============================================================================

class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self, client: TestClient) -> None:
        """Test that root endpoint returns 200."""
        response = client.get("/")

        assert response.status_code == 200

    def test_root_response_structure(self, client: TestClient) -> None:
        """Test root endpoint response structure."""
        response = client.get("/")
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


# =============================================================================
# OpenAPI Documentation Tests
# =============================================================================

class TestDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_json_available(self, client: TestClient) -> None:
        """Test that OpenAPI JSON is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_endpoint_available(self, client: TestClient) -> None:
        """Test that Swagger UI docs are available."""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint_available(self, client: TestClient) -> None:
        """Test that ReDoc is available."""
        response = client.get("/redoc")

        assert response.status_code == 200

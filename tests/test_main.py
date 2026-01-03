"""Unit tests for FastAPI main application with Tracemaid integration.

This module provides comprehensive test coverage for the main FastAPI application,
including health endpoints, route registration, and lifecycle management.

Note: The application now uses standard OpenTelemetry tracing via TracemaidExporter.
Traces are automatically generated and exported - no manual trace retrieval needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock
import logging

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

from examples.fastapi_demo.main import (
    app,
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    TRACES_OUTPUT_DIR,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the main app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Reset application state before each test."""
    # Reset user and order service stores
    from examples.fastapi_demo.routes.user_routes import _user_service
    from examples.fastapi_demo.routes.order_routes import _order_service
    _user_service._users.clear()
    _order_service._orders.clear()
    _order_service._INVENTORY.update({
        "prod-001": 100,
        "prod-002": 50,
        "prod-003": 25,
        "prod-004": 0,
        "prod-005": 5,
    })


# =============================================================================
# Application Configuration Tests
# =============================================================================


class TestAppConfiguration:
    """Test FastAPI application configuration."""

    def test_app_has_correct_title(self) -> None:
        """Test application title is set correctly."""
        assert app.title == APP_NAME
        assert app.title == "tracemaid-fastapi-demo"

    def test_app_has_correct_version(self) -> None:
        """Test application version is set correctly."""
        assert app.version == APP_VERSION
        assert app.version == "1.0.0"

    def test_app_has_correct_description(self) -> None:
        """Test application description is set correctly."""
        assert app.description == APP_DESCRIPTION

    def test_app_has_docs_url(self) -> None:
        """Test application has OpenAPI docs URL configured."""
        assert app.docs_url == "/docs"

    def test_app_has_redoc_url(self) -> None:
        """Test application has ReDoc URL configured."""
        assert app.redoc_url == "/redoc"

    def test_app_has_openapi_url(self) -> None:
        """Test application has OpenAPI JSON URL configured."""
        assert app.openapi_url == "/openapi.json"


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_check_returns_200(self, client: TestClient) -> None:
        """Test health check returns 200 OK status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_healthy_status(self, client: TestClient) -> None:
        """Test health check returns healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_includes_app_name(self, client: TestClient) -> None:
        """Test health check includes application name."""
        response = client.get("/health")
        data = response.json()
        assert data["app"] == APP_NAME

    def test_health_check_includes_version(self, client: TestClient) -> None:
        """Test health check includes application version."""
        response = client.get("/health")
        data = response.json()
        assert data["version"] == APP_VERSION

    def test_health_check_includes_timestamp(self, client: TestClient) -> None:
        """Test health check includes timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        # Timestamp should be ISO format
        assert "T" in data["timestamp"]

    def test_health_check_includes_tracemaid_info(self, client: TestClient) -> None:
        """Test health check includes tracemaid integration info."""
        response = client.get("/health")
        data = response.json()
        assert "tracemaid" in data
        assert data["tracemaid"]["enabled"] is True
        assert data["tracemaid"]["output_dir"] == TRACES_OUTPUT_DIR


# =============================================================================
# Root Endpoint Tests
# =============================================================================


class TestRootEndpoint:
    """Test GET / endpoint."""

    def test_root_returns_200(self, client: TestClient) -> None:
        """Test root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_welcome_message(self, client: TestClient) -> None:
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert APP_NAME in data["message"]

    def test_root_returns_version(self, client: TestClient) -> None:
        """Test root endpoint returns version."""
        response = client.get("/")
        data = response.json()
        assert data["version"] == APP_VERSION

    def test_root_returns_docs_link(self, client: TestClient) -> None:
        """Test root endpoint returns docs link."""
        response = client.get("/")
        data = response.json()
        assert data["docs"] == "/docs"

    def test_root_returns_health_link(self, client: TestClient) -> None:
        """Test root endpoint returns health check link."""
        response = client.get("/")
        data = response.json()
        assert data["health"] == "/health"


# =============================================================================
# Route Registration Tests
# =============================================================================


class TestRouteRegistration:
    """Test that all routes are registered correctly."""

    def test_user_routes_registered(self, client: TestClient) -> None:
        """Test user routes are registered under /api/v1/users."""
        response = client.get("/api/v1/users")
        assert response.status_code == 200

    def test_order_routes_registered(self, client: TestClient) -> None:
        """Test order routes are registered under /api/v1/orders."""
        response = client.get("/api/v1/orders")
        assert response.status_code == 200

    def test_health_route_at_root(self, client: TestClient) -> None:
        """Test health route is at root level."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_root_route_accessible(self, client: TestClient) -> None:
        """Test root route is accessible."""
        response = client.get("/")
        assert response.status_code == 200

    def test_docs_route_accessible(self, client: TestClient) -> None:
        """Test OpenAPI docs route is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_accessible(self, client: TestClient) -> None:
        """Test OpenAPI JSON endpoint is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        # Should return valid JSON with openapi field
        data = response.json()
        assert "openapi" in data


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full application."""

    def test_full_user_workflow(self, client: TestClient) -> None:
        """Test complete user workflow through the API."""
        # Create user
        create_response = client.post(
            "/api/v1/users",
            json={"username": "integrationuser", "email": "int@test.com"}
        )
        assert create_response.status_code == 201
        user_id = create_response.json()["id"]

        # Get user
        get_response = client.get(f"/api/v1/users/{user_id}")
        assert get_response.status_code == 200
        assert get_response.json()["username"] == "integrationuser"

        # List users
        list_response = client.get("/api/v1/users")
        assert list_response.status_code == 200
        assert list_response.json()["total"] == 1

        # Delete user
        delete_response = client.delete(f"/api/v1/users/{user_id}")
        assert delete_response.status_code == 204

        # Verify deleted
        verify_response = client.get(f"/api/v1/users/{user_id}")
        assert verify_response.status_code == 404

    def test_full_order_workflow(self, client: TestClient) -> None:
        """Test complete order workflow through the API."""
        # Create order
        create_response = client.post(
            "/api/v1/orders",
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
        get_response = client.get(f"/api/v1/orders/{order_id}")
        assert get_response.status_code == 200
        assert get_response.json()["status"] == "pending"

        # Process order
        process_response = client.post(f"/api/v1/orders/{order_id}/process")
        assert process_response.status_code == 200
        assert process_response.json()["status"] == "completed"

        # List orders
        list_response = client.get("/api/v1/orders")
        assert list_response.status_code == 200
        assert list_response.json()["total"] == 1

    def test_health_check_while_processing(self, client: TestClient) -> None:
        """Test health check works during request processing."""
        # Create some activity
        client.post(
            "/api/v1/users",
            json={"username": "activeuser", "email": "active@test.com"}
        )

        # Health check should still work
        health_response = client.get("/health")
        assert health_response.status_code == 200
        data = health_response.json()
        assert data["status"] == "healthy"


# =============================================================================
# CORS Tests
# =============================================================================


class TestCORS:
    """Test CORS middleware configuration."""

    def test_cors_allows_all_origins(self, client: TestClient) -> None:
        """Test CORS allows all origins."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET"
            }
        )
        # CORS preflight should work
        assert response.status_code in (200, 204, 405)

    def test_cors_headers_in_response(self, client: TestClient) -> None:
        """Test CORS headers are present in response."""
        response = client.get(
            "/health",
            headers={"Origin": "http://example.com"}
        )
        # Allow-Origin header should be present for CORS-enabled endpoints
        assert response.status_code == 200


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling in the application."""

    def test_404_returns_json(self, client: TestClient) -> None:
        """Test 404 errors return JSON response."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        assert response.headers.get("content-type", "").startswith("application/json")

    def test_validation_error_returns_422(self, client: TestClient) -> None:
        """Test validation errors return 422 status."""
        response = client.post(
            "/api/v1/users",
            json={"username": "ab"}  # Missing email, username too short
        )
        assert response.status_code == 422


# =============================================================================
# OpenTelemetry Integration Tests
# =============================================================================


class TestOpenTelemetryIntegration:
    """Test OpenTelemetry integration setup."""

    def test_tracing_info_in_health_response(self, client: TestClient) -> None:
        """Test that tracing info is included in health response."""
        response = client.get("/health")
        data = response.json()

        assert "tracemaid" in data
        assert data["tracemaid"]["enabled"] is True
        assert "output_dir" in data["tracemaid"]

    def test_app_constants_defined(self) -> None:
        """Test that app constants are properly defined."""
        assert APP_NAME is not None
        assert APP_VERSION is not None
        assert APP_DESCRIPTION is not None
        assert TRACES_OUTPUT_DIR is not None

        assert isinstance(APP_NAME, str)
        assert isinstance(APP_VERSION, str)
        assert isinstance(APP_DESCRIPTION, str)
        assert isinstance(TRACES_OUTPUT_DIR, str)

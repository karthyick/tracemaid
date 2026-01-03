#!/usr/bin/env python
"""
Demo runner script for FastAPI tracemaid integration demonstration.

This script starts the FastAPI demo server, runs sample requests to demonstrate
all tracemaid features, and displays formatted log output showing trace
capabilities.

Usage:
    python run_demo.py

Features Demonstrated:
    - Automatic trace ID generation and propagation
    - Correlation ID injection across requests
    - Request/response logging with trace context
    - All log levels: DEBUG, INFO, WARNING, ERROR
    - Service-level span generation
    - End-to-end distributed tracing
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import signal
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "http://127.0.0.1:8000"
STARTUP_TIMEOUT = 10  # seconds to wait for server startup


# =============================================================================
# Console Output Formatting
# =============================================================================

class Colors:
    """ANSI color codes for console output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str) -> None:
    """Print a formatted header."""
    print()
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print()


def print_subheader(text: str) -> None:
    """Print a formatted subheader."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'-' * 50}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'-' * 50}{Colors.ENDC}")


def print_request(method: str, endpoint: str, trace_id: Optional[str] = None) -> None:
    """Print a formatted request."""
    print()
    trace_info = f" (trace_id: {trace_id[:8]}...)" if trace_id else ""
    print(f"{Colors.YELLOW}>>> {method} {endpoint}{trace_info}{Colors.ENDC}")


def print_response(status_code: int, data: Any, headers: Dict[str, str]) -> None:
    """Print a formatted response."""
    if status_code >= 400:
        color = Colors.RED
        status_text = "ERROR"
    elif status_code >= 300:
        color = Colors.YELLOW
        status_text = "REDIRECT"
    else:
        color = Colors.GREEN
        status_text = "SUCCESS"

    print(f"{color}<<< {status_code} {status_text}{Colors.ENDC}")

    # Print trace headers
    trace_id = headers.get("x-trace-id", "N/A")
    span_id = headers.get("x-span-id", "N/A")
    duration = headers.get("x-request-duration-ms", "N/A")

    print(f"    {Colors.BLUE}Trace ID:{Colors.ENDC} {trace_id}")
    print(f"    {Colors.BLUE}Span ID:{Colors.ENDC} {span_id}")
    print(f"    {Colors.BLUE}Duration:{Colors.ENDC} {duration}ms")

    # Print response body (formatted)
    if data:
        formatted = json.dumps(data, indent=2)
        for line in formatted.split("\n")[:10]:  # Limit to 10 lines
            print(f"    {line}")
        if len(formatted.split("\n")) > 10:
            print(f"    {Colors.CYAN}... (truncated){Colors.ENDC}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {text}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {text}")


# =============================================================================
# HTTP Client
# =============================================================================

class DemoClient:
    """HTTP client for demo requests."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.trace_id: Optional[str] = None

    def set_trace_id(self, trace_id: str) -> None:
        """Set trace ID for subsequent requests."""
        self.trace_id = trace_id

    def clear_trace_id(self) -> None:
        """Clear the trace ID."""
        self.trace_id = None

    def get_headers(self) -> Dict[str, str]:
        """Get request headers with optional trace ID."""
        headers = {"Content-Type": "application/json"}
        if self.trace_id:
            headers["X-Trace-Id"] = self.trace_id
        return headers

    def get(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> httpx.Response:
        """Make a GET request."""
        print_request("GET", endpoint, self.trace_id)
        response = self.client.get(
            f"{self.base_url}{endpoint}",
            headers=self.get_headers(),
            params=params
        )
        self._print_response(response)
        return response

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """Make a POST request."""
        print_request("POST", endpoint, self.trace_id)
        response = self.client.post(
            f"{self.base_url}{endpoint}",
            headers=self.get_headers(),
            json=data
        )
        self._print_response(response)
        return response

    def delete(self, endpoint: str) -> httpx.Response:
        """Make a DELETE request."""
        print_request("DELETE", endpoint, self.trace_id)
        response = self.client.delete(
            f"{self.base_url}{endpoint}",
            headers=self.get_headers()
        )
        self._print_response(response)
        return response

    def _print_response(self, response: httpx.Response) -> None:
        """Print response details."""
        try:
            data = response.json()
        except Exception:
            data = response.text
        print_response(response.status_code, data, dict(response.headers))

    def close(self) -> None:
        """Close the client."""
        self.client.close()


# =============================================================================
# Demo Scenarios
# =============================================================================

def wait_for_server(base_url: str, timeout: int = 10) -> bool:
    """Wait for the server to be ready.

    Args:
        base_url: Base URL of the server
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False otherwise
    """
    print_info(f"Waiting for server at {base_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                print_success("Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(0.5)
        print(".", end="", flush=True)

    print()
    print_error(f"Server did not start within {timeout} seconds")
    return False


def demo_health_check(client: DemoClient) -> None:
    """Demonstrate health check endpoint."""
    print_subheader("1. Health Check Demonstration")
    print_info("Testing the health check endpoint with automatic trace generation...")

    client.get("/health")
    print_success("Health check shows tracemaid middleware is active!")


def demo_user_operations(client: DemoClient) -> str:
    """Demonstrate user CRUD operations.

    Returns:
        Created user ID for subsequent demos
    """
    print_subheader("2. User CRUD Operations")
    print_info("Demonstrating user operations with trace context propagation...")

    # Create user
    print("\n2.1 Creating a new user:")
    response = client.post("/api/v1/users", {
        "username": "demo_user",
        "email": "demo@tracemaid.example"
    })
    user_id = response.json().get("id", "")

    # Get user
    print("\n2.2 Retrieving the user:")
    client.get(f"/api/v1/users/{user_id}")

    # List users
    print("\n2.3 Listing all users:")
    client.get("/api/v1/users")

    print_success("User operations completed with full tracing!")
    return user_id


def demo_order_operations(client: DemoClient, user_id: str) -> str:
    """Demonstrate order operations.

    Args:
        user_id: User ID for order creation

    Returns:
        Created order ID for subsequent demos
    """
    print_subheader("3. Order Operations")
    print_info("Demonstrating order lifecycle with detailed trace spans...")

    # Create order
    print("\n3.1 Creating a new order:")
    response = client.post("/api/v1/orders", {
        "user_id": user_id,
        "items": [
            {"product_id": "prod-001", "quantity": 2, "unit_price": 29.99},
            {"product_id": "prod-002", "quantity": 1, "unit_price": 49.99}
        ]
    })
    order_id = response.json().get("id", "")

    # Get order
    print("\n3.2 Retrieving the order:")
    client.get(f"/api/v1/orders/{order_id}")

    # Process order
    print("\n3.3 Processing the order (multi-step with trace spans):")
    client.post(f"/api/v1/orders/{order_id}/process")

    print_success("Order operations completed with detailed trace spans!")
    return order_id


def demo_error_scenarios(client: DemoClient) -> None:
    """Demonstrate error handling and logging."""
    print_subheader("4. Error Scenarios & Log Levels")
    print_info("Demonstrating WARNING and ERROR level logs...")

    # Not found
    print("\n4.1 Getting non-existent user (triggers WARNING):")
    client.get("/api/v1/users/nonexistent-user-id")

    # Not found order
    print("\n4.2 Getting non-existent order (triggers WARNING):")
    client.get("/api/v1/orders/nonexistent-order-id")

    # Duplicate user (triggers ERROR)
    print("\n4.3 Creating duplicate user (triggers ERROR):")
    client.post("/api/v1/users", {
        "username": "demo_user",
        "email": "duplicate@example.com"
    })

    print_success("Error scenarios demonstrated - check logs for WARNING/ERROR entries!")


def demo_trace_propagation(client: DemoClient) -> None:
    """Demonstrate trace ID propagation."""
    print_subheader("5. Trace ID Propagation")
    print_info("Demonstrating manual trace ID propagation across requests...")

    # Set a custom trace ID
    custom_trace_id = "demo1234567890abcdef1234567890ab"
    client.set_trace_id(custom_trace_id)
    print(f"\n{Colors.CYAN}Setting custom trace ID: {custom_trace_id}{Colors.ENDC}")

    # Make requests with same trace ID
    print("\n5.1 First request with custom trace ID:")
    client.post("/api/v1/users", {
        "username": "traced_user_1",
        "email": "traced1@example.com"
    })

    print("\n5.2 Second request with same trace ID:")
    client.post("/api/v1/users", {
        "username": "traced_user_2",
        "email": "traced2@example.com"
    })

    print("\n5.3 Third request with same trace ID:")
    client.get("/api/v1/users")

    client.clear_trace_id()
    print_success("All three requests share the same trace ID for correlation!")


def demo_trace_data_retrieval(client: DemoClient) -> None:
    """Demonstrate trace data retrieval."""
    print_subheader("6. Trace Data Retrieval")
    print_info("Retrieving collected trace data from services...")

    # Get middleware trace data
    print("\n6.1 Getting middleware request trace data:")
    response = client.get("/trace-data")
    trace_count = response.json().get("trace_count", 0)
    print(f"    {Colors.GREEN}Total traces collected: {trace_count}{Colors.ENDC}")

    # Get user service trace data
    print("\n6.2 Getting user service trace spans:")
    response = client.get("/api/v1/users/trace-data")
    span_count = response.json().get("total_spans", 0)
    print(f"    {Colors.GREEN}User service spans: {span_count}{Colors.ENDC}")

    # Get order service trace data
    print("\n6.3 Getting order service trace spans:")
    response = client.get("/api/v1/orders/trace-data")
    span_count = response.json().get("total_spans", 0)
    print(f"    {Colors.GREEN}Order service spans: {span_count}{Colors.ENDC}")

    print_success("Trace data retrieval complete!")


def demo_order_cancellation(client: DemoClient) -> None:
    """Demonstrate order cancellation workflow."""
    print_subheader("7. Order Cancellation")
    print_info("Demonstrating order cancellation with trace context...")

    # Create an order to cancel
    print("\n7.1 Creating order for cancellation:")
    response = client.post("/api/v1/orders", {
        "user_id": "cancel-demo-user",
        "items": [
            {"product_id": "prod-003", "quantity": 1, "unit_price": 19.99}
        ]
    })
    order_id = response.json().get("id", "")

    # Cancel order
    print("\n7.2 Cancelling the order:")
    client.post(f"/api/v1/orders/{order_id}/cancel", {
        "reason": "demo_cancellation"
    })

    print_success("Order cancellation workflow completed!")


def demo_cleanup(client: DemoClient, user_id: str) -> None:
    """Cleanup demo data."""
    print_subheader("8. Cleanup")
    print_info("Cleaning up demo data and trace collections...")

    # Delete demo user
    print("\n8.1 Deleting demo user:")
    client.delete(f"/api/v1/users/{user_id}")

    # Clear trace data
    print("\n8.2 Clearing trace data:")
    client.post("/trace-data/clear")
    client.post("/api/v1/users/trace-data/clear")
    client.post("/api/v1/orders/trace-data/clear")

    print_success("Cleanup complete!")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_demos_against_running_server(base_url: str) -> None:
    """Run demo scenarios against a running server.

    Args:
        base_url: Base URL of the running server
    """
    client = DemoClient(base_url)

    try:
        # Run demo scenarios
        demo_health_check(client)

        user_id = demo_user_operations(client)

        order_id = demo_order_operations(client, user_id)

        demo_error_scenarios(client)

        demo_trace_propagation(client)

        demo_trace_data_retrieval(client)

        demo_order_cancellation(client)

        demo_cleanup(client, user_id)

        # Summary
        print_header("Demo Complete!")
        print(f"""
{Colors.GREEN}Tracemaid Integration Features Demonstrated:{Colors.ENDC}

  1. {Colors.CYAN}Automatic Trace ID Generation{Colors.ENDC}
     - Every request gets a unique trace ID
     - Trace IDs are returned in response headers

  2. {Colors.CYAN}Trace ID Propagation{Colors.ENDC}
     - Custom trace IDs can be provided via X-Trace-Id header
     - Trace IDs are consistent across related requests

  3. {Colors.CYAN}Request/Response Logging{Colors.ENDC}
     - All requests are logged with trace context
     - Duration tracking in response headers

  4. {Colors.CYAN}Multi-Level Logging{Colors.ENDC}
     - DEBUG: Span creation details
     - INFO: Request start/complete, operation success
     - WARNING: Not found, low inventory
     - ERROR: Validation failures, processing errors

  5. {Colors.CYAN}Service-Level Spans{Colors.ENDC}
     - Detailed operation spans within services
     - Parent-child span relationships

  6. {Colors.CYAN}Trace Data Retrieval{Colors.ENDC}
     - Middleware and service-level trace collection
     - OTLP-compatible span format

{Colors.BOLD}To visualize traces with tracemaid:{Colors.ENDC}

    from tracemaid import OTelParser, MermaidGenerator

    parser = OTelParser()
    trace = parser.parse_otlp(trace_data)

    generator = MermaidGenerator()
    diagram = generator.generate(trace)
    print(diagram)
""")

    finally:
        client.close()


def start_server_and_run_demos() -> None:
    """Start the server and run demos."""
    import multiprocessing
    import uvicorn

    def run_server():
        """Run the FastAPI server in a subprocess."""
        from examples.fastapi_demo.main import app
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

    print_header("Tracemaid FastAPI Demo")
    print_info("Starting FastAPI demo server...")

    # Start server in subprocess
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    try:
        # Wait for server to be ready
        if not wait_for_server(BASE_URL, STARTUP_TIMEOUT):
            print_error("Failed to start server. Exiting.")
            server_process.terminate()
            sys.exit(1)

        # Run demos
        run_demos_against_running_server(BASE_URL)

    finally:
        # Cleanup
        print_info("Shutting down server...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        print_success("Server stopped.")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tracemaid FastAPI demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server and run demos
    python run_demo.py

    # Run demos against already running server
    python run_demo.py --url http://localhost:8000

    # Run with custom server URL
    python run_demo.py --url http://192.168.1.100:8000
        """
    )
    parser.add_argument(
        "--url",
        default=None,
        help="URL of already running server (skips server startup)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    args = parser.parse_args()

    # Disable colors if requested or on Windows without color support
    if args.no_color or (sys.platform == "win32" and not os.environ.get("TERM")):
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''

    if args.url:
        # Run against existing server
        print_header("Tracemaid FastAPI Demo")
        print_info(f"Connecting to existing server at {args.url}...")

        if wait_for_server(args.url, timeout=5):
            run_demos_against_running_server(args.url)
        else:
            print_error(f"Cannot connect to server at {args.url}")
            sys.exit(1)
    else:
        # Start server and run demos
        start_server_and_run_demos()


if __name__ == "__main__":
    main()

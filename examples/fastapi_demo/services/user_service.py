"""
User service module demonstrating tracemaid integration.

This module provides a UserService class that simulates user CRUD operations
and generates OpenTelemetry-compatible trace data for analysis with tracemaid.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from examples.fastapi_demo.services.tracing import SpanData

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class User:
    """Represents a user entity.

    Attributes:
        id: Unique identifier for the user
        username: User's username
        email: User's email address
        created_at: Timestamp of creation
        is_active: Whether the user account is active
    """
    id: str
    username: str
    email: str
    created_at: float = field(default_factory=time.time)
    is_active: bool = True


class UserService:
    """Service for user CRUD operations with trace data generation.

    This service simulates user operations and generates OpenTelemetry-compatible
    trace data that can be analyzed using tracemaid.

    Example:
        >>> service = UserService()
        >>> user, traces = service.create_user("john_doe", "john@example.com")
        >>> # traces can be parsed by tracemaid.OTelParser
        >>> from tracemaid import OTelParser
        >>> parser = OTelParser()
        >>> trace = parser.parse_otlp({"spans": [t.to_otlp_dict() for t in traces]})
    """

    SERVICE_NAME = "user-service"

    def __init__(self) -> None:
        """Initialize the UserService with an in-memory user store."""
        self._users: Dict[str, User] = {}
        self._traces: List[SpanData] = []
        logger.info(
            "UserService initialized",
            extra={"service": self.SERVICE_NAME}
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
            end_time=0,  # Set when span completes
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
        logger.info(
            "Span completed: %s (%.2fms)",
            span.operation,
            duration_ms,
            extra={
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "status": status,
                "duration_ms": duration_ms
            }
        )

    def get_user(
        self,
        user_id: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[Optional[User], List[SpanData]]:
        """Retrieve a user by ID.

        Args:
            user_id: The ID of the user to retrieve
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (User or None if not found, list of trace spans)
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        # Create root span for get_user operation
        root_span = self._create_span(
            operation="get_user",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={"user.id": user_id, "operation.type": "read"}
        )

        logger.info(
            "Getting user",
            extra={
                "user_id": user_id,
                "trace_id": trace_id,
                "span_id": root_span.span_id
            }
        )

        # Create child span for cache check simulation
        cache_span = self._create_span(
            operation="check_user_cache",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"cache.type": "memory"}
        )

        # Simulate cache lookup
        time.sleep(0.001)  # 1ms cache check
        cache_hit = user_id in self._users

        self._complete_span(
            cache_span,
            status="OK",
            additional_attributes={"cache.hit": str(cache_hit).lower()}
        )
        spans.append(cache_span)

        # Get user from store
        user = self._users.get(user_id)

        if user:
            self._complete_span(
                root_span,
                status="OK",
                additional_attributes={
                    "user.found": "true",
                    "user.username": user.username
                }
            )
            logger.info(
                "User found: %s",
                user.username,
                extra={"user_id": user_id}
            )
        else:
            self._complete_span(
                root_span,
                status="OK",  # Not an error, just not found
                additional_attributes={"user.found": "false"}
            )
            logger.warning(
                "User not found",
                extra={"user_id": user_id}
            )

        spans.append(root_span)
        return user, spans

    def create_user(
        self,
        username: str,
        email: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[User, List[SpanData]]:
        """Create a new user.

        Args:
            username: The username for the new user
            email: The email address for the new user
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (created User, list of trace spans)

        Raises:
            ValueError: If username or email is empty
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        # Create root span for create_user operation
        root_span = self._create_span(
            operation="create_user",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={
                "user.username": username,
                "user.email": email,
                "operation.type": "write"
            }
        )

        logger.info(
            "Creating user",
            extra={
                "username": username,
                "email": email,
                "trace_id": trace_id,
                "span_id": root_span.span_id
            }
        )

        # Validation span
        validation_span = self._create_span(
            operation="validate_user_input",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"validation.type": "input"}
        )

        # Perform validation
        validation_errors: List[str] = []
        if not username or not username.strip():
            validation_errors.append("username is required")
        if not email or not email.strip():
            validation_errors.append("email is required")
        if username and len(username) < 3:
            validation_errors.append("username must be at least 3 characters")

        time.sleep(0.001)  # Simulate validation time

        if validation_errors:
            self._complete_span(
                validation_span,
                status="ERROR",
                additional_attributes={"validation.errors": ", ".join(validation_errors)}
            )
            spans.append(validation_span)

            self._complete_span(
                root_span,
                status="ERROR",
                additional_attributes={"error.message": "Validation failed"}
            )
            spans.append(root_span)

            logger.error(
                "User validation failed",
                extra={"errors": validation_errors}
            )
            raise ValueError(f"Validation failed: {', '.join(validation_errors)}")

        self._complete_span(validation_span, status="OK")
        spans.append(validation_span)

        # Check for duplicate username
        duplicate_check_span = self._create_span(
            operation="check_duplicate_username",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"check.type": "uniqueness"}
        )

        time.sleep(0.002)  # Simulate database lookup

        for existing_user in self._users.values():
            if existing_user.username == username:
                self._complete_span(
                    duplicate_check_span,
                    status="ERROR",
                    additional_attributes={"duplicate.found": "true"}
                )
                spans.append(duplicate_check_span)

                self._complete_span(
                    root_span,
                    status="ERROR",
                    additional_attributes={"error.message": "Username already exists"}
                )
                spans.append(root_span)

                logger.error(
                    "Duplicate username",
                    extra={"username": username}
                )
                raise ValueError(f"Username '{username}' already exists")

        self._complete_span(
            duplicate_check_span,
            status="OK",
            additional_attributes={"duplicate.found": "false"}
        )
        spans.append(duplicate_check_span)

        # Create user
        persist_span = self._create_span(
            operation="persist_user",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"persistence.type": "memory"}
        )

        user_id = self._generate_span_id()  # Reuse UUID generation
        user = User(
            id=user_id,
            username=username.strip(),
            email=email.strip()
        )

        self._users[user_id] = user
        time.sleep(0.003)  # Simulate database write

        self._complete_span(
            persist_span,
            status="OK",
            additional_attributes={"user.id": user_id}
        )
        spans.append(persist_span)

        # Complete root span
        self._complete_span(
            root_span,
            status="OK",
            additional_attributes={
                "user.id": user_id,
                "user.created": "true"
            }
        )
        spans.append(root_span)

        logger.info(
            "User created successfully",
            extra={
                "user_id": user_id,
                "username": username
            }
        )

        return user, spans

    def delete_user(
        self,
        user_id: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[bool, List[SpanData]]:
        """Delete a user by ID.

        Args:
            user_id: The ID of the user to delete
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (success boolean, list of trace spans)
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        # Create root span for delete_user operation
        root_span = self._create_span(
            operation="delete_user",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={"user.id": user_id, "operation.type": "delete"}
        )

        logger.info(
            "Deleting user",
            extra={
                "user_id": user_id,
                "trace_id": trace_id,
                "span_id": root_span.span_id
            }
        )

        # Check if user exists
        lookup_span = self._create_span(
            operation="lookup_user_for_delete",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"lookup.reason": "pre-delete-check"}
        )

        time.sleep(0.001)  # Simulate lookup
        user = self._users.get(user_id)

        self._complete_span(
            lookup_span,
            status="OK",
            additional_attributes={"user.exists": str(user is not None).lower()}
        )
        spans.append(lookup_span)

        if not user:
            self._complete_span(
                root_span,
                status="OK",  # Not an error, user just doesn't exist
                additional_attributes={
                    "delete.success": "false",
                    "delete.reason": "user_not_found"
                }
            )
            spans.append(root_span)

            logger.warning(
                "Cannot delete user: not found",
                extra={"user_id": user_id}
            )
            return False, spans

        # Soft delete span (deactivate)
        deactivate_span = self._create_span(
            operation="deactivate_user",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"deactivation.type": "soft_delete"}
        )

        user.is_active = False
        time.sleep(0.001)

        self._complete_span(deactivate_span, status="OK")
        spans.append(deactivate_span)

        # Hard delete span
        remove_span = self._create_span(
            operation="remove_user_from_store",
            parent_span_id=root_span.span_id,
            trace_id=trace_id,
            attributes={"removal.type": "hard_delete"}
        )

        del self._users[user_id]
        time.sleep(0.002)  # Simulate database delete

        self._complete_span(remove_span, status="OK")
        spans.append(remove_span)

        # Complete root span
        self._complete_span(
            root_span,
            status="OK",
            additional_attributes={"delete.success": "true"}
        )
        spans.append(root_span)

        logger.info(
            "User deleted successfully",
            extra={"user_id": user_id}
        )

        return True, spans

    def get_trace_data(self) -> List[Dict[str, Any]]:
        """Get all collected trace data in OTLP format.

        Returns:
            List of span dictionaries in OTLP format
        """
        return [span.to_otlp_dict() for span in self._traces]

    def clear_traces(self) -> None:
        """Clear all collected trace data."""
        self._traces.clear()
        logger.debug("Trace data cleared")

    def list_users(
        self,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None
    ) -> tuple[List[User], List[SpanData]]:
        """List all users.

        Args:
            trace_id: Optional trace ID for distributed tracing
            parent_span_id: Optional parent span ID

        Returns:
            Tuple of (list of Users, list of trace spans)
        """
        trace_id = trace_id or self._generate_trace_id()
        spans: List[SpanData] = []

        root_span = self._create_span(
            operation="list_users",
            parent_span_id=parent_span_id,
            trace_id=trace_id,
            attributes={"operation.type": "read"}
        )

        logger.info(
            "Listing all users",
            extra={
                "trace_id": trace_id,
                "span_id": root_span.span_id
            }
        )

        time.sleep(0.002)  # Simulate query time
        users = list(self._users.values())

        self._complete_span(
            root_span,
            status="OK",
            additional_attributes={"users.count": str(len(users))}
        )
        spans.append(root_span)

        logger.info(
            "Listed %d users",
            len(users)
        )

        return users, spans

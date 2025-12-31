"""
User routes module for FastAPI demo application.

This module provides API endpoints for user operations with tracemaid
request tracing and correlation ID propagation.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field

from examples.fastapi_demo.services import UserService, User, SpanData

# Configure module logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/users", tags=["users"])

# Service instance (in production, use dependency injection)
_user_service = UserService()


class UserCreateRequest(BaseModel):
    """Request model for creating a user.

    Attributes:
        username: Username for the new user (3-50 characters)
        email: Valid email address for the user
    """
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Username for the new user"
    )
    email: EmailStr = Field(..., description="Email address for the user")


class UserResponse(BaseModel):
    """Response model for user data.

    Attributes:
        id: Unique user identifier
        username: User's username
        email: User's email address
        created_at: Unix timestamp of creation
        is_active: Whether the user is active
    """
    id: str
    username: str
    email: str
    created_at: float
    is_active: bool


class UserListResponse(BaseModel):
    """Response model for listing users.

    Attributes:
        users: List of user objects
        total: Total count of users
    """
    users: List[UserResponse]
    total: int


class TraceResponse(BaseModel):
    """Response model including trace data.

    Attributes:
        data: The response data
        trace_id: The trace ID for this request
        spans: Number of spans generated
    """
    data: Any
    trace_id: str
    spans_count: int


def _get_trace_context(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Extract trace context from request headers.

    Args:
        request: The incoming FastAPI request

    Returns:
        Tuple of (trace_id, parent_span_id)
    """
    trace_id = request.headers.get("X-Trace-Id")
    parent_span_id = request.headers.get("X-Parent-Span-Id")

    # Generate trace_id if not provided
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


def _user_to_response(user: User) -> UserResponse:
    """Convert User dataclass to UserResponse model.

    Args:
        user: User dataclass instance

    Returns:
        UserResponse model instance
    """
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        is_active=user.is_active
    )


# ============================================================================
# Static routes (must be defined before dynamic routes with path parameters)
# ============================================================================

@router.get("/trace-data", response_model=Dict[str, Any])
async def get_trace_data() -> Dict[str, Any]:
    """Get all collected trace data from the user service.

    This endpoint is useful for debugging and analyzing traces.

    Returns:
        Dictionary containing all collected trace spans in OTLP format
    """
    trace_data = _user_service.get_trace_data()
    return {
        "service": "user-service",
        "spans": trace_data,
        "total_spans": len(trace_data)
    }


@router.post("/trace-data/clear", status_code=204)
async def clear_trace_data() -> None:
    """Clear all collected trace data from the user service.

    This endpoint is useful for resetting trace collection between tests.
    """
    _user_service.clear_traces()
    logger.info("User service trace data cleared")


@router.get("", response_model=UserListResponse)
async def list_users(
    request: Request,
    response: Response,
    include_trace: bool = False
) -> UserListResponse | TraceResponse:
    """List all users.

    Retrieves all users from the service with full tracemaid tracing.

    Args:
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        include_trace: Whether to include trace data in response

    Returns:
        UserListResponse with list of users and total count
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "GET /users - trace_id: %s",
        trace_id,
        extra={"trace_id": trace_id}
    )

    users, spans = _user_service.list_users(
        trace_id=trace_id,
        parent_span_id=parent_span_id
    )

    _add_trace_headers(response, trace_id, spans)

    user_responses = [_user_to_response(user) for user in users]
    list_response = UserListResponse(users=user_responses, total=len(users))

    if include_trace:
        return TraceResponse(
            data=list_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return list_response


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreateRequest,
    request: Request,
    response: Response,
    include_trace: bool = False
) -> UserResponse | TraceResponse:
    """Create a new user.

    Creates a user with validation and full tracemaid tracing.

    Args:
        user_data: The user creation request data
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        include_trace: Whether to include trace data in response

    Returns:
        UserResponse with created user data

    Raises:
        HTTPException: 400 for validation errors, 409 for duplicate username
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "POST /users - username: %s, trace_id: %s",
        user_data.username,
        trace_id,
        extra={
            "username": user_data.username,
            "email": user_data.email,
            "trace_id": trace_id
        }
    )

    try:
        user, spans = _user_service.create_user(
            username=user_data.username,
            email=user_data.email,
            trace_id=trace_id,
            parent_span_id=parent_span_id
        )
    except ValueError as e:
        error_message = str(e)
        logger.error(
            "User creation failed: %s",
            error_message,
            extra={"username": user_data.username, "trace_id": trace_id}
        )

        if "already exists" in error_message.lower():
            raise HTTPException(
                status_code=409,
                detail=error_message
            )
        raise HTTPException(
            status_code=400,
            detail=error_message
        )

    _add_trace_headers(response, trace_id, spans)

    user_response = _user_to_response(user)

    if include_trace:
        return TraceResponse(
            data=user_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return user_response


# ============================================================================
# Dynamic routes (with path parameters)
# ============================================================================

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    request: Request,
    response: Response,
    include_trace: bool = False
) -> UserResponse | TraceResponse:
    """Get a user by ID.

    Retrieves a user from the service layer with full tracemaid tracing.
    Trace context is propagated via headers.

    Args:
        user_id: The unique identifier of the user
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)
        include_trace: Whether to include trace data in response

    Returns:
        UserResponse with user data, or TraceResponse if include_trace=True

    Raises:
        HTTPException: 404 if user not found
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "GET /users/%s - trace_id: %s",
        user_id,
        trace_id,
        extra={"user_id": user_id, "trace_id": trace_id}
    )

    user, spans = _user_service.get_user(
        user_id=user_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id
    )

    _add_trace_headers(response, trace_id, spans)

    if not user:
        logger.warning(
            "User not found: %s",
            user_id,
            extra={"user_id": user_id, "trace_id": trace_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"User with ID '{user_id}' not found"
        )

    user_response = _user_to_response(user)

    if include_trace:
        return TraceResponse(
            data=user_response.model_dump(),
            trace_id=trace_id,
            spans_count=len(spans)
        )

    return user_response


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    request: Request,
    response: Response
) -> None:
    """Delete a user by ID.

    Soft deletes and then removes a user with full tracemaid tracing.

    Args:
        user_id: The unique identifier of the user to delete
        request: The incoming request (for trace context)
        response: The outgoing response (for trace headers)

    Raises:
        HTTPException: 404 if user not found
    """
    trace_id, parent_span_id = _get_trace_context(request)

    logger.info(
        "DELETE /users/%s - trace_id: %s",
        user_id,
        trace_id,
        extra={"user_id": user_id, "trace_id": trace_id}
    )

    success, spans = _user_service.delete_user(
        user_id=user_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id
    )

    _add_trace_headers(response, trace_id, spans)

    if not success:
        logger.warning(
            "User not found for deletion: %s",
            user_id,
            extra={"user_id": user_id, "trace_id": trace_id}
        )
        raise HTTPException(
            status_code=404,
            detail=f"User with ID '{user_id}' not found"
        )

    logger.info(
        "User deleted successfully: %s",
        user_id,
        extra={"user_id": user_id, "trace_id": trace_id}
    )

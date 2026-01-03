"""
User routes module for FastAPI demo application.

This module provides API endpoints for user operations with automatic
OpenTelemetry tracing. Traces are automatically exported as Mermaid
diagrams by TracemaidExporter - no manual trace retrieval needed!
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field

from examples.fastapi_demo.services import UserService, User

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


@router.get("")
async def list_users() -> UserListResponse:
    """List all users.

    Retrieves all users from the service. Traces are automatically
    generated and exported as Mermaid diagrams.

    Returns:
        UserListResponse with list of users and total count
    """
    logger.info("GET /users")

    users = _user_service.list_users()

    user_responses = [_user_to_response(user) for user in users]
    return UserListResponse(users=user_responses, total=len(users))


@router.post("", status_code=201)
async def create_user(user_data: UserCreateRequest) -> UserResponse:
    """Create a new user.

    Creates a user with validation. Traces are automatically generated
    and exported as Mermaid diagrams.

    Args:
        user_data: The user creation request data

    Returns:
        UserResponse with created user data

    Raises:
        HTTPException: 400 for validation errors, 409 for duplicate username
    """
    logger.info("POST /users - username: %s", user_data.username)

    try:
        user = _user_service.create_user(
            username=user_data.username,
            email=user_data.email
        )
    except ValueError as e:
        error_message = str(e)
        logger.error("User creation failed: %s", error_message)

        if "already exists" in error_message.lower():
            raise HTTPException(status_code=409, detail=error_message)
        raise HTTPException(status_code=400, detail=error_message)

    return _user_to_response(user)


@router.get("/{user_id}")
async def get_user(user_id: str) -> UserResponse:
    """Get a user by ID.

    Retrieves a user from the service. Traces are automatically
    generated and exported as Mermaid diagrams.

    Args:
        user_id: The unique identifier of the user

    Returns:
        UserResponse with user data

    Raises:
        HTTPException: 404 if user not found
    """
    logger.info("GET /users/%s", user_id)

    user = _user_service.get_user(user_id=user_id)

    if not user:
        logger.warning("User not found: %s", user_id)
        raise HTTPException(
            status_code=404,
            detail=f"User with ID '{user_id}' not found"
        )

    return _user_to_response(user)


@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: str) -> None:
    """Delete a user by ID.

    Soft deletes and then removes a user. Traces are automatically
    generated and exported as Mermaid diagrams.

    Args:
        user_id: The unique identifier of the user to delete

    Raises:
        HTTPException: 404 if user not found
    """
    logger.info("DELETE /users/%s", user_id)

    success = _user_service.delete_user(user_id=user_id)

    if not success:
        logger.warning("User not found for deletion: %s", user_id)
        raise HTTPException(
            status_code=404,
            detail=f"User with ID '{user_id}' not found"
        )

    logger.info("User deleted successfully: %s", user_id)

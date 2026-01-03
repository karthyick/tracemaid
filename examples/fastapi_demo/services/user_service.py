"""
User service module - Clean implementation without manual tracing.

OpenTelemetry auto-instrumentation (FastAPI) handles span creation automatically.
Tracemaid exporter receives these spans and generates Mermaid diagrams.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Represents a user entity."""
    id: str
    username: str
    email: str
    created_at: float = field(default_factory=time.time)
    is_active: bool = True


class UserService:
    """Service for user operations - no manual tracing code."""

    def __init__(self) -> None:
        self._users: Dict[str, User] = {}
        logger.info("UserService initialized")

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def get_user(self, user_id: str) -> Optional[User]:
        """Retrieve a user by ID."""
        logger.info("Getting user %s", user_id)
        return self._users.get(user_id)

    def create_user(self, username: str, email: str) -> User:
        """Create a new user."""
        logger.info("Creating user: %s", username)

        if not username or not username.strip():
            raise ValueError("Username is required")
        if not email or not email.strip():
            raise ValueError("Email is required")
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")

        # Check for duplicate username
        for existing_user in self._users.values():
            if existing_user.username == username:
                raise ValueError(f"Username '{username}' already exists")

        user_id = self._generate_id()
        user = User(
            id=user_id,
            username=username.strip(),
            email=email.strip()
        )
        self._users[user_id] = user

        logger.info("User created: %s", user_id)
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete a user by ID."""
        logger.info("Deleting user %s", user_id)

        if user_id not in self._users:
            logger.warning("User not found: %s", user_id)
            return False

        del self._users[user_id]
        logger.info("User deleted: %s", user_id)
        return True

    def list_users(self) -> List[User]:
        """List all users."""
        logger.info("Listing users")
        return list(self._users.values())

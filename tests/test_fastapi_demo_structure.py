"""Unit tests for FastAPI demo directory structure validation.

This module verifies that the examples/fastapi_demo directory structure
meets all acceptance criteria for Task 1.
"""

import os
from pathlib import Path

import pytest


# Base path to the tracemaid project root
PROJECT_ROOT = Path(__file__).parent.parent
FASTAPI_DEMO_DIR = PROJECT_ROOT / "examples" / "fastapi_demo"


class TestDirectoryStructure:
    """Test suite for validating directory structure specification."""

    def test_examples_directory_exists(self) -> None:
        """Verify examples directory exists at project root."""
        examples_dir = PROJECT_ROOT / "examples"
        assert examples_dir.exists(), f"Examples directory not found at {examples_dir}"
        assert examples_dir.is_dir(), f"{examples_dir} exists but is not a directory"

    def test_fastapi_demo_directory_exists(self) -> None:
        """Verify fastapi_demo directory exists under examples."""
        assert FASTAPI_DEMO_DIR.exists(), (
            f"fastapi_demo directory not found at {FASTAPI_DEMO_DIR}"
        )
        assert FASTAPI_DEMO_DIR.is_dir(), (
            f"{FASTAPI_DEMO_DIR} exists but is not a directory"
        )

    def test_routes_subdirectory_exists(self) -> None:
        """Verify routes subdirectory exists under fastapi_demo."""
        routes_dir = FASTAPI_DEMO_DIR / "routes"
        assert routes_dir.exists(), f"Routes directory not found at {routes_dir}"
        assert routes_dir.is_dir(), f"{routes_dir} exists but is not a directory"

    def test_services_subdirectory_exists(self) -> None:
        """Verify services subdirectory exists under fastapi_demo."""
        services_dir = FASTAPI_DEMO_DIR / "services"
        assert services_dir.exists(), f"Services directory not found at {services_dir}"
        assert services_dir.is_dir(), f"{services_dir} exists but is not a directory"


class TestInitFiles:
    """Test suite for verifying all __init__.py files are present."""

    def test_fastapi_demo_init_exists(self) -> None:
        """Verify __init__.py exists in fastapi_demo directory."""
        init_file = FASTAPI_DEMO_DIR / "__init__.py"
        assert init_file.exists(), (
            f"__init__.py not found at {init_file}"
        )
        assert init_file.is_file(), f"{init_file} exists but is not a file"

    def test_fastapi_demo_init_has_content(self) -> None:
        """Verify __init__.py in fastapi_demo has proper docstring."""
        init_file = FASTAPI_DEMO_DIR / "__init__.py"
        content = init_file.read_text()
        assert len(content) > 0, "__init__.py should not be empty"
        assert '"""' in content or "'''" in content, (
            "__init__.py should contain a docstring"
        )
        assert "__version__" in content, (
            "__init__.py should define __version__"
        )

    def test_routes_init_exists(self) -> None:
        """Verify __init__.py exists in routes directory."""
        init_file = FASTAPI_DEMO_DIR / "routes" / "__init__.py"
        assert init_file.exists(), f"__init__.py not found at {init_file}"
        assert init_file.is_file(), f"{init_file} exists but is not a file"

    def test_routes_init_has_content(self) -> None:
        """Verify __init__.py in routes has proper docstring."""
        init_file = FASTAPI_DEMO_DIR / "routes" / "__init__.py"
        content = init_file.read_text()
        assert len(content) > 0, "routes/__init__.py should not be empty"
        assert '"""' in content or "'''" in content, (
            "routes/__init__.py should contain a docstring"
        )

    def test_services_init_exists(self) -> None:
        """Verify __init__.py exists in services directory."""
        init_file = FASTAPI_DEMO_DIR / "services" / "__init__.py"
        assert init_file.exists(), f"__init__.py not found at {init_file}"
        assert init_file.is_file(), f"{init_file} exists but is not a file"

    def test_services_init_has_content(self) -> None:
        """Verify __init__.py in services has proper docstring."""
        init_file = FASTAPI_DEMO_DIR / "services" / "__init__.py"
        content = init_file.read_text()
        assert len(content) > 0, "services/__init__.py should not be empty"
        assert '"""' in content or "'''" in content, (
            "services/__init__.py should contain a docstring"
        )


class TestRequirementsTxt:
    """Test suite for validating requirements.txt content."""

    def test_requirements_file_exists(self) -> None:
        """Verify requirements.txt exists in fastapi_demo directory."""
        req_file = FASTAPI_DEMO_DIR / "requirements.txt"
        assert req_file.exists(), f"requirements.txt not found at {req_file}"
        assert req_file.is_file(), f"{req_file} exists but is not a file"

    def test_requirements_contains_fastapi(self) -> None:
        """Verify requirements.txt contains fastapi package."""
        req_file = FASTAPI_DEMO_DIR / "requirements.txt"
        content = req_file.read_text().lower()
        assert "fastapi" in content, (
            "requirements.txt must contain fastapi package"
        )

    def test_requirements_contains_uvicorn(self) -> None:
        """Verify requirements.txt contains uvicorn package."""
        req_file = FASTAPI_DEMO_DIR / "requirements.txt"
        content = req_file.read_text().lower()
        assert "uvicorn" in content, (
            "requirements.txt must contain uvicorn package"
        )

    def test_requirements_contains_httpx(self) -> None:
        """Verify requirements.txt contains httpx package."""
        req_file = FASTAPI_DEMO_DIR / "requirements.txt"
        content = req_file.read_text().lower()
        assert "httpx" in content, (
            "requirements.txt must contain httpx package"
        )

    def test_requirements_contains_pytest(self) -> None:
        """Verify requirements.txt contains pytest package."""
        req_file = FASTAPI_DEMO_DIR / "requirements.txt"
        content = req_file.read_text().lower()
        assert "pytest" in content, (
            "requirements.txt must contain pytest package"
        )

    def test_requirements_has_version_specifiers(self) -> None:
        """Verify requirements.txt has version specifiers for packages."""
        req_file = FASTAPI_DEMO_DIR / "requirements.txt"
        content = req_file.read_text()
        # Check that at least some packages have version constraints
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        has_versions = any(
            ">=" in line or "==" in line or "~=" in line or "<" in line
            for line in lines
        )
        assert has_versions, (
            "requirements.txt should specify version constraints for packages"
        )


class TestPackageImportability:
    """Test suite for verifying packages can be imported."""

    def test_fastapi_demo_is_importable(self) -> None:
        """Verify fastapi_demo package can be imported as a module."""
        import sys
        # Add examples directory to path temporarily
        examples_path = str(PROJECT_ROOT / "examples")
        if examples_path not in sys.path:
            sys.path.insert(0, examples_path)
        try:
            import fastapi_demo
            assert hasattr(fastapi_demo, "__version__")
        finally:
            if examples_path in sys.path:
                sys.path.remove(examples_path)

    def test_routes_subpackage_exists(self) -> None:
        """Verify routes subpackage structure is valid."""
        routes_init = FASTAPI_DEMO_DIR / "routes" / "__init__.py"
        assert routes_init.exists()
        # Verify it's a valid Python file by checking syntax
        content = routes_init.read_text()
        try:
            compile(content, str(routes_init), "exec")
        except SyntaxError as e:
            pytest.fail(f"routes/__init__.py has syntax error: {e}")

    def test_services_subpackage_exists(self) -> None:
        """Verify services subpackage structure is valid."""
        services_init = FASTAPI_DEMO_DIR / "services" / "__init__.py"
        assert services_init.exists()
        # Verify it's a valid Python file by checking syntax
        content = services_init.read_text()
        try:
            compile(content, str(services_init), "exec")
        except SyntaxError as e:
            pytest.fail(f"services/__init__.py has syntax error: {e}")


class TestDirectoryStructureCompleteness:
    """Integration tests for overall directory structure completeness."""

    def test_complete_structure_matches_specification(self) -> None:
        """Verify the complete directory structure matches specification."""
        expected_structure = {
            "examples/fastapi_demo/__init__.py",
            "examples/fastapi_demo/requirements.txt",
            "examples/fastapi_demo/routes/__init__.py",
            "examples/fastapi_demo/services/__init__.py",
        }

        actual_files = set()
        for root, dirs, files in os.walk(PROJECT_ROOT / "examples"):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for file in files:
                full_path = Path(root) / file
                relative_path = full_path.relative_to(PROJECT_ROOT)
                # Convert to forward slashes for consistent comparison
                actual_files.add(str(relative_path).replace("\\", "/"))

        for expected in expected_structure:
            assert expected in actual_files, (
                f"Expected file {expected} not found in directory structure"
            )

    def test_no_unexpected_files_in_fastapi_demo(self) -> None:
        """Verify no unexpected files exist (except __pycache__)."""
        expected_files = {
            "__init__.py",
            "requirements.txt",
        }
        expected_dirs = {"routes", "services", "__pycache__"}

        for item in FASTAPI_DEMO_DIR.iterdir():
            if item.is_file():
                assert item.name in expected_files, (
                    f"Unexpected file found: {item.name}"
                )
            elif item.is_dir():
                assert item.name in expected_dirs, (
                    f"Unexpected directory found: {item.name}"
                )


class TestServicesSubpackage:
    """Test suite for validating services subpackage structure."""

    def test_services_has_user_service(self) -> None:
        """Verify user_service.py exists in services directory."""
        user_service = FASTAPI_DEMO_DIR / "services" / "user_service.py"
        assert user_service.exists(), f"user_service.py not found at {user_service}"

    def test_services_has_order_service(self) -> None:
        """Verify order_service.py exists in services directory."""
        order_service = FASTAPI_DEMO_DIR / "services" / "order_service.py"
        assert order_service.exists(), f"order_service.py not found at {order_service}"

    def test_services_has_tracing_module(self) -> None:
        """Verify tracing.py exists in services directory."""
        tracing = FASTAPI_DEMO_DIR / "services" / "tracing.py"
        assert tracing.exists(), f"tracing.py not found at {tracing}"

    def test_services_imports_work(self) -> None:
        """Verify services can be imported successfully."""
        import sys
        examples_path = str(PROJECT_ROOT / "examples")
        if examples_path not in sys.path:
            sys.path.insert(0, examples_path)
        try:
            from fastapi_demo.services import (
                UserService,
                User,
                SpanData,
                OrderService,
                Order,
                OrderItem,
                OrderStatus,
            )
            # Verify classes are importable
            assert UserService is not None
            assert User is not None
            assert SpanData is not None
            assert OrderService is not None
            assert Order is not None
            assert OrderItem is not None
            assert OrderStatus is not None
        finally:
            if examples_path in sys.path:
                sys.path.remove(examples_path)

"""
Unit tests for project structure and configuration validation.

Tests verify that the project structure is correctly initialized
with all required directories, files, and valid configurations.
"""

import os
import sys
import tomllib
from pathlib import Path

import pytest


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestDirectoryStructure:
    """Tests for verifying directory structure."""

    def test_project_root_exists(self) -> None:
        """Verify the project root directory exists."""
        assert PROJECT_ROOT.exists(), f"Project root {PROJECT_ROOT} does not exist"
        assert PROJECT_ROOT.is_dir(), f"Project root {PROJECT_ROOT} is not a directory"

    def test_tracemaid_package_exists(self) -> None:
        """Verify the tracemaid package directory exists."""
        package_dir = PROJECT_ROOT / "tracemaid"
        assert package_dir.exists(), f"Package directory {package_dir} does not exist"
        assert package_dir.is_dir(), f"Package directory {package_dir} is not a directory"

    def test_core_subpackage_exists(self) -> None:
        """Verify the core subpackage directory exists."""
        core_dir = PROJECT_ROOT / "tracemaid" / "core"
        assert core_dir.exists(), f"Core directory {core_dir} does not exist"
        assert core_dir.is_dir(), f"Core directory {core_dir} is not a directory"

    def test_utils_subpackage_exists(self) -> None:
        """Verify the utils subpackage directory exists."""
        utils_dir = PROJECT_ROOT / "tracemaid" / "utils"
        assert utils_dir.exists(), f"Utils directory {utils_dir} does not exist"
        assert utils_dir.is_dir(), f"Utils directory {utils_dir} is not a directory"

    def test_tests_directory_exists(self) -> None:
        """Verify the tests directory exists."""
        tests_dir = PROJECT_ROOT / "tests"
        assert tests_dir.exists(), f"Tests directory {tests_dir} does not exist"
        assert tests_dir.is_dir(), f"Tests directory {tests_dir} is not a directory"


class TestInitFiles:
    """Tests for verifying __init__.py files."""

    def test_main_package_init_exists(self) -> None:
        """Verify tracemaid/__init__.py exists."""
        init_file = PROJECT_ROOT / "tracemaid" / "__init__.py"
        assert init_file.exists(), f"Init file {init_file} does not exist"
        assert init_file.is_file(), f"Init file {init_file} is not a file"

    def test_core_init_exists(self) -> None:
        """Verify tracemaid/core/__init__.py exists."""
        init_file = PROJECT_ROOT / "tracemaid" / "core" / "__init__.py"
        assert init_file.exists(), f"Init file {init_file} does not exist"
        assert init_file.is_file(), f"Init file {init_file} is not a file"

    def test_utils_init_exists(self) -> None:
        """Verify tracemaid/utils/__init__.py exists."""
        init_file = PROJECT_ROOT / "tracemaid" / "utils" / "__init__.py"
        assert init_file.exists(), f"Init file {init_file} does not exist"
        assert init_file.is_file(), f"Init file {init_file} is not a file"

    def test_tests_init_exists(self) -> None:
        """Verify tests/__init__.py exists."""
        init_file = PROJECT_ROOT / "tests" / "__init__.py"
        assert init_file.exists(), f"Init file {init_file} does not exist"
        assert init_file.is_file(), f"Init file {init_file} is not a file"

    def test_main_init_has_version(self) -> None:
        """Verify tracemaid/__init__.py has __version__ defined."""
        init_file = PROJECT_ROOT / "tracemaid" / "__init__.py"
        content = init_file.read_text()
        assert "__version__" in content, "__version__ not found in main __init__.py"

    def test_main_init_has_author(self) -> None:
        """Verify tracemaid/__init__.py has __author__ defined."""
        init_file = PROJECT_ROOT / "tracemaid" / "__init__.py"
        content = init_file.read_text()
        assert "__author__" in content, "__author__ not found in main __init__.py"

    def test_core_init_has_docstring(self) -> None:
        """Verify tracemaid/core/__init__.py has a module docstring."""
        init_file = PROJECT_ROOT / "tracemaid" / "core" / "__init__.py"
        content = init_file.read_text()
        assert content.strip().startswith('"""'), "core/__init__.py missing docstring"

    def test_utils_init_has_docstring(self) -> None:
        """Verify tracemaid/utils/__init__.py has a module docstring."""
        init_file = PROJECT_ROOT / "tracemaid" / "utils" / "__init__.py"
        content = init_file.read_text()
        assert content.strip().startswith('"""'), "utils/__init__.py missing docstring"


class TestPyprojectToml:
    """Tests for verifying pyproject.toml configuration."""

    def test_pyproject_exists(self) -> None:
        """Verify pyproject.toml exists."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_file.exists(), f"pyproject.toml not found at {pyproject_file}"
        assert pyproject_file.is_file(), f"{pyproject_file} is not a file"

    def test_pyproject_is_valid_toml(self) -> None:
        """Verify pyproject.toml is valid TOML."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert isinstance(config, dict), "pyproject.toml should parse to a dict"

    def test_pyproject_has_build_system(self) -> None:
        """Verify pyproject.toml has build-system section."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "build-system" in config, "build-system section missing"
        assert "requires" in config["build-system"], "build-system.requires missing"
        assert "build-backend" in config["build-system"], "build-system.build-backend missing"

    def test_pyproject_has_project_section(self) -> None:
        """Verify pyproject.toml has project section."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "project" in config, "project section missing"

    def test_pyproject_has_name(self) -> None:
        """Verify pyproject.toml has project name."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert config["project"].get("name") == "tracemaid", "project.name should be 'tracemaid'"

    def test_pyproject_has_version(self) -> None:
        """Verify pyproject.toml has project version."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "version" in config["project"], "project.version missing"
        assert config["project"]["version"] == "0.1.0", "project.version should be '0.1.0'"

    def test_pyproject_has_description(self) -> None:
        """Verify pyproject.toml has project description."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "description" in config["project"], "project.description missing"
        assert len(config["project"]["description"]) > 0, "project.description should not be empty"

    def test_pyproject_has_python_requires(self) -> None:
        """Verify pyproject.toml specifies Python version requirement."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "requires-python" in config["project"], "requires-python missing"
        assert ">=3.9" in config["project"]["requires-python"], "Should require Python 3.9+"

    def test_pyproject_has_dependencies(self) -> None:
        """Verify pyproject.toml has dependencies listed."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "dependencies" in config["project"], "project.dependencies missing"
        deps = config["project"]["dependencies"]
        assert isinstance(deps, list), "dependencies should be a list"
        assert len(deps) > 0, "dependencies should not be empty"

    def test_pyproject_has_optional_dependencies(self) -> None:
        """Verify pyproject.toml has optional dependencies for dev."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "optional-dependencies" in config["project"], "optional-dependencies missing"
        assert "dev" in config["project"]["optional-dependencies"], "dev dependencies missing"

    def test_pyproject_has_scripts(self) -> None:
        """Verify pyproject.toml has CLI entry point."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "scripts" in config["project"], "project.scripts missing"
        assert "tracemaid" in config["project"]["scripts"], "tracemaid CLI entry point missing"

    def test_pyproject_has_urls(self) -> None:
        """Verify pyproject.toml has project URLs."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "urls" in config["project"], "project.urls missing"

    def test_pyproject_has_pytest_config(self) -> None:
        """Verify pyproject.toml has pytest configuration."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "tool" in config, "tool section missing"
        assert "pytest" in config["tool"], "tool.pytest missing"


class TestPackageImports:
    """Tests for verifying package imports work correctly."""

    def test_import_tracemaid(self) -> None:
        """Verify tracemaid can be imported."""
        import tracemaid
        assert tracemaid is not None

    def test_tracemaid_has_version(self) -> None:
        """Verify tracemaid.__version__ is accessible."""
        import tracemaid
        assert hasattr(tracemaid, "__version__")
        assert tracemaid.__version__ == "0.1.0"

    def test_tracemaid_has_author(self) -> None:
        """Verify tracemaid.__author__ is accessible."""
        import tracemaid
        assert hasattr(tracemaid, "__author__")
        assert tracemaid.__author__ == "KR"

    def test_import_tracemaid_core(self) -> None:
        """Verify tracemaid.core can be imported."""
        import tracemaid.core
        assert tracemaid.core is not None

    def test_import_tracemaid_utils(self) -> None:
        """Verify tracemaid.utils can be imported."""
        import tracemaid.utils
        assert tracemaid.utils is not None


class TestVersionConsistency:
    """Tests for verifying version consistency across files."""

    def test_version_matches_pyproject(self) -> None:
        """Verify __version__ matches pyproject.toml version."""
        import tracemaid

        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)

        assert tracemaid.__version__ == config["project"]["version"], (
            f"Version mismatch: __init__.py has {tracemaid.__version__}, "
            f"pyproject.toml has {config['project']['version']}"
        )


class TestReadmeDocumentation:
    """Tests for verifying README.md documentation."""

    def test_readme_exists(self) -> None:
        """Verify README.md exists."""
        readme_file = PROJECT_ROOT / "README.md"
        assert readme_file.exists(), f"README.md not found at {readme_file}"
        assert readme_file.is_file(), f"{readme_file} is not a file"

    def test_readme_not_empty(self) -> None:
        """Verify README.md has content."""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text()
        assert len(content) > 100, "README.md should have substantial content"

    def test_readme_has_project_title(self) -> None:
        """Verify README.md has project title."""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text()
        assert "# tracemaid" in content, "README.md should have project title"

    def test_readme_has_installation_section(self) -> None:
        """Verify README.md has installation instructions."""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text()
        assert "## Installation" in content, "README.md should have installation section"

    def test_readme_has_usage_examples(self) -> None:
        """Verify README.md has usage examples."""
        readme_file = PROJECT_ROOT / "README.md"
        content = readme_file.read_text()
        assert "```python" in content, "README.md should have Python code examples"


class TestDependencyRequirements:
    """Tests for verifying dependency requirements are valid."""

    def test_has_numpy_dependency(self) -> None:
        """Verify numpy is in dependencies."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        deps = config["project"]["dependencies"]
        assert any("numpy" in dep for dep in deps), "numpy should be in dependencies"

    def test_has_scipy_dependency(self) -> None:
        """Verify scipy is in dependencies."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        deps = config["project"]["dependencies"]
        assert any("scipy" in dep for dep in deps), "scipy should be in dependencies"

    def test_has_sklearn_dependency(self) -> None:
        """Verify scikit-learn is in dependencies."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        deps = config["project"]["dependencies"]
        assert any("scikit-learn" in dep for dep in deps), "scikit-learn should be in dependencies"

    def test_has_pytest_dev_dependency(self) -> None:
        """Verify pytest is in dev dependencies."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        dev_deps = config["project"]["optional-dependencies"]["dev"]
        assert any("pytest" in dep for dep in dev_deps), "pytest should be in dev dependencies"


class TestPackageMetadata:
    """Tests for verifying package metadata is complete."""

    def test_has_email_attribute(self) -> None:
        """Verify tracemaid.__email__ is accessible."""
        import tracemaid
        assert hasattr(tracemaid, "__email__")
        assert "@" in tracemaid.__email__, "__email__ should be a valid email"

    def test_pyproject_has_keywords(self) -> None:
        """Verify pyproject.toml has keywords."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "keywords" in config["project"], "project.keywords missing"
        keywords = config["project"]["keywords"]
        assert len(keywords) >= 3, "Should have at least 3 keywords"

    def test_pyproject_has_classifiers(self) -> None:
        """Verify pyproject.toml has classifiers."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "classifiers" in config["project"], "project.classifiers missing"
        classifiers = config["project"]["classifiers"]
        assert len(classifiers) >= 5, "Should have at least 5 classifiers"

    def test_pyproject_has_license(self) -> None:
        """Verify pyproject.toml has license."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "license" in config["project"], "project.license missing"

    def test_pyproject_has_authors(self) -> None:
        """Verify pyproject.toml has authors."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_file, "rb") as f:
            config = tomllib.load(f)
        assert "authors" in config["project"], "project.authors missing"
        authors = config["project"]["authors"]
        assert len(authors) >= 1, "Should have at least 1 author"
        assert "name" in authors[0], "Author should have name"
        assert "email" in authors[0], "Author should have email"

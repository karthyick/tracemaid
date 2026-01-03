"""Unit tests for tracemaid.integrations instrumentation functions.

This module tests all auto-instrumentation functions including:
- instrument_requests
- instrument_httpx
- instrument_sqlalchemy
- instrument_redis
- instrument_logging
- instrument_all
"""

from __future__ import annotations

import pytest


class TestInstrumentRequests:
    """Tests for instrument_requests function."""

    def test_instrument_requests_callable(self) -> None:
        """Test instrument_requests is callable."""
        from tracemaid.integrations import instrument_requests
        assert callable(instrument_requests)

    def test_instrument_requests_no_error(self) -> None:
        """Test instrument_requests runs without error."""
        from tracemaid.integrations import instrument_requests
        # Should not raise - either succeeds or logs warning
        instrument_requests()

    def test_instrument_requests_idempotent(self) -> None:
        """Test calling instrument_requests multiple times doesn't error."""
        from tracemaid.integrations import instrument_requests
        instrument_requests()
        instrument_requests()  # Should not raise


class TestInstrumentHttpx:
    """Tests for instrument_httpx function."""

    def test_instrument_httpx_callable(self) -> None:
        """Test instrument_httpx is callable."""
        from tracemaid.integrations import instrument_httpx
        assert callable(instrument_httpx)

    def test_instrument_httpx_no_error(self) -> None:
        """Test instrument_httpx runs without error."""
        from tracemaid.integrations import instrument_httpx
        instrument_httpx()

    def test_instrument_httpx_idempotent(self) -> None:
        """Test calling instrument_httpx multiple times doesn't error."""
        from tracemaid.integrations import instrument_httpx
        instrument_httpx()
        instrument_httpx()


class TestInstrumentSqlalchemy:
    """Tests for instrument_sqlalchemy function."""

    def test_instrument_sqlalchemy_callable(self) -> None:
        """Test instrument_sqlalchemy is callable."""
        from tracemaid.integrations import instrument_sqlalchemy
        assert callable(instrument_sqlalchemy)

    def test_instrument_sqlalchemy_no_error(self) -> None:
        """Test instrument_sqlalchemy runs without error."""
        from tracemaid.integrations import instrument_sqlalchemy
        instrument_sqlalchemy()

    def test_instrument_sqlalchemy_accepts_engine(self) -> None:
        """Test instrument_sqlalchemy accepts engine parameter."""
        from tracemaid.integrations import instrument_sqlalchemy
        from unittest.mock import MagicMock
        mock_engine = MagicMock()
        # Should not raise
        instrument_sqlalchemy(engine=mock_engine)


class TestInstrumentRedis:
    """Tests for instrument_redis function."""

    def test_instrument_redis_callable(self) -> None:
        """Test instrument_redis is callable."""
        from tracemaid.integrations import instrument_redis
        assert callable(instrument_redis)

    def test_instrument_redis_no_error(self) -> None:
        """Test instrument_redis runs without error."""
        from tracemaid.integrations import instrument_redis
        instrument_redis()

    def test_instrument_redis_accepts_client(self) -> None:
        """Test instrument_redis accepts client parameter."""
        from tracemaid.integrations import instrument_redis
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        # Should not raise
        instrument_redis(client=mock_client)


class TestInstrumentLogging:
    """Tests for instrument_logging function."""

    def test_instrument_logging_callable(self) -> None:
        """Test instrument_logging is callable."""
        from tracemaid.integrations import instrument_logging
        assert callable(instrument_logging)

    def test_instrument_logging_no_error(self) -> None:
        """Test instrument_logging runs without error."""
        from tracemaid.integrations import instrument_logging
        instrument_logging()


class TestInstrumentAll:
    """Tests for instrument_all convenience function."""

    def test_instrument_all_callable(self) -> None:
        """Test instrument_all is callable."""
        from tracemaid.integrations import instrument_all
        assert callable(instrument_all)

    def test_instrument_all_returns_dict(self) -> None:
        """Test instrument_all returns a dictionary of results."""
        from tracemaid.integrations import instrument_all
        result = instrument_all()

        assert isinstance(result, dict)
        assert "requests" in result
        assert "httpx" in result
        assert "sqlalchemy" in result
        assert "redis" in result
        assert "logging" in result

    def test_instrument_all_values_are_bool(self) -> None:
        """Test instrument_all returns boolean values."""
        from tracemaid.integrations import instrument_all
        result = instrument_all()

        for key, value in result.items():
            assert isinstance(value, bool), f"{key} should be bool, got {type(value)}"

    def test_instrument_all_idempotent(self) -> None:
        """Test calling instrument_all multiple times doesn't error."""
        from tracemaid.integrations import instrument_all
        result1 = instrument_all()
        result2 = instrument_all()

        # Results should have same keys
        assert result1.keys() == result2.keys()

    def test_instrument_all_accepts_sqlalchemy_engine(self) -> None:
        """Test instrument_all accepts sqlalchemy_engine parameter."""
        from tracemaid.integrations import instrument_all
        from unittest.mock import MagicMock
        mock_engine = MagicMock()
        # Should not raise
        result = instrument_all(sqlalchemy_engine=mock_engine)
        assert isinstance(result, dict)

    def test_instrument_all_accepts_redis_client(self) -> None:
        """Test instrument_all accepts redis_client parameter."""
        from tracemaid.integrations import instrument_all
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        # Should not raise
        result = instrument_all(redis_client=mock_client)
        assert isinstance(result, dict)

    def test_instrument_all_accepts_both_params(self) -> None:
        """Test instrument_all accepts both parameters."""
        from tracemaid.integrations import instrument_all
        from unittest.mock import MagicMock
        mock_engine = MagicMock()
        mock_client = MagicMock()
        # Should not raise
        result = instrument_all(sqlalchemy_engine=mock_engine, redis_client=mock_client)
        assert isinstance(result, dict)


class TestIntegrationsExports:
    """Test that all functions are properly exported from the package."""

    def test_all_functions_exported(self) -> None:
        """Test all instrumentation functions are exported."""
        from tracemaid import integrations

        expected_exports = [
            "setup_fastapi_tracing",
            "setup_tracing",
            "get_tracer",
            "shutdown_tracing",
            "instrument_requests",
            "instrument_logging",
            "instrument_httpx",
            "instrument_sqlalchemy",
            "instrument_redis",
            "instrument_all",
        ]

        for name in expected_exports:
            assert hasattr(integrations, name), f"{name} not exported"

    def test_imports_from_package(self) -> None:
        """Test functions can be imported directly from package."""
        from tracemaid.integrations import (
            instrument_requests,
            instrument_httpx,
            instrument_sqlalchemy,
            instrument_redis,
            instrument_logging,
            instrument_all,
        )

        # All should be callable
        assert callable(instrument_requests)
        assert callable(instrument_httpx)
        assert callable(instrument_sqlalchemy)
        assert callable(instrument_redis)
        assert callable(instrument_logging)
        assert callable(instrument_all)

    def test_instrument_all_in_integrations_all(self) -> None:
        """Test instrument_all is in the __all__ list."""
        from tracemaid.integrations import __all__
        assert "instrument_all" in __all__

    def test_all_new_functions_in_all(self) -> None:
        """Test all new instrumentation functions are in __all__."""
        from tracemaid.integrations import __all__

        new_functions = [
            "instrument_requests",
            "instrument_httpx",
            "instrument_sqlalchemy",
            "instrument_redis",
            "instrument_logging",
            "instrument_all",
        ]

        for func_name in new_functions:
            assert func_name in __all__, f"{func_name} not in __all__"


class TestInstrumentAllResultValues:
    """Test that instrument_all correctly reports which libraries are available."""

    def test_requests_instrumentation_available(self) -> None:
        """Test requests instrumentation is available (installed in this env)."""
        from tracemaid.integrations import instrument_all
        result = instrument_all()
        # We installed opentelemetry-instrumentation-requests
        assert result["requests"] is True

    def test_httpx_instrumentation_available(self) -> None:
        """Test httpx instrumentation is available (installed in this env)."""
        from tracemaid.integrations import instrument_all
        result = instrument_all()
        # We installed opentelemetry-instrumentation-httpx
        assert result["httpx"] is True

    def test_sqlalchemy_instrumentation_available(self) -> None:
        """Test sqlalchemy instrumentation is available (installed in this env)."""
        from tracemaid.integrations import instrument_all
        result = instrument_all()
        # We installed opentelemetry-instrumentation-sqlalchemy
        assert result["sqlalchemy"] is True


class TestFunctionSignatures:
    """Test function signatures accept expected parameters."""

    def test_instrument_sqlalchemy_signature(self) -> None:
        """Test instrument_sqlalchemy accepts engine keyword arg."""
        import inspect
        from tracemaid.integrations import instrument_sqlalchemy

        sig = inspect.signature(instrument_sqlalchemy)
        params = list(sig.parameters.keys())
        assert "engine" in params

    def test_instrument_redis_signature(self) -> None:
        """Test instrument_redis accepts client keyword arg."""
        import inspect
        from tracemaid.integrations import instrument_redis

        sig = inspect.signature(instrument_redis)
        params = list(sig.parameters.keys())
        assert "client" in params

    def test_instrument_all_signature(self) -> None:
        """Test instrument_all accepts expected keyword args."""
        import inspect
        from tracemaid.integrations import instrument_all

        sig = inspect.signature(instrument_all)
        params = list(sig.parameters.keys())
        assert "sqlalchemy_engine" in params
        assert "redis_client" in params

    def test_instrument_all_return_type(self) -> None:
        """Test instrument_all has correct return type annotation."""
        import inspect
        from tracemaid.integrations import instrument_all

        sig = inspect.signature(instrument_all)
        # Check return annotation contains dict and bool
        return_annotation = str(sig.return_annotation)
        assert "dict" in return_annotation
        assert "str" in return_annotation
        assert "bool" in return_annotation

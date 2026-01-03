"""Unit tests for TracemaidExporter.

This module provides comprehensive test coverage for the TracemaidExporter,
which converts OpenTelemetry spans to Mermaid diagrams.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExportResult
    from opentelemetry.trace import SpanContext, TraceFlags, StatusCode, Status
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from tracemaid.exporters import TracemaidExporter


# Skip all tests if OpenTelemetry is not available
pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE,
    reason="OpenTelemetry SDK not installed"
)


class MockSpanContext:
    """Mock SpanContext for testing."""

    def __init__(self, trace_id: int, span_id: int):
        self.trace_id = trace_id
        self.span_id = span_id
        self.trace_flags = TraceFlags(0x01)


class MockParent:
    """Mock parent span for testing."""

    def __init__(self, span_id: int):
        self.span_id = span_id


class MockResource:
    """Mock Resource for testing."""

    def __init__(self, service_name: str = "test-service"):
        self.attributes = {"service.name": service_name}


class MockStatus:
    """Mock Status for testing."""

    def __init__(self, status_code: StatusCode = StatusCode.OK):
        self.status_code = status_code


class MockReadableSpan:
    """Mock ReadableSpan for testing."""

    def __init__(
        self,
        name: str,
        trace_id: int,
        span_id: int,
        parent_span_id: Optional[int] = None,
        service_name: str = "test-service",
        start_time: int = 1000000000,
        end_time: int = 2000000000,
        status_code: StatusCode = StatusCode.OK,
        attributes: Optional[dict] = None,
    ):
        self.name = name
        self.context = MockSpanContext(trace_id, span_id)
        self.parent = MockParent(parent_span_id) if parent_span_id else None
        self.resource = MockResource(service_name)
        self.start_time = start_time
        self.end_time = end_time
        self.status = MockStatus(status_code)
        self.attributes = attributes or {}


class TestTracemaidExporterInit:
    """Test TracemaidExporter initialization."""

    def test_default_initialization(self) -> None:
        """Test exporter initializes with default values."""
        exporter = TracemaidExporter()

        assert exporter.output_dir is None
        assert exporter.max_spans == 15
        assert exporter.enable_styling is True
        assert exporter.include_metadata is False
        assert exporter.console_output is True
        assert exporter.flush_interval_seconds == 2.0

    def test_custom_initialization(self) -> None:
        """Test exporter initializes with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TracemaidExporter(
                output_dir=tmpdir,
                max_spans=20,
                enable_styling=False,
                include_metadata=True,
                console_output=False,
                flush_interval_seconds=5.0,
            )

            assert exporter.output_dir == Path(tmpdir)
            assert exporter.max_spans == 20
            assert exporter.enable_styling is False
            assert exporter.include_metadata is True
            assert exporter.console_output is False
            assert exporter.flush_interval_seconds == 5.0

    def test_creates_output_directory(self) -> None:
        """Test exporter creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "traces" / "nested"
            exporter = TracemaidExporter(output_dir=str(output_path))

            assert output_path.exists()
            assert output_path.is_dir()

    def test_callback_initialization(self) -> None:
        """Test exporter initializes with callback function."""
        callback = MagicMock()
        exporter = TracemaidExporter(on_diagram_generated=callback)

        assert exporter.on_diagram_generated == callback


class TestTracemaidExporterExport:
    """Test TracemaidExporter export method."""

    def test_export_empty_spans_returns_success(self) -> None:
        """Test exporting empty span list returns success."""
        exporter = TracemaidExporter(console_output=False)
        result = exporter.export([])

        assert result == SpanExportResult.SUCCESS

    def test_export_collects_spans_by_trace_id(self) -> None:
        """Test export groups spans by trace_id."""
        exporter = TracemaidExporter(console_output=False, flush_interval_seconds=10.0)

        trace_id = 0x12345678901234567890123456789012
        # Use spans with parent_span_id so they're not treated as root spans
        # (root spans trigger immediate processing)
        spans = [
            MockReadableSpan("span1", trace_id, 0x1234567890123456, parent_span_id=0xabcdef),
            MockReadableSpan("span2", trace_id, 0x1234567890123457, parent_span_id=0xabcdef),
        ]

        result = exporter.export(spans)

        assert result == SpanExportResult.SUCCESS
        trace_id_hex = format(trace_id, "032x")
        assert trace_id_hex in exporter._trace_spans
        assert len(exporter._trace_spans[trace_id_hex]) == 2

    def test_export_separates_different_traces(self) -> None:
        """Test export separates spans from different traces."""
        exporter = TracemaidExporter(console_output=False, flush_interval_seconds=10.0)

        trace_id_1 = 0x11111111111111111111111111111111
        trace_id_2 = 0x22222222222222222222222222222222

        # Use spans with parent_span_id so they're not treated as root spans
        # (root spans trigger immediate processing)
        spans = [
            MockReadableSpan("span1", trace_id_1, 0x1111111111111111, parent_span_id=0xabcdef),
            MockReadableSpan("span2", trace_id_2, 0x2222222222222222, parent_span_id=0xfedcba),
        ]

        result = exporter.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert len(exporter._trace_spans) == 2

    def test_export_processes_trace_immediately_when_root_span_arrives(self) -> None:
        """Test that receiving a root span (no parent) triggers immediate trace processing."""
        exporter = TracemaidExporter(console_output=False, flush_interval_seconds=100.0)

        trace_id = 0x12345678901234567890123456789012
        root_span_id = 0xabcdef0123456789

        # First, export some child spans (with parent)
        child_spans = [
            MockReadableSpan("child1", trace_id, 0x1111111111111111, parent_span_id=root_span_id),
            MockReadableSpan("child2", trace_id, 0x2222222222222222, parent_span_id=root_span_id),
        ]
        exporter.export(child_spans)

        # Child spans should be in buffer (not processed yet)
        trace_id_hex = format(trace_id, "032x")
        assert trace_id_hex in exporter._trace_spans
        assert len(exporter._trace_spans[trace_id_hex]) == 2

        # Now export the root span (no parent)
        root_span = [MockReadableSpan("root", trace_id, root_span_id, parent_span_id=None)]

        with patch.object(exporter, '_process_trace') as mock_process:
            exporter.export(root_span)
            # Root span should trigger immediate processing
            mock_process.assert_called_once_with(trace_id_hex)


class TestTracemaidExporterSpanConversion:
    """Test TracemaidExporter span conversion."""

    def test_readable_span_to_dict_basic(self) -> None:
        """Test basic span to dictionary conversion."""
        exporter = TracemaidExporter(console_output=False)

        span = MockReadableSpan(
            name="test_operation",
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            service_name="my-service",
        )

        span_dict = exporter._readable_span_to_dict(span)

        assert span_dict["name"] == "test_operation"
        assert span_dict["traceId"] == "12345678901234567890123456789012"
        assert span_dict["spanId"] == "1234567890123456"
        assert span_dict["_serviceName"] == "my-service"

    def test_readable_span_to_dict_with_parent(self) -> None:
        """Test span to dictionary conversion with parent span."""
        exporter = TracemaidExporter(console_output=False)

        span = MockReadableSpan(
            name="child_operation",
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            parent_span_id=0xabcdef0123456789,
        )

        span_dict = exporter._readable_span_to_dict(span)

        assert span_dict["parentSpanId"] == "abcdef0123456789"

    def test_readable_span_to_dict_without_parent(self) -> None:
        """Test span to dictionary conversion without parent span."""
        exporter = TracemaidExporter(console_output=False)

        span = MockReadableSpan(
            name="root_operation",
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            parent_span_id=None,
        )

        span_dict = exporter._readable_span_to_dict(span)

        assert span_dict["parentSpanId"] == ""

    def test_readable_span_to_dict_status_ok(self) -> None:
        """Test span with OK status."""
        exporter = TracemaidExporter(console_output=False)

        span = MockReadableSpan(
            name="success_operation",
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            status_code=StatusCode.OK,
        )

        span_dict = exporter._readable_span_to_dict(span)

        assert span_dict["status"]["code"] == 1  # OK

    def test_readable_span_to_dict_status_error(self) -> None:
        """Test span with ERROR status."""
        exporter = TracemaidExporter(console_output=False)

        span = MockReadableSpan(
            name="error_operation",
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            status_code=StatusCode.ERROR,
        )

        span_dict = exporter._readable_span_to_dict(span)

        assert span_dict["status"]["code"] == 2  # ERROR

    def test_readable_span_to_dict_with_attributes(self) -> None:
        """Test span to dictionary conversion with attributes."""
        exporter = TracemaidExporter(console_output=False)

        span = MockReadableSpan(
            name="operation_with_attrs",
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
            attributes={"key1": "value1", "key2": 123},
        )

        span_dict = exporter._readable_span_to_dict(span)

        assert len(span_dict["attributes"]) == 2


class TestTracemaidExporterForceFlush:
    """Test TracemaidExporter force_flush method."""

    def test_force_flush_empty(self) -> None:
        """Test force flush with no pending traces."""
        exporter = TracemaidExporter(console_output=False)
        result = exporter.force_flush()

        assert result is True

    def test_force_flush_processes_pending_traces(self) -> None:
        """Test force flush calls _process_trace for pending traces."""
        exporter = TracemaidExporter(console_output=False, flush_interval_seconds=100.0)

        trace_id = 0x12345678901234567890123456789012
        # Use span with parent_span_id so it's not treated as a root span
        # (root spans trigger immediate processing)
        spans = [MockReadableSpan("span1", trace_id, 0x1234567890123456, parent_span_id=0xabcdef)]

        exporter.export(spans)
        assert len(exporter._trace_spans) > 0

        # Track if _process_trace is called
        with patch.object(exporter, '_process_trace') as mock_process:
            exporter.force_flush()
            # force_flush should call _process_trace for each trace
            mock_process.assert_called()


class TestTracemaidExporterShutdown:
    """Test TracemaidExporter shutdown method."""

    def test_shutdown_calls_force_flush(self) -> None:
        """Test shutdown calls force_flush."""
        exporter = TracemaidExporter(console_output=False)

        with patch.object(exporter, 'force_flush', return_value=True) as mock_flush:
            exporter.shutdown()
            mock_flush.assert_called_once()


class TestTracemaidExporterCallback:
    """Test TracemaidExporter callback functionality."""

    def test_callback_not_called_when_not_set(self) -> None:
        """Test no error when callback is not set."""
        exporter = TracemaidExporter(
            console_output=False,
            on_diagram_generated=None
        )

        # Should not raise
        exporter._output_diagram("trace123", "mermaid code", 5, 3)

    def test_callback_error_is_caught(self) -> None:
        """Test callback errors are caught and logged."""
        def failing_callback(trace_id, diagram):
            raise RuntimeError("Callback failed")

        exporter = TracemaidExporter(
            console_output=False,
            on_diagram_generated=failing_callback
        )

        # Should not raise
        exporter._output_diagram("trace123", "mermaid code", 5, 3)


class TestTracemaidExporterFileOutput:
    """Test TracemaidExporter file output."""

    def test_file_output_creates_file(self) -> None:
        """Test that file output creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TracemaidExporter(
                output_dir=tmpdir,
                console_output=False
            )

            exporter._output_diagram("abc12345", "graph TD\n  A --> B", 10, 5)

            files = list(Path(tmpdir).glob("*.mmd"))
            assert len(files) == 1
            assert "abc12345" in files[0].name

    def test_file_output_contains_diagram(self) -> None:
        """Test that file output contains the diagram."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TracemaidExporter(
                output_dir=tmpdir,
                console_output=False
            )

            diagram = "graph TD\n  A --> B"
            exporter._output_diagram("abc12345", diagram, 10, 5)

            files = list(Path(tmpdir).glob("*.mmd"))
            content = files[0].read_text()
            assert "graph TD" in content
            assert "A --> B" in content

    def test_file_output_contains_metadata(self) -> None:
        """Test that file output contains metadata comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TracemaidExporter(
                output_dir=tmpdir,
                console_output=False
            )

            exporter._output_diagram("abc12345", "graph TD\n  A --> B", 10, 5)

            files = list(Path(tmpdir).glob("*.mmd"))
            content = files[0].read_text()
            assert "Trace ID:" in content
            assert "abc12345" in content
            assert "5/10" in content  # selected/total


class TestTracemaidExporterIntegration:
    """Integration tests for TracemaidExporter."""

    def test_full_export_workflow(self) -> None:
        """Test complete export workflow."""
        callback_results = []

        def capture_callback(trace_id, diagram):
            callback_results.append((trace_id, diagram))

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = TracemaidExporter(
                output_dir=tmpdir,
                console_output=False,
                on_diagram_generated=capture_callback,
                flush_interval_seconds=0.0,  # Immediate flush
            )

            trace_id = 0x12345678901234567890123456789012
            spans = [
                MockReadableSpan(
                    name="root_operation",
                    trace_id=trace_id,
                    span_id=0x1234567890123456,
                ),
                MockReadableSpan(
                    name="child_operation",
                    trace_id=trace_id,
                    span_id=0x1234567890123457,
                    parent_span_id=0x1234567890123456,
                ),
            ]

            # Export twice to trigger processing
            exporter.export(spans)
            exporter.export([])  # Trigger processing of complete traces

            # Force flush to ensure processing
            exporter.force_flush()

    def test_thread_safety(self) -> None:
        """Test exporter is thread-safe."""
        import threading

        exporter = TracemaidExporter(console_output=False, flush_interval_seconds=100.0)

        def export_spans():
            for _ in range(10):
                trace_id = 0x12345678901234567890123456789012
                span = MockReadableSpan("span", trace_id, 0x1234567890123456)
                exporter.export([span])

        threads = [threading.Thread(target=export_spans) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not raise any errors
        exporter.shutdown()

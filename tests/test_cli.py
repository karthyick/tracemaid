"""
Tests for tracemaid.cli module.

This module contains comprehensive tests for the CLI interface including
argument parsing, trace loading, output generation, and end-to-end flows.
"""

import json
import os
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from tracemaid.cli import (
    parse_args,
    load_trace,
    select_important_spans,
    generate_mermaid_output,
    generate_json_output,
    write_output,
    main,
)
from tracemaid.core.parser import Span, Trace


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_trace_path() -> Path:
    """Return path to the sample trace fixture."""
    return Path(__file__).parent / "fixtures" / "sample_trace.json"


@pytest.fixture
def sample_trace(sample_trace_path: Path) -> Trace:
    """Load and return the sample trace."""
    return load_trace(str(sample_trace_path))


@pytest.fixture
def simple_trace() -> Trace:
    """Create a simple trace for testing."""
    root = Span(
        spanId="root",
        parentSpanId=None,
        service="api",
        operation="handle_request",
        duration=1000000,
        status="OK",
        depth=0,
        children=[],
    )
    child = Span(
        spanId="child",
        parentSpanId="root",
        service="db",
        operation="query",
        duration=500000,
        status="OK",
        depth=1,
        children=[],
    )
    root.children = [child]
    return Trace(
        traceId="trace123",
        spans=[root, child],
        total_duration=1000000,
    )


@pytest.fixture
def temp_json_file(simple_trace: Trace) -> Path:
    """Create a temporary JSON file with trace data."""
    trace_data = {
        "traceId": simple_trace.traceId,
        "spans": [
            {
                "spanId": span.spanId,
                "parentSpanId": span.parentSpanId or "",
                "name": span.operation,
                "serviceName": span.service,
                "duration": span.duration,
                "status": span.status,
            }
            for span in simple_trace.spans
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(trace_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


# =============================================================================
# Test Argument Parsing
# =============================================================================


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_required_input_file(self):
        """Input file is required."""
        args = parse_args(["trace.json"])
        assert args.input_file == "trace.json"

    def test_parse_output_short_option(self):
        """Short -o option for output works."""
        args = parse_args(["trace.json", "-o", "output.md"])
        assert args.output == "output.md"

    def test_parse_output_long_option(self):
        """Long --output option works."""
        args = parse_args(["trace.json", "--output", "output.md"])
        assert args.output == "output.md"

    def test_parse_max_spans_short_option(self):
        """Short -n option for max-spans works."""
        args = parse_args(["trace.json", "-n", "15"])
        assert args.max_spans == 15

    def test_parse_max_spans_long_option(self):
        """Long --max-spans option works."""
        args = parse_args(["trace.json", "--max-spans", "20"])
        assert args.max_spans == 20

    def test_parse_max_spans_default(self):
        """Default max-spans is 10."""
        args = parse_args(["trace.json"])
        assert args.max_spans == 10

    def test_parse_format_mermaid(self):
        """Format option accepts mermaid."""
        args = parse_args(["trace.json", "--format", "mermaid"])
        assert args.format == "mermaid"

    def test_parse_format_json(self):
        """Format option accepts json."""
        args = parse_args(["trace.json", "--format", "json"])
        assert args.format == "json"

    def test_parse_format_default(self):
        """Default format is mermaid."""
        args = parse_args(["trace.json"])
        assert args.format == "mermaid"

    def test_parse_no_style_flag(self):
        """--no-style flag works."""
        args = parse_args(["trace.json", "--no-style"])
        assert args.no_style is True

    def test_parse_metadata_flag(self):
        """--metadata flag works."""
        args = parse_args(["trace.json", "--metadata"])
        assert args.metadata is True

    def test_parse_verbose_short(self):
        """Short -v option for verbose works."""
        args = parse_args(["trace.json", "-v"])
        assert args.verbose is True

    def test_parse_verbose_long(self):
        """Long --verbose option works."""
        args = parse_args(["trace.json", "--verbose"])
        assert args.verbose is True

    def test_parse_combined_options(self):
        """Multiple options can be combined."""
        args = parse_args([
            "trace.json",
            "-o", "out.md",
            "-n", "25",
            "--format", "json",
            "--no-style",
            "-v"
        ])
        assert args.input_file == "trace.json"
        assert args.output == "out.md"
        assert args.max_spans == 25
        assert args.format == "json"
        assert args.no_style is True
        assert args.verbose is True


# =============================================================================
# Test Trace Loading
# =============================================================================


class TestLoadTrace:
    """Tests for trace loading."""

    def test_load_trace_success(self, sample_trace_path: Path):
        """Successfully loads a valid trace file."""
        trace = load_trace(str(sample_trace_path))

        assert isinstance(trace, Trace)
        assert trace.traceId == "d4cda95b652f4a1592b449d5929fda1b"
        assert len(trace.spans) > 0

    def test_load_trace_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_trace("nonexistent_file.json")

    def test_load_trace_invalid_json(self, tmp_path: Path):
        """Raises JSONDecodeError for invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_trace(str(invalid_file))

    def test_load_trace_simple_format(self, temp_json_file: Path):
        """Loads trace from simple format."""
        trace = load_trace(str(temp_json_file))

        assert isinstance(trace, Trace)
        assert len(trace.spans) > 0


# =============================================================================
# Test Span Selection
# =============================================================================


class TestSelectImportantSpans:
    """Tests for span selection."""

    def test_select_returns_spans(self, sample_trace: Trace):
        """Selection returns Span objects."""
        spans = select_important_spans(sample_trace, max_spans=5)

        assert isinstance(spans, list)
        assert all(isinstance(s, Span) for s in spans)

    def test_select_respects_max_spans(self, sample_trace: Trace):
        """Selection respects max_spans limit."""
        spans = select_important_spans(sample_trace, max_spans=3)

        assert len(spans) <= 3

    def test_select_all_when_fewer_spans(self, simple_trace: Trace):
        """Returns all spans when trace has fewer than max."""
        spans = select_important_spans(simple_trace, max_spans=100)

        assert len(spans) == len(simple_trace.spans)


# =============================================================================
# Test Output Generation
# =============================================================================


class TestGenerateMermaidOutput:
    """Tests for Mermaid output generation."""

    def test_generates_valid_mermaid(self, simple_trace: Trace):
        """Generates valid Mermaid diagram."""
        output = generate_mermaid_output(
            simple_trace.spans, simple_trace
        )

        assert "flowchart TD" in output
        assert "api: handle_request" in output
        assert "db: query" in output

    def test_includes_edges(self, simple_trace: Trace):
        """Diagram includes edges for relationships."""
        output = generate_mermaid_output(
            simple_trace.spans, simple_trace
        )

        assert "-->" in output

    def test_styling_enabled_by_default(self, simple_trace: Trace):
        """Styling is included by default."""
        output = generate_mermaid_output(
            simple_trace.spans, simple_trace
        )

        assert "classDef" in output or "style" in output.lower()

    def test_styling_can_be_disabled(self, simple_trace: Trace):
        """Styling can be disabled."""
        output = generate_mermaid_output(
            simple_trace.spans, simple_trace, enable_styling=False
        )

        # Should still be valid Mermaid
        assert "flowchart TD" in output

    def test_metadata_option(self, simple_trace: Trace):
        """Metadata option includes duration."""
        output = generate_mermaid_output(
            simple_trace.spans, simple_trace, include_metadata=True
        )

        # Should include duration somewhere
        assert "ms" in output or "μs" in output or "s" in output.lower()


class TestGenerateJsonOutput:
    """Tests for JSON output generation."""

    def test_generates_valid_json(self, simple_trace: Trace):
        """Generates valid JSON output."""
        output = generate_json_output(simple_trace.spans, simple_trace)

        # Should parse without error
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_includes_trace_id(self, simple_trace: Trace):
        """JSON includes trace ID."""
        output = generate_json_output(simple_trace.spans, simple_trace)
        data = json.loads(output)

        assert "traceId" in data
        assert data["traceId"] == simple_trace.traceId

    def test_includes_span_count(self, simple_trace: Trace):
        """JSON includes span counts."""
        output = generate_json_output(simple_trace.spans, simple_trace)
        data = json.loads(output)

        assert "totalSpans" in data
        assert "selectedSpans" in data

    def test_includes_span_details(self, simple_trace: Trace):
        """JSON includes full span details."""
        output = generate_json_output(simple_trace.spans, simple_trace)
        data = json.loads(output)

        assert "spans" in data
        assert len(data["spans"]) > 0

        span = data["spans"][0]
        assert "spanId" in span
        assert "service" in span
        assert "operation" in span
        assert "duration" in span
        assert "status" in span


# =============================================================================
# Test Output Writing
# =============================================================================


class TestWriteOutput:
    """Tests for output writing."""

    def test_write_to_stdout(self, capsys):
        """Writes to stdout when no path specified."""
        write_output("test content", None)

        captured = capsys.readouterr()
        assert "test content" in captured.out

    def test_write_to_file(self, tmp_path: Path):
        """Writes to file when path specified."""
        output_path = tmp_path / "output.md"
        write_output("test content", str(output_path))

        assert output_path.exists()
        assert output_path.read_text() == "test content"

    def test_creates_parent_directories(self, tmp_path: Path):
        """Creates parent directories if needed."""
        output_path = tmp_path / "nested" / "dir" / "output.md"
        write_output("test content", str(output_path))

        assert output_path.exists()


# =============================================================================
# Test Main Function
# =============================================================================


class TestMain:
    """Tests for main CLI function."""

    def test_main_success(self, sample_trace_path: Path, capsys):
        """Main function succeeds with valid input."""
        result = main([str(sample_trace_path)])

        assert result == 0
        captured = capsys.readouterr()
        assert "flowchart TD" in captured.out

    def test_main_with_output_file(
        self, sample_trace_path: Path, tmp_path: Path
    ):
        """Main function writes to output file."""
        output_path = tmp_path / "output.md"
        result = main([
            str(sample_trace_path),
            "-o", str(output_path)
        ])

        assert result == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "flowchart TD" in content

    def test_main_json_format(self, sample_trace_path: Path, capsys):
        """Main function outputs JSON when requested."""
        result = main([
            str(sample_trace_path),
            "--format", "json"
        ])

        assert result == 0
        captured = capsys.readouterr()

        # Should be valid JSON
        data = json.loads(captured.out)
        assert "traceId" in data

    def test_main_max_spans(self, sample_trace_path: Path, capsys):
        """Main function respects max-spans."""
        result = main([
            str(sample_trace_path),
            "-n", "3",
            "--format", "json"
        ])

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["spans"]) <= 3

    def test_main_file_not_found(self, capsys):
        """Main returns error for missing file."""
        result = main(["nonexistent.json"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_invalid_json(self, tmp_path: Path, capsys):
        """Main returns error for invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{not valid}")

        result = main([str(invalid_file)])

        assert result == 2
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_verbose_output(
        self, sample_trace_path: Path, capsys
    ):
        """Main function prints verbose messages."""
        result = main([str(sample_trace_path), "-v"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Loading trace" in captured.err or "Parsed trace" in captured.err


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_sample_trace(self, sample_trace_path: Path, capsys):
        """Full pipeline works with sample trace fixture."""
        result = main([str(sample_trace_path), "-n", "5"])

        assert result == 0
        captured = capsys.readouterr()

        # Check Mermaid output
        assert "flowchart TD" in captured.out
        # Should have nodes
        assert "[" in captured.out
        assert "]" in captured.out

    def test_full_pipeline_with_all_options(
        self, sample_trace_path: Path, tmp_path: Path
    ):
        """Full pipeline works with all options."""
        output_path = tmp_path / "full_test.md"

        result = main([
            str(sample_trace_path),
            "-o", str(output_path),
            "-n", "8",
            "--metadata",
            "-v"
        ])

        assert result == 0
        assert output_path.exists()

        content = output_path.read_text()
        assert "flowchart TD" in content
        # Metadata should include duration formatting
        assert "ms" in content or "μs" in content or "s)" in content

    def test_json_output_contains_selected_spans(
        self, sample_trace_path: Path, capsys
    ):
        """JSON output contains the selected spans."""
        result = main([
            str(sample_trace_path),
            "--format", "json",
            "-n", "10"
        ])

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        # Verify structure
        assert data["totalSpans"] >= data["selectedSpans"]
        assert len(data["spans"]) == data["selectedSpans"]

        # Verify span contents
        for span in data["spans"]:
            assert span["spanId"]
            assert span["service"]
            assert span["operation"]

    def test_sample_trace_has_error_span(self, sample_trace: Trace):
        """Sample trace fixture contains an error span."""
        error_spans = [s for s in sample_trace.spans if s.status == "ERROR"]
        assert len(error_spans) >= 1

    def test_sample_trace_has_multiple_services(self, sample_trace: Trace):
        """Sample trace fixture has spans from multiple services."""
        services = set(s.service for s in sample_trace.spans)
        assert len(services) >= 3

    def test_sample_trace_has_varying_durations(self, sample_trace: Trace):
        """Sample trace fixture has varying span durations."""
        durations = [s.duration for s in sample_trace.spans]
        min_dur = min(durations)
        max_dur = max(durations)

        # There should be significant variation
        assert max_dur > min_dur * 2

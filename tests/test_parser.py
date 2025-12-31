"""
Unit tests for tracemaid.core.parser module.

Tests cover Span and Trace dataclasses, and OTelParser functionality
for parsing various OpenTelemetry trace formats.
"""

import json
from typing import Any, Dict

import pytest

from tracemaid.core.parser import OTelParser, Span, Trace


class TestSpanDataclass:
    """Tests for Span dataclass validation and behavior."""

    def test_span_creation_minimal(self) -> None:
        """Test creating a span with minimal required fields."""
        span = Span(
            spanId="abc123",
            parentSpanId=None,
            service="test-service",
            operation="test-operation",
            duration=1000,
            status="OK"
        )
        assert span.spanId == "abc123"
        assert span.parentSpanId is None
        assert span.service == "test-service"
        assert span.operation == "test-operation"
        assert span.duration == 1000
        assert span.status == "OK"
        assert span.depth == 0
        assert span.children == []

    def test_span_creation_with_parent(self) -> None:
        """Test creating a span with a parent span ID."""
        span = Span(
            spanId="child123",
            parentSpanId="parent456",
            service="child-service",
            operation="child-op",
            duration=500,
            status="OK"
        )
        assert span.parentSpanId == "parent456"

    def test_span_creation_with_depth(self) -> None:
        """Test creating a span with explicit depth."""
        span = Span(
            spanId="deep123",
            parentSpanId="parent",
            service="svc",
            operation="op",
            duration=100,
            status="OK",
            depth=3
        )
        assert span.depth == 3

    def test_span_creation_with_children(self) -> None:
        """Test creating a span with children."""
        child = Span(
            spanId="child",
            parentSpanId="parent",
            service="svc",
            operation="child-op",
            duration=50,
            status="OK"
        )
        parent = Span(
            spanId="parent",
            parentSpanId=None,
            service="svc",
            operation="parent-op",
            duration=100,
            status="OK",
            children=[child]
        )
        assert len(parent.children) == 1
        assert parent.children[0].spanId == "child"

    def test_span_empty_spanid_raises_error(self) -> None:
        """Test that empty spanId raises ValueError."""
        with pytest.raises(ValueError, match="spanId cannot be empty"):
            Span(
                spanId="",
                parentSpanId=None,
                service="svc",
                operation="op",
                duration=100,
                status="OK"
            )

    def test_span_negative_duration_raises_error(self) -> None:
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration cannot be negative"):
            Span(
                spanId="abc",
                parentSpanId=None,
                service="svc",
                operation="op",
                duration=-100,
                status="OK"
            )

    def test_span_negative_depth_raises_error(self) -> None:
        """Test that negative depth raises ValueError."""
        with pytest.raises(ValueError, match="depth cannot be negative"):
            Span(
                spanId="abc",
                parentSpanId=None,
                service="svc",
                operation="op",
                duration=100,
                status="OK",
                depth=-1
            )

    def test_span_zero_duration_allowed(self) -> None:
        """Test that zero duration is allowed."""
        span = Span(
            spanId="abc",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=0,
            status="OK"
        )
        assert span.duration == 0

    def test_span_error_status(self) -> None:
        """Test span with ERROR status."""
        span = Span(
            spanId="err123",
            parentSpanId=None,
            service="svc",
            operation="failed-op",
            duration=100,
            status="ERROR"
        )
        assert span.status == "ERROR"

    def test_span_unset_status(self) -> None:
        """Test span with UNSET status."""
        span = Span(
            spanId="unset123",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=100,
            status="UNSET"
        )
        assert span.status == "UNSET"


class TestTraceDataclass:
    """Tests for Trace dataclass validation and behavior."""

    def test_trace_creation_minimal(self) -> None:
        """Test creating a trace with minimal required fields."""
        trace = Trace(
            traceId="trace123",
            spans=[],
            total_duration=0
        )
        assert trace.traceId == "trace123"
        assert trace.spans == []
        assert trace.total_duration == 0

    def test_trace_creation_with_spans(self) -> None:
        """Test creating a trace with spans."""
        span = Span(
            spanId="span1",
            parentSpanId=None,
            service="svc",
            operation="op",
            duration=1000,
            status="OK"
        )
        trace = Trace(
            traceId="trace123",
            spans=[span],
            total_duration=1000
        )
        assert len(trace.spans) == 1
        assert trace.spans[0].spanId == "span1"

    def test_trace_empty_traceid_raises_error(self) -> None:
        """Test that empty traceId raises ValueError."""
        with pytest.raises(ValueError, match="traceId cannot be empty"):
            Trace(traceId="", spans=[], total_duration=0)

    def test_trace_negative_total_duration_raises_error(self) -> None:
        """Test that negative total_duration raises ValueError."""
        with pytest.raises(ValueError, match="total_duration cannot be negative"):
            Trace(traceId="trace123", spans=[], total_duration=-100)

    def test_trace_root_span_property(self) -> None:
        """Test root_span property returns correct span."""
        root = Span(
            spanId="root",
            parentSpanId=None,
            service="svc",
            operation="root-op",
            duration=1000,
            status="OK"
        )
        child = Span(
            spanId="child",
            parentSpanId="root",
            service="svc",
            operation="child-op",
            duration=500,
            status="OK"
        )
        trace = Trace(
            traceId="trace123",
            spans=[root, child],
            total_duration=1000
        )
        assert trace.root_span is not None
        assert trace.root_span.spanId == "root"

    def test_trace_root_span_none_when_no_root(self) -> None:
        """Test root_span property returns None when no root span."""
        child = Span(
            spanId="child",
            parentSpanId="missing-parent",
            service="svc",
            operation="child-op",
            duration=500,
            status="OK"
        )
        trace = Trace(
            traceId="trace123",
            spans=[child],
            total_duration=500
        )
        assert trace.root_span is None

    def test_trace_span_count_property(self) -> None:
        """Test span_count property."""
        spans = [
            Span(spanId=f"span{i}", parentSpanId=None, service="svc",
                 operation="op", duration=100, status="OK")
            for i in range(5)
        ]
        trace = Trace(traceId="trace123", spans=spans, total_duration=100)
        assert trace.span_count == 5

    def test_trace_span_count_empty(self) -> None:
        """Test span_count property with no spans."""
        trace = Trace(traceId="trace123", spans=[], total_duration=0)
        assert trace.span_count == 0


class TestOTelParserSimpleFormat:
    """Tests for OTelParser with simplified span format."""

    def test_parse_simple_format(self) -> None:
        """Test parsing simplified format with direct spans array."""
        data = {
            "traceId": "abc123",
            "spans": [
                {
                    "spanId": "span1",
                    "parentSpanId": None,
                    "name": "root-operation",
                    "serviceName": "test-service",
                    "duration": 1000,
                    "status": {"code": 1}
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.traceId == "abc123"
        assert len(trace.spans) == 1
        assert trace.spans[0].operation == "root-operation"

    def test_parse_json_string(self) -> None:
        """Test parsing from JSON string."""
        data = {
            "traceId": "trace123",
            "spans": [
                {
                    "spanId": "span1",
                    "name": "operation",
                    "serviceName": "service",
                    "duration": 500,
                    "status": "OK"
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_json(json.dumps(data))

        assert trace.traceId == "trace123"
        assert len(trace.spans) == 1

    def test_parse_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises JSONDecodeError."""
        parser = OTelParser()
        with pytest.raises(json.JSONDecodeError):
            parser.parse_json("not valid json")

    def test_parse_missing_trace_id_from_spans(self) -> None:
        """Test extracting trace ID from spans when not at root."""
        data = {
            "spans": [
                {
                    "spanId": "span1",
                    "traceId": "extracted-trace-id",
                    "name": "op",
                    "duration": 100
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.traceId == "extracted-trace-id"


class TestOTelParserResourceSpansFormat:
    """Tests for OTelParser with OTLP resourceSpans format."""

    def test_parse_resource_spans_format(self) -> None:
        """Test parsing standard OTLP resourceSpans format."""
        data = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "my-service"}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "trace123",
                                    "spanId": "span1",
                                    "parentSpanId": "",
                                    "name": "GET /api",
                                    "startTimeUnixNano": "1704067200000000000",
                                    "endTimeUnixNano": "1704067200100000000",
                                    "status": {"code": 1}
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.traceId == "trace123"
        assert len(trace.spans) == 1
        assert trace.spans[0].service == "my-service"
        assert trace.spans[0].operation == "GET /api"

    def test_parse_resource_spans_with_dict_attributes(self) -> None:
        """Test parsing with dict-style resource attributes."""
        data = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": {
                            "service.name": "dict-service"
                        }
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "trace123",
                                    "spanId": "span1",
                                    "name": "operation",
                                    "duration": 100
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.spans[0].service == "dict-service"

    def test_parse_instrumentation_library_spans(self) -> None:
        """Test parsing older instrumentationLibrarySpans format."""
        data = {
            "resourceSpans": [
                {
                    "resource": {},
                    "instrumentationLibrarySpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "trace123",
                                    "spanId": "span1",
                                    "name": "operation",
                                    "duration": 100
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert len(trace.spans) == 1


class TestOTelParserJaegerFormat:
    """Tests for OTelParser with Jaeger-style format."""

    def test_parse_jaeger_format(self) -> None:
        """Test parsing Jaeger-style trace format."""
        data = {
            "data": [
                {
                    "traceID": "jaeger-trace-123",
                    "processes": {
                        "p1": {"serviceName": "jaeger-service"}
                    },
                    "spans": [
                        {
                            "spanID": "span1",
                            "processID": "p1",
                            "operationName": "jaeger-operation",
                            "duration": 5000
                        }
                    ]
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.traceId == "jaeger-trace-123"
        assert trace.spans[0].service == "jaeger-service"
        assert trace.spans[0].operation == "jaeger-operation"

    def test_parse_jaeger_format_with_traceid_lowercase(self) -> None:
        """Test parsing Jaeger format with lowercase traceId key."""
        data = {
            "data": [
                {
                    "traceId": "jaeger-trace-456",
                    "spans": [
                        {
                            "spanID": "span1",
                            "operationName": "op",
                            "duration": 100
                        }
                    ]
                }
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.traceId == "jaeger-trace-456"


class TestOTelParserSpanParsing:
    """Tests for span field parsing in OTelParser."""

    def test_parse_span_id_variants(self) -> None:
        """Test parsing different span ID field names."""
        variants = [
            {"spanId": "id1", "name": "op", "duration": 100},
            {"spanID": "id2", "name": "op", "duration": 100},
            {"span_id": "id3", "name": "op", "duration": 100},
        ]

        for variant in variants:
            data = {"traceId": "trace", "spans": [variant]}
            parser = OTelParser()
            trace = parser.parse_otlp(data)
            assert trace.spans[0].spanId in ["id1", "id2", "id3"]

    def test_parse_parent_span_id_variants(self) -> None:
        """Test parsing different parent span ID field names."""
        variants = [
            {"spanId": "s1", "parentSpanId": "p1", "name": "op", "duration": 100},
            {"spanId": "s2", "parentSpanID": "p2", "name": "op", "duration": 100},
            {"spanId": "s3", "parent_span_id": "p3", "name": "op", "duration": 100},
        ]

        for i, variant in enumerate(variants, 1):
            data = {"traceId": "trace", "spans": [variant]}
            parser = OTelParser()
            trace = parser.parse_otlp(data)
            assert trace.spans[0].parentSpanId == f"p{i}"

    def test_parse_empty_parent_span_id_as_none(self) -> None:
        """Test that empty string parent span ID becomes None."""
        data = {
            "traceId": "trace",
            "spans": [
                {"spanId": "s1", "parentSpanId": "", "name": "op", "duration": 100}
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].parentSpanId is None

    def test_parse_operation_name_variants(self) -> None:
        """Test parsing different operation name field names."""
        variants = [
            {"spanId": "s1", "name": "name-op", "duration": 100},
            {"spanId": "s2", "operationName": "opname-op", "duration": 100},
            {"spanId": "s3", "operation_name": "opund-op", "duration": 100},
        ]

        expected_ops = ["name-op", "opname-op", "opund-op"]
        for variant, expected in zip(variants, expected_ops):
            data = {"traceId": "trace", "spans": [variant]}
            parser = OTelParser()
            trace = parser.parse_otlp(data)
            assert trace.spans[0].operation == expected


class TestOTelParserDurationCalculation:
    """Tests for duration calculation in OTelParser."""

    def test_duration_direct_field(self) -> None:
        """Test duration from direct duration field."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "name": "op", "duration": 5000}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].duration == 5000

    def test_duration_from_nanosecond_timestamps(self) -> None:
        """Test duration calculation from nanosecond timestamps."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "startTimeUnixNano": "1704067200000000000",
                "endTimeUnixNano": "1704067200100000000"
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        # 100ms in nanoseconds = 100,000,000 ns = 100,000 µs
        assert trace.spans[0].duration == 100000

    def test_duration_from_integer_timestamps(self) -> None:
        """Test duration calculation from integer timestamps."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "startTimeUnixNano": 1704067200000000000,
                "endTimeUnixNano": 1704067200050000000
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        # 50ms in nanoseconds = 50,000,000 ns = 50,000 µs
        assert trace.spans[0].duration == 50000

    def test_duration_large_value_conversion(self) -> None:
        """Test duration conversion for very large nanosecond values."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 2000000000000000  # Very large, should be divided by 1000
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].duration == 2000000000000  # Divided by 1000

    def test_duration_float_value(self) -> None:
        """Test duration with float value."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 1234.5
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].duration == 1234


class TestOTelParserStatusExtraction:
    """Tests for status extraction in OTelParser."""

    def test_status_code_integer_ok(self) -> None:
        """Test status extraction from integer code (OK=1)."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "name": "op", "duration": 100, "status": {"code": 1}}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "OK"

    def test_status_code_integer_error(self) -> None:
        """Test status extraction from integer code (ERROR=2)."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "name": "op", "duration": 100, "status": {"code": 2}}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "ERROR"

    def test_status_code_integer_unset(self) -> None:
        """Test status extraction from integer code (UNSET=0)."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "name": "op", "duration": 100, "status": {"code": 0}}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "UNSET"

    def test_status_string(self) -> None:
        """Test status extraction from string status."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "name": "op", "duration": 100, "status": "error"}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "ERROR"

    def test_status_from_error_message(self) -> None:
        """Test status extraction from error message in status object."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 100,
                "status": {"message": "An error occurred"}
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "ERROR"

    def test_status_from_error_tag(self) -> None:
        """Test status extraction from error tag."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 100,
                "tags": [{"key": "error", "value": True}]
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "ERROR"

    def test_status_default_ok(self) -> None:
        """Test default status is OK when not specified."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "name": "op", "duration": 100}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "OK"


class TestOTelParserSpanTree:
    """Tests for span tree building in OTelParser."""

    def test_parent_child_relationship(self) -> None:
        """Test parent-child relationships are established correctly."""
        data = {
            "traceId": "trace",
            "spans": [
                {"spanId": "parent", "parentSpanId": None, "name": "parent-op", "duration": 1000},
                {"spanId": "child", "parentSpanId": "parent", "name": "child-op", "duration": 500}
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        parent = next(s for s in trace.spans if s.spanId == "parent")
        assert len(parent.children) == 1
        assert parent.children[0].spanId == "child"

    def test_depth_calculation(self) -> None:
        """Test depth calculation for nested spans."""
        data = {
            "traceId": "trace",
            "spans": [
                {"spanId": "root", "parentSpanId": None, "name": "root", "duration": 1000},
                {"spanId": "level1", "parentSpanId": "root", "name": "level1", "duration": 500},
                {"spanId": "level2", "parentSpanId": "level1", "name": "level2", "duration": 250}
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        root = next(s for s in trace.spans if s.spanId == "root")
        level1 = next(s for s in trace.spans if s.spanId == "level1")
        level2 = next(s for s in trace.spans if s.spanId == "level2")

        assert root.depth == 0
        assert level1.depth == 1
        assert level2.depth == 2

    def test_multiple_children(self) -> None:
        """Test parent with multiple children."""
        data = {
            "traceId": "trace",
            "spans": [
                {"spanId": "parent", "parentSpanId": None, "name": "parent", "duration": 1000},
                {"spanId": "child1", "parentSpanId": "parent", "name": "child1", "duration": 300},
                {"spanId": "child2", "parentSpanId": "parent", "name": "child2", "duration": 400},
                {"spanId": "child3", "parentSpanId": "parent", "name": "child3", "duration": 300}
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        parent = next(s for s in trace.spans if s.spanId == "parent")
        assert len(parent.children) == 3

    def test_orphan_span_handling(self) -> None:
        """Test handling of spans whose parent is not in the trace."""
        data = {
            "traceId": "trace",
            "spans": [
                {"spanId": "orphan", "parentSpanId": "missing", "name": "orphan", "duration": 100}
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        # Orphan should be treated as root (depth 0)
        assert trace.spans[0].depth == 0


class TestOTelParserAdditionalFormats:
    """Tests for additional OTLP format variations."""

    def test_single_trace_object_format(self) -> None:
        """Test parsing single trace object with traceId but no spans key."""
        data = {"traceId": "trace-single"}
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.traceId == "trace-single"
        assert len(trace.spans) == 0

    def test_dict_attribute_with_nested_stringvalue(self) -> None:
        """Test parsing dict-style attributes with nested stringValue."""
        data = {
            "resourceSpans": [{
                "resource": {
                    "attributes": {
                        "service.name": {"stringValue": "nested-service"}
                    }
                },
                "scopeSpans": [{
                    "spans": [{
                        "traceId": "trace456",
                        "spanId": "span1",
                        "name": "op",
                        "duration": 100
                    }]
                }]
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].service == "nested-service"

    def test_dict_attribute_with_string_value_key(self) -> None:
        """Test parsing dict-style attributes with string_value key."""
        data = {
            "resourceSpans": [{
                "resource": {
                    "attributes": {
                        "service.name": {"string_value": "string-value-service"}
                    }
                },
                "scopeSpans": [{
                    "spans": [{
                        "traceId": "trace789",
                        "spanId": "span1",
                        "name": "op",
                        "duration": 100
                    }]
                }]
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].service == "string-value-service"

    def test_status_code_string_field(self) -> None:
        """Test status extraction from statusCode string field."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 100,
                "status": {"statusCode": "ERROR"}
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "ERROR"

    def test_status_code_integer_via_statuscode_key(self) -> None:
        """Test status extraction from statusCode integer field."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 100,
                "status": {"statusCode": 2}
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "ERROR"


class TestOTelParserTimestampFormats:
    """Tests for various timestamp formats in OTelParser."""

    def test_duration_from_microsecond_timestamps(self) -> None:
        """Test duration calculation from microsecond timestamps."""
        data = {
            "traceId": "trace-us",
            "spans": [{
                "spanId": "span1",
                "name": "op",
                "startTime": 1704067200000000,
                "endTime": 1704067200100000
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].duration == 100000

    def test_duration_from_millisecond_timestamps(self) -> None:
        """Test duration calculation from millisecond timestamps."""
        data = {
            "traceId": "trace-ms",
            "spans": [{
                "spanId": "span1",
                "name": "op",
                "startTime": 1704067200000,
                "endTime": 1704067200100
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].duration == 100000

    def test_duration_from_small_timestamps(self) -> None:
        """Test duration calculation from small nanosecond timestamps."""
        data = {
            "traceId": "trace-small",
            "spans": [{
                "spanId": "span1",
                "name": "op",
                "startTimeUnixNano": 1000000000,
                "endTimeUnixNano": 1100000000
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        # 100ms in nanoseconds = 100,000 µs
        assert trace.spans[0].duration == 100000

    def test_duration_with_start_time_key(self) -> None:
        """Test duration using startTime key (not UnixNano)."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "span1",
                "name": "op",
                "start_time_unix_nano": 1704067200000000000,
                "end_time_unix_nano": 1704067200050000000
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].duration == 50000


class TestOTelParserEdgeCases:
    """Tests for edge cases in OTelParser."""

    def test_unsupported_format_raises_error(self) -> None:
        """Test that unsupported format raises ValueError."""
        data = {"unknown": "format"}
        parser = OTelParser()
        with pytest.raises(ValueError, match="Unsupported OTLP format"):
            parser.parse_otlp(data)

    def test_missing_trace_id_raises_error(self) -> None:
        """Test that missing trace ID raises ValueError."""
        data = {"spans": [{"spanId": "s1", "name": "op", "duration": 100}]}
        parser = OTelParser()
        with pytest.raises(ValueError, match="Could not determine trace ID"):
            parser.parse_otlp(data)

    def test_empty_spans_list(self) -> None:
        """Test parsing with empty spans list."""
        data = {"traceId": "trace", "spans": []}
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.traceId == "trace"
        assert len(trace.spans) == 0
        assert trace.total_duration == 0

    def test_total_duration_from_root_span(self) -> None:
        """Test total duration is taken from root span."""
        data = {
            "traceId": "trace",
            "spans": [
                {"spanId": "root", "parentSpanId": None, "name": "root", "duration": 1000},
                {"spanId": "child", "parentSpanId": "root", "name": "child", "duration": 500}
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.total_duration == 1000

    def test_total_duration_fallback_to_max(self) -> None:
        """Test total duration falls back to max when no root span."""
        data = {
            "traceId": "trace",
            "spans": [
                {"spanId": "s1", "parentSpanId": "missing", "name": "op1", "duration": 500},
                {"spanId": "s2", "parentSpanId": "missing", "name": "op2", "duration": 800}
            ]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.total_duration == 800

    def test_service_name_unknown_default(self) -> None:
        """Test service name defaults to 'unknown'."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "name": "op", "duration": 100}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.spans[0].service == "unknown"

    def test_operation_name_unknown_default(self) -> None:
        """Test operation name defaults to 'unknown'."""
        data = {
            "traceId": "trace",
            "spans": [{"spanId": "s1", "duration": 100}]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)

        assert trace.spans[0].operation == "unknown"

    def test_parser_can_be_reused(self) -> None:
        """Test that parser can parse multiple traces."""
        parser = OTelParser()

        trace1 = parser.parse_otlp({
            "traceId": "trace1",
            "spans": [{"spanId": "s1", "name": "op1", "duration": 100}]
        })

        trace2 = parser.parse_otlp({
            "traceId": "trace2",
            "spans": [{"spanId": "s2", "name": "op2", "duration": 200}]
        })

        assert trace1.traceId == "trace1"
        assert trace2.traceId == "trace2"
        assert trace1.spans[0].spanId == "s1"
        assert trace2.spans[0].spanId == "s2"

    def test_status_empty_string(self) -> None:
        """Test status extraction from empty string status."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 100,
                "status": ""
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "UNSET"

    def test_error_tag_false_value(self) -> None:
        """Test that error tag with false value doesn't mark as ERROR."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 100,
                "tags": [{"key": "error", "value": False}]
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "OK"

    def test_resource_spans_empty_scope_spans(self) -> None:
        """Test parsing with empty scopeSpans array."""
        data = {
            "resourceSpans": [{
                "resource": {},
                "scopeSpans": []
            }]
        }
        parser = OTelParser()
        with pytest.raises(ValueError, match="Could not determine trace ID"):
            parser.parse_otlp(data)

    def test_extract_trace_id_with_traceid_uppercase(self) -> None:
        """Test extracting trace ID with traceID key from spans."""
        data = {
            "spans": [{
                "spanId": "s1",
                "traceID": "upper-trace-id",
                "name": "op",
                "duration": 100
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.traceId == "upper-trace-id"

    def test_jaeger_processid_lowercase(self) -> None:
        """Test Jaeger format with processId (lowercase i)."""
        data = {
            "data": [{
                "traceID": "trace-jaeger",
                "processes": {
                    "p1": {"serviceName": "service-from-process"}
                },
                "spans": [{
                    "spanID": "span1",
                    "processId": "p1",
                    "operationName": "op",
                    "duration": 100
                }]
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].service == "service-from-process"

    def test_resource_attribute_list_with_string_value(self) -> None:
        """Test resource attribute list format with string_value key."""
        data = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"string_value": "attr-list-service"}}
                    ]
                },
                "scopeSpans": [{
                    "spans": [{
                        "traceId": "trace123",
                        "spanId": "span1",
                        "name": "op",
                        "duration": 100
                    }]
                }]
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].service == "attr-list-service"

    def test_status_unknown_code_defaults_to_unset(self) -> None:
        """Test unknown status code defaults to UNSET."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": 100,
                "status": {"code": 99}
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].status == "UNSET"

    def test_duration_string_type_returns_zero(self) -> None:
        """Test that non-numeric string duration returns zero."""
        data = {
            "traceId": "trace",
            "spans": [{
                "spanId": "s1",
                "name": "op",
                "duration": "invalid"
            }]
        }
        parser = OTelParser()
        trace = parser.parse_otlp(data)
        assert trace.spans[0].duration == 0

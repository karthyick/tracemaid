"""
tracemaid.core.parser - OpenTelemetry trace parsing module.

This module provides dataclasses and parser for converting OpenTelemetry
trace data into internal representations for analysis and visualization.

Classes:
    Span: Dataclass representing a single trace span
    Trace: Dataclass representing a complete trace with all spans
    OTelParser: Parser for OpenTelemetry JSON and OTLP formats
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    """Represents a single span in an OpenTelemetry trace.

    Attributes:
        spanId: Unique identifier for this span
        parentSpanId: ID of the parent span (None for root spans)
        service: Name of the service that generated this span
        operation: Name of the operation/method being traced
        duration: Duration of the span in microseconds
        status: Status code of the span (OK, ERROR, UNSET)
        depth: Nesting depth in the span tree (0 for root)
        children: List of child spans
    """
    spanId: str
    parentSpanId: Optional[str]
    service: str
    operation: str
    duration: int
    status: str
    depth: int = 0
    children: List[Span] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate span data after initialization."""
        if not self.spanId:
            raise ValueError("spanId cannot be empty")
        if self.duration < 0:
            raise ValueError("duration cannot be negative")
        if self.depth < 0:
            raise ValueError("depth cannot be negative")


@dataclass
class Trace:
    """Represents a complete OpenTelemetry trace.

    Attributes:
        traceId: Unique identifier for this trace
        spans: List of all spans in the trace
        total_duration: Total duration of the trace in microseconds
    """
    traceId: str
    spans: List[Span]
    total_duration: int

    def __post_init__(self) -> None:
        """Validate trace data after initialization."""
        if not self.traceId:
            raise ValueError("traceId cannot be empty")
        if self.total_duration < 0:
            raise ValueError("total_duration cannot be negative")

    @property
    def root_span(self) -> Optional[Span]:
        """Get the root span of the trace.

        Returns:
            The root span (parentSpanId is None) or None if not found
        """
        for span in self.spans:
            if span.parentSpanId is None:
                return span
        return None

    @property
    def span_count(self) -> int:
        """Get the total number of spans in the trace."""
        return len(self.spans)


class OTelParser:
    """Parser for OpenTelemetry trace formats.

    Supports parsing from:
    - JSON string format
    - OTLP (OpenTelemetry Protocol) data format

    Example:
        >>> parser = OTelParser()
        >>> trace = parser.parse_json(json_string)
        >>> print(trace.root_span.operation)
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._span_map: Dict[str, Span] = {}

    def parse_json(self, json_str: str) -> Trace:
        """Parse OpenTelemetry trace from JSON string.

        Args:
            json_str: JSON string containing OpenTelemetry trace data

        Returns:
            Trace object containing all parsed spans

        Raises:
            json.JSONDecodeError: If JSON is invalid
            ValueError: If required fields are missing
        """
        data = json.loads(json_str)
        return self.parse_otlp(data)

    def parse_otlp(self, data: Dict[str, Any]) -> Trace:
        """Parse OpenTelemetry trace from OTLP data structure.

        Supports multiple OTLP formats:
        - Standard OTLP export format with resourceSpans
        - Simplified format with direct spans array
        - Jaeger-style format with data/traces arrays

        Args:
            data: Dictionary containing OTLP trace data

        Returns:
            Trace object containing all parsed spans

        Raises:
            ValueError: If required fields are missing or data is invalid
        """
        self._span_map = {}
        raw_spans: List[Dict[str, Any]] = []
        trace_id: Optional[str] = None

        # Handle different OTLP formats
        if "resourceSpans" in data:
            # Standard OTLP export format
            raw_spans, trace_id = self._extract_from_resource_spans(data["resourceSpans"])
        elif "data" in data and isinstance(data["data"], list):
            # Jaeger-style format
            raw_spans, trace_id = self._extract_from_jaeger_format(data["data"])
        elif "spans" in data:
            # Simplified format with direct spans array
            raw_spans = data["spans"]
            trace_id = data.get("traceId") or self._extract_trace_id_from_spans(raw_spans)
        elif "traceId" in data and "spans" not in data:
            # Single trace object format
            raw_spans = data.get("spans", [])
            trace_id = data["traceId"]
        else:
            raise ValueError("Unsupported OTLP format: missing resourceSpans, data, or spans")

        if not trace_id:
            raise ValueError("Could not determine trace ID from data")

        # Parse all spans
        spans = self._parse_spans(raw_spans)

        # Build parent-child relationships and calculate depths
        self._build_span_tree(spans)

        # Calculate total duration
        total_duration = self._calculate_total_duration(spans)

        return Trace(
            traceId=trace_id,
            spans=spans,
            total_duration=total_duration
        )

    def _extract_from_resource_spans(
        self, resource_spans: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Extract spans from OTLP resourceSpans format.

        Args:
            resource_spans: List of resourceSpan objects

        Returns:
            Tuple of (raw_spans list, trace_id)
        """
        raw_spans: List[Dict[str, Any]] = []
        trace_id: Optional[str] = None

        for resource_span in resource_spans:
            # Get service name from resource attributes
            service_name = self._extract_service_name(resource_span.get("resource", {}))

            # Get spans from scopeSpans (newer format) or instrumentationLibrarySpans (older)
            scope_spans = resource_span.get("scopeSpans", [])
            if not scope_spans:
                scope_spans = resource_span.get("instrumentationLibrarySpans", [])

            for scope_span in scope_spans:
                for span in scope_span.get("spans", []):
                    # Inject service name into span for later use
                    span["_serviceName"] = service_name
                    raw_spans.append(span)

                    # Extract trace ID from first span
                    if trace_id is None:
                        trace_id = span.get("traceId")

        return raw_spans, trace_id

    def _extract_from_jaeger_format(
        self, data: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Extract spans from Jaeger-style format.

        Args:
            data: List containing trace data in Jaeger format

        Returns:
            Tuple of (raw_spans list, trace_id)
        """
        raw_spans: List[Dict[str, Any]] = []
        trace_id: Optional[str] = None

        for trace_data in data:
            if "traceID" in trace_data or "traceId" in trace_data:
                trace_id = trace_data.get("traceID") or trace_data.get("traceId")

            # Get processes (service mapping)
            processes = trace_data.get("processes", {})

            for span in trace_data.get("spans", []):
                # Resolve service name from process reference
                process_id = span.get("processID") or span.get("processId")
                if process_id and process_id in processes:
                    service_name = processes[process_id].get("serviceName", "unknown")
                    span["_serviceName"] = service_name
                raw_spans.append(span)

        return raw_spans, trace_id

    def _extract_service_name(self, resource: Dict[str, Any]) -> str:
        """Extract service name from resource attributes.

        Args:
            resource: Resource object containing attributes

        Returns:
            Service name string, defaults to 'unknown'
        """
        attributes = resource.get("attributes", [])

        # Handle both list and dict attribute formats
        if isinstance(attributes, list):
            for attr in attributes:
                key = attr.get("key", "")
                if key == "service.name":
                    value = attr.get("value", {})
                    return value.get("stringValue", value.get("string_value", "unknown"))
        elif isinstance(attributes, dict):
            service_attr = attributes.get("service.name")
            if service_attr:
                if isinstance(service_attr, str):
                    return service_attr
                return service_attr.get("stringValue", service_attr.get("string_value", "unknown"))

        return "unknown"

    def _extract_trace_id_from_spans(self, spans: List[Dict[str, Any]]) -> Optional[str]:
        """Extract trace ID from spans array.

        Args:
            spans: List of span dictionaries

        Returns:
            Trace ID string or None if not found
        """
        for span in spans:
            trace_id = span.get("traceId") or span.get("traceID")
            if trace_id:
                return trace_id
        return None

    def _parse_spans(self, raw_spans: List[Dict[str, Any]]) -> List[Span]:
        """Parse raw span dictionaries into Span objects.

        Args:
            raw_spans: List of span dictionaries from OTLP data

        Returns:
            List of Span objects
        """
        spans: List[Span] = []

        for raw_span in raw_spans:
            span = self._parse_single_span(raw_span)
            spans.append(span)
            self._span_map[span.spanId] = span

        return spans

    def _parse_single_span(self, raw_span: Dict[str, Any]) -> Span:
        """Parse a single span dictionary into a Span object.

        Args:
            raw_span: Dictionary containing span data

        Returns:
            Span object
        """
        # Get span ID (handle different key formats)
        span_id = (
            raw_span.get("spanId") or
            raw_span.get("spanID") or
            raw_span.get("span_id") or
            ""
        )

        # Get parent span ID
        parent_span_id = (
            raw_span.get("parentSpanId") or
            raw_span.get("parentSpanID") or
            raw_span.get("parent_span_id")
        )
        # Treat empty string as None
        if parent_span_id == "":
            parent_span_id = None

        # Get service name (may have been injected during parsing)
        service = raw_span.get("_serviceName") or raw_span.get("serviceName") or "unknown"

        # Get operation name
        operation = (
            raw_span.get("name") or
            raw_span.get("operationName") or
            raw_span.get("operation_name") or
            "unknown"
        )

        # Calculate duration
        duration = self._calculate_span_duration(raw_span)

        # Get status
        status = self._extract_status(raw_span)

        return Span(
            spanId=span_id,
            parentSpanId=parent_span_id,
            service=service,
            operation=operation,
            duration=duration,
            status=status,
            depth=0,  # Will be calculated in _build_span_tree
            children=[]
        )

    def _calculate_span_duration(self, raw_span: Dict[str, Any]) -> int:
        """Calculate span duration from raw span data.

        Args:
            raw_span: Dictionary containing span timing data

        Returns:
            Duration in microseconds
        """
        # Try direct duration field first (may be in microseconds already, e.g., Jaeger)
        if "duration" in raw_span:
            duration = raw_span["duration"]
            if isinstance(duration, int):
                # If duration looks like nanoseconds (very large), convert to microseconds
                if duration > 1_000_000_000_000:
                    return duration // 1000
                return duration
            if isinstance(duration, float):
                return int(duration)
            return 0

        # Calculate from start and end times (typically in nanoseconds for OTel)
        start_time = (
            raw_span.get("startTimeUnixNano") or
            raw_span.get("startTime") or
            raw_span.get("start_time_unix_nano") or
            0
        )
        end_time = (
            raw_span.get("endTimeUnixNano") or
            raw_span.get("endTime") or
            raw_span.get("end_time_unix_nano") or
            0
        )

        # Handle string timestamps
        if isinstance(start_time, str):
            start_time = int(start_time)
        if isinstance(end_time, str):
            end_time = int(end_time)

        # Calculate duration in the original unit
        duration_raw = end_time - start_time

        # Determine if timestamps are nanoseconds or microseconds
        # Nanosecond timestamps are typically > 1e18, microsecond are ~1e15-1e16
        if start_time > 1_000_000_000_000_000_000:
            # Nanoseconds - convert to microseconds
            return max(0, duration_raw // 1000)
        elif start_time > 1_000_000_000_000_000:
            # Already microseconds
            return max(0, duration_raw)
        elif start_time > 1_000_000_000_000:
            # Milliseconds - convert to microseconds
            return max(0, duration_raw * 1000)
        else:
            # Assume nanoseconds for standard OTel format
            return max(0, duration_raw // 1000)

    def _extract_status(self, raw_span: Dict[str, Any]) -> str:
        """Extract status code from span data.

        Args:
            raw_span: Dictionary containing span data

        Returns:
            Status string: 'OK', 'ERROR', or 'UNSET'
        """
        status = raw_span.get("status", {})

        if isinstance(status, str):
            return status.upper() if status else "UNSET"

        if isinstance(status, dict):
            # Check for status code
            code = status.get("code")
            if code is None:
                code = status.get("statusCode")
            if code is not None:
                if isinstance(code, int):
                    # OTLP status codes: 0=UNSET, 1=OK, 2=ERROR
                    return {0: "UNSET", 1: "OK", 2: "ERROR"}.get(code, "UNSET")
                if isinstance(code, str):
                    return code.upper()

            # Check message for error indication
            message = status.get("message", "")
            if message and "error" in message.lower():
                return "ERROR"

        # Check tags for error
        tags = raw_span.get("tags", [])
        for tag in tags:
            if isinstance(tag, dict):
                if tag.get("key") == "error" and tag.get("value"):
                    return "ERROR"

        return "OK"

    def _build_span_tree(self, spans: List[Span]) -> None:
        """Build parent-child relationships and calculate depths.

        Args:
            spans: List of Span objects to process
        """
        # First pass: establish parent-child relationships
        for span in spans:
            if span.parentSpanId and span.parentSpanId in self._span_map:
                parent = self._span_map[span.parentSpanId]
                if span not in parent.children:
                    parent.children.append(span)

        # Second pass: calculate depths using BFS
        root_spans = [s for s in spans if s.parentSpanId is None]

        # If no root spans found, find spans whose parent isn't in our span map
        if not root_spans:
            root_spans = [
                s for s in spans
                if s.parentSpanId not in self._span_map
            ]

        # Set root spans depth to 0 and propagate
        for root in root_spans:
            root.depth = 0
            self._calculate_depths_recursive(root)

    def _calculate_depths_recursive(self, span: Span) -> None:
        """Recursively calculate depths for span and its children.

        Args:
            span: Span to process
        """
        for child in span.children:
            child.depth = span.depth + 1
            self._calculate_depths_recursive(child)

    def _calculate_total_duration(self, spans: List[Span]) -> int:
        """Calculate total trace duration from spans.

        Args:
            spans: List of Span objects

        Returns:
            Total duration in nanoseconds (max of all span durations for root)
        """
        if not spans:
            return 0

        # Find root span duration (most accurate)
        for span in spans:
            if span.parentSpanId is None:
                return span.duration

        # Fallback: return max duration
        return max(span.duration for span in spans)

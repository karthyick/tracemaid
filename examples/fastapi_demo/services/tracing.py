"""
Shared tracing utilities for FastAPI demo services.

This module provides common tracing infrastructure used by all services
in the FastAPI demo application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SpanData:
    """Represents trace span data for tracemaid analysis.

    Attributes:
        span_id: Unique identifier for this span
        parent_span_id: ID of the parent span (None for root)
        trace_id: Trace identifier this span belongs to
        service: Name of the service
        operation: Name of the operation
        start_time: Start timestamp in nanoseconds
        end_time: End timestamp in nanoseconds
        status: Span status (OK, ERROR, UNSET)
        attributes: Additional span attributes
    """
    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    service: str
    operation: str
    start_time: int
    end_time: int
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_otlp_dict(self) -> Dict[str, Any]:
        """Convert to OTLP-compatible dictionary format.

        Returns:
            Dictionary in OTLP span format
        """
        return {
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "traceId": self.trace_id,
            "name": self.operation,
            "_serviceName": self.service,
            "startTimeUnixNano": str(self.start_time),
            "endTimeUnixNano": str(self.end_time),
            "status": {"code": 1 if self.status == "OK" else 2},
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in self.attributes.items()
            ]
        }

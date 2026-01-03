"""
TracemaidExporter - OpenTelemetry SpanExporter that generates Mermaid diagrams.

This exporter automatically converts OpenTelemetry traces to Mermaid diagrams
using tracemaid's intelligent span selection algorithms.

The exporter follows standard OpenTelemetry behavior:
- Traces are automatically exported when spans complete
- No need to call separate endpoints to retrieve trace data
- Mermaid diagrams are generated automatically

Example:
    >>> from opentelemetry import trace
    >>> from opentelemetry.sdk.trace import TracerProvider
    >>> from opentelemetry.sdk.trace.export import BatchSpanProcessor
    >>> from tracemaid.exporters import TracemaidExporter
    >>>
    >>> exporter = TracemaidExporter(
    ...     output_dir="./traces",
    ...     max_spans=15,
    ...     enable_styling=True
    ... )
    >>> provider = TracerProvider()
    >>> provider.add_span_processor(BatchSpanProcessor(exporter))
    >>> trace.set_tracer_provider(provider)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from tracemaid.core.parser import Span, Trace, OTelParser
from tracemaid.core.selector import SpanSelector
from tracemaid.core.mermaid import MermaidGenerator

logger = logging.getLogger(__name__)


class TracemaidExporter(SpanExporter):
    """OpenTelemetry SpanExporter that generates Mermaid diagrams from traces.

    This exporter collects spans and automatically generates Mermaid diagrams
    when a trace is complete (no more spans arriving for that trace).

    Attributes:
        output_dir: Directory to save Mermaid diagrams (None for console only)
        max_spans: Maximum spans to include in diagram (default: 15)
        enable_styling: Whether to apply error/slow span styling (default: True)
        include_metadata: Whether to include duration/depth in labels (default: False)
        on_diagram_generated: Optional callback when a diagram is generated

    Example:
        >>> exporter = TracemaidExporter(
        ...     output_dir="./traces",
        ...     max_spans=20,
        ...     on_diagram_generated=lambda trace_id, diagram: print(f"Generated: {trace_id}")
        ... )
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        max_spans: int = 15,
        enable_styling: bool = True,
        include_metadata: bool = False,
        console_output: bool = True,
        on_diagram_generated: Optional[Callable[[str, str], None]] = None,
        flush_interval_seconds: float = 2.0,
    ) -> None:
        """Initialize the TracemaidExporter.

        Args:
            output_dir: Directory to save Mermaid diagrams. If None, only console output.
            max_spans: Maximum number of spans to include in diagram.
            enable_styling: Whether to apply error/slow span styling.
            include_metadata: Whether to include duration/depth in node labels.
            console_output: Whether to print diagrams to console.
            on_diagram_generated: Optional callback(trace_id, diagram) when diagram is generated.
            flush_interval_seconds: Time to wait before considering a trace complete.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_spans = max_spans
        self.enable_styling = enable_styling
        self.include_metadata = include_metadata
        self.console_output = console_output
        self.on_diagram_generated = on_diagram_generated
        self.flush_interval_seconds = flush_interval_seconds

        # Thread-safe storage for collecting spans by trace_id
        self._lock = threading.Lock()
        self._trace_spans: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._trace_last_update: Dict[str, float] = {}

        # Core tracemaid components
        self._parser = OTelParser()
        self._selector = SpanSelector()
        self._generator = MermaidGenerator()

        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("TracemaidExporter: Saving diagrams to %s", self.output_dir)

        logger.info(
            "TracemaidExporter initialized: max_spans=%d, styling=%s, output_dir=%s",
            self.max_spans,
            self.enable_styling,
            self.output_dir,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export a batch of spans.

        This method is called automatically by the OpenTelemetry SDK when spans
        are ready to be exported. It collects spans by trace_id and generates
        Mermaid diagrams when traces are complete.

        Traces are considered complete when:
        1. A root span (no parent) is received, OR
        2. No new spans have arrived for flush_interval_seconds

        Args:
            spans: Sequence of completed spans to export.

        Returns:
            SpanExportResult.SUCCESS on successful processing.
        """
        if not spans:
            return SpanExportResult.SUCCESS

        logger.debug("Exporting %d spans", len(spans))
        current_time = datetime.now().timestamp()
        traces_with_root: set = set()

        with self._lock:
            # Group spans by trace_id
            for span in spans:
                trace_id = format(span.context.trace_id, "032x")
                span_dict = self._readable_span_to_dict(span)
                self._trace_spans[trace_id].append(span_dict)
                self._trace_last_update[trace_id] = current_time
                logger.debug("Collected span '%s' for trace %s (parent: %s)",
                           span.name, trace_id[:8], span.parent)

                # Check if this is a root span (no parent)
                if not span.parent:
                    traces_with_root.add(trace_id)
                    logger.info("Root span detected for trace %s, marking for processing", trace_id[:8])

            # Check for complete traces:
            # 1. Traces that received a root span (immediately complete)
            # 2. Traces with no updates for flush_interval (timeout-based completion)
            complete_traces = list(traces_with_root)
            for trace_id, last_update in list(self._trace_last_update.items()):
                if trace_id not in traces_with_root:
                    if current_time - last_update >= self.flush_interval_seconds:
                        complete_traces.append(trace_id)

        # Process complete traces outside the lock
        for trace_id in complete_traces:
            self._process_trace(trace_id)

        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all pending traces.

        Generates Mermaid diagrams for all collected traces immediately,
        regardless of flush_interval_seconds.

        Args:
            timeout_millis: Maximum time to wait for flush (milliseconds).

        Returns:
            True if flush succeeded, False otherwise.
        """
        with self._lock:
            trace_ids = list(self._trace_spans.keys())

        for trace_id in trace_ids:
            self._process_trace(trace_id)

        return True

    def shutdown(self) -> None:
        """Shutdown the exporter.

        Flushes all pending traces before shutdown.
        """
        logger.info("TracemaidExporter shutting down, flushing pending traces...")
        self.force_flush()
        logger.info("TracemaidExporter shutdown complete")

    def _readable_span_to_dict(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert a ReadableSpan to OTLP-compatible dictionary format.

        Args:
            span: The OpenTelemetry ReadableSpan to convert.

        Returns:
            Dictionary in OTLP span format that tracemaid parser can process.
        """
        # Extract span context
        context = span.context
        parent_id = span.parent.span_id if span.parent else None

        # Determine service name from resource attributes
        service_name = "unknown"
        if span.resource and span.resource.attributes:
            service_name = span.resource.attributes.get(
                "service.name", "unknown"
            )

        # Determine status
        status_code = 1  # OK
        if span.status and span.status.status_code:
            from opentelemetry.trace import StatusCode
            if span.status.status_code == StatusCode.ERROR:
                status_code = 2  # ERROR

        # Convert attributes
        attributes = []
        if span.attributes:
            for key, value in span.attributes.items():
                attributes.append({
                    "key": key,
                    "value": {"stringValue": str(value)}
                })

        return {
            "traceId": format(context.trace_id, "032x"),
            "spanId": format(context.span_id, "016x"),
            "parentSpanId": format(parent_id, "016x") if parent_id else "",
            "name": span.name,
            "_serviceName": service_name,
            "startTimeUnixNano": str(span.start_time) if span.start_time else "0",
            "endTimeUnixNano": str(span.end_time) if span.end_time else "0",
            "status": {"code": status_code},
            "attributes": attributes,
        }

    def _process_trace(self, trace_id: str) -> None:
        """Process a complete trace and generate Mermaid diagram.

        Args:
            trace_id: The trace ID to process.
        """
        with self._lock:
            if trace_id not in self._trace_spans:
                return
            spans = self._trace_spans.pop(trace_id)
            self._trace_last_update.pop(trace_id, None)

        if not spans:
            return

        try:
            # Build OTLP format for parser
            otlp_data = {
                "traceId": trace_id,
                "spans": spans
            }

            # Parse trace using tracemaid parser
            trace = self._parser.parse_otlp(otlp_data)

            if not trace or not trace.spans:
                logger.warning("No spans parsed for trace %s", trace_id)
                return

            # Select important spans using tracemaid's ML algorithms
            selected_spans = self._selector.select_from_trace(
                trace,
                max_spans=self.max_spans
            )

            # Generate Mermaid diagram
            diagram = self._generator.generate(
                selected_spans,
                trace,
                enable_styling=self.enable_styling
            )

            # Output the diagram
            self._output_diagram(trace_id, diagram, len(spans), len(selected_spans))

        except Exception as e:
            logger.error(
                "Failed to generate Mermaid diagram for trace %s: %s",
                trace_id,
                str(e),
                exc_info=True
            )

    def _output_diagram(
        self,
        trace_id: str,
        diagram: str,
        total_spans: int,
        selected_spans: int
    ) -> None:
        """Output the generated Mermaid diagram.

        Args:
            trace_id: The trace ID.
            diagram: The generated Mermaid diagram.
            total_spans: Total number of spans in the trace.
            selected_spans: Number of spans selected for the diagram.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Console output
        if self.console_output:
            print(f"\n{'='*60}")
            print(f"TRACE: {trace_id}")
            print(f"Spans: {selected_spans}/{total_spans} selected")
            print(f"{'='*60}")
            print(diagram)
            print(f"{'='*60}\n")

        # File output
        if self.output_dir:
            filename = f"trace_{trace_id[:8]}_{timestamp}.mmd"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"%%{{init: {{'theme': 'default'}}}}%%\n")
                f.write(f"%% Trace ID: {trace_id}\n")
                f.write(f"%% Generated: {datetime.now().isoformat()}\n")
                f.write(f"%% Spans: {selected_spans}/{total_spans} selected\n\n")
                f.write(diagram)

            logger.info("Mermaid diagram saved: %s", filepath)

        # Callback
        if self.on_diagram_generated:
            try:
                self.on_diagram_generated(trace_id, diagram)
            except Exception as e:
                logger.error("Callback failed for trace %s: %s", trace_id, str(e))

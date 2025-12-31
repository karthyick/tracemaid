"""
tracemaid.core.mermaid - Mermaid diagram generation module.

This module provides functionality to convert selected important spans
into Mermaid flowchart diagram syntax for visualization.

Classes:
    MermaidGenerator: Generates Mermaid flowchart diagrams from spans
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from tracemaid.core.parser import Span, Trace


@dataclass
class MermaidStyle:
    """Style configuration for Mermaid diagram elements.

    Attributes:
        error_color: Fill color for error spans (default: red)
        slow_color: Fill color for slow spans (default: orange)
        slow_threshold_percentile: Percentile above which spans are considered slow
        node_shape: Shape for regular nodes (default: rectangle with rounded corners)
    """
    error_color: str = "#ff6b6b"
    error_stroke: str = "#c92a2a"
    slow_color: str = "#ffa94d"
    slow_stroke: str = "#e67700"
    normal_color: str = "#74c0fc"
    normal_stroke: str = "#1c7ed6"
    slow_threshold_percentile: float = 90.0
    node_shape_start: str = "["
    node_shape_end: str = "]"


class MermaidGenerator:
    """Generates Mermaid flowchart diagrams from trace spans.

    This class converts a list of selected important spans into valid
    Mermaid flowchart syntax. It supports:
    - Node generation with service and operation labels
    - Edge generation for parent-child relationships
    - Styling for error and slow spans

    Example:
        >>> from tracemaid.core.mermaid import MermaidGenerator
        >>> from tracemaid.core.parser import Span, Trace
        >>>
        >>> generator = MermaidGenerator()
        >>> spans = [...]  # List of Span objects
        >>> trace = Trace(...)
        >>> diagram = generator.generate(spans, trace)
        >>> print(diagram)
        flowchart TD
            span1[service-a: get_user]
            span2[service-b: query_db]
            span1 --> span2
    """

    # Characters that need escaping in Mermaid labels
    SPECIAL_CHARS = {
        '"': '#quot;',
        "'": '#apos;',
        '<': '#lt;',
        '>': '#gt;',
        '&': '#amp;',
        '[': '#91;',
        ']': '#93;',
        '{': '#123;',
        '}': '#125;',
        '(': '#40;',
        ')': '#41;',
        '|': '#124;',
        '\\': '#92;',
        '/': '#47;',
    }

    def __init__(self, style: Optional[MermaidStyle] = None) -> None:
        """Initialize the MermaidGenerator.

        Args:
            style: Optional style configuration. Uses defaults if not provided.
        """
        self.style = style or MermaidStyle()
        self._span_set: Set[str] = set()
        self._all_spans: Dict[str, Span] = {}
        self._slow_threshold: float = 0.0

    def generate(
        self,
        spans: List[Span],
        trace: Trace,
        enable_styling: bool = True,
        title: Optional[str] = None
    ) -> str:
        """Generate a Mermaid flowchart diagram from spans.

        Args:
            spans: List of selected important spans to include in diagram
            trace: The complete trace object for context
            enable_styling: Whether to apply styling for error/slow spans
            title: Optional title for the diagram

        Returns:
            Mermaid flowchart diagram as a string
        """
        if not spans:
            return "flowchart TD\n    empty[No spans to display]"

        # Build lookup structures
        self._span_set = {span.spanId for span in spans}
        self._all_spans = {span.spanId: span for span in trace.spans}

        # Calculate slow threshold based on durations
        if enable_styling and spans:
            durations = sorted([s.duration for s in spans])
            idx = int(len(durations) * self.style.slow_threshold_percentile / 100)
            idx = min(idx, len(durations) - 1)
            self._slow_threshold = durations[idx] if durations else 0

        lines: List[str] = []

        # Header
        lines.append("flowchart TD")

        # Optional title as subgraph
        if title:
            escaped_title = self._escape_label(title)
            lines.append(f"    subgraph {escaped_title}")

        # Generate nodes
        for span in spans:
            node_line = self._generate_node(span)
            lines.append(f"    {node_line}")

        # Close title subgraph if used
        if title:
            lines.append("    end")

        # Generate edges
        edges = self._generate_edges(spans)
        for edge in edges:
            lines.append(f"    {edge}")

        # Generate styles if enabled
        if enable_styling:
            style_lines = self._generate_styles(spans)
            for style_line in style_lines:
                lines.append(f"    {style_line}")

        return "\n".join(lines)

    def _generate_node(self, span: Span) -> str:
        """Generate a Mermaid node definition for a span.

        Creates a node with format: spanId[service: operation]
        Special characters in labels are escaped.

        Args:
            span: The span to generate a node for

        Returns:
            Mermaid node definition string
        """
        # Create sanitized node ID (Mermaid IDs can only contain alphanumeric and underscore)
        node_id = self._sanitize_node_id(span.spanId)

        # Create label with service and operation
        label = f"{span.service}: {span.operation}"
        escaped_label = self._escape_label(label)

        # Build node with configured shape
        shape_start = self.style.node_shape_start
        shape_end = self.style.node_shape_end

        return f'{node_id}{shape_start}"{escaped_label}"{shape_end}'

    def _generate_edges(self, spans: List[Span]) -> List[str]:
        """Generate Mermaid edge definitions for parent-child relationships.

        Only includes edges where both parent and child are in the
        selected spans list.

        Args:
            spans: List of selected spans

        Returns:
            List of Mermaid edge definition strings
        """
        edges: List[str] = []

        for span in spans:
            if span.parentSpanId and span.parentSpanId in self._span_set:
                # Both parent and child are in selection
                parent_id = self._sanitize_node_id(span.parentSpanId)
                child_id = self._sanitize_node_id(span.spanId)
                edges.append(f"{parent_id} --> {child_id}")
            elif span.parentSpanId:
                # Parent exists but not in selection - find nearest ancestor in selection
                ancestor_id = self._find_nearest_selected_ancestor(span.parentSpanId)
                if ancestor_id:
                    parent_id = self._sanitize_node_id(ancestor_id)
                    child_id = self._sanitize_node_id(span.spanId)
                    # Use dotted line to indicate indirect relationship
                    edges.append(f"{parent_id} -.-> {child_id}")

        return edges

    def _find_nearest_selected_ancestor(self, parent_span_id: str) -> Optional[str]:
        """Find the nearest ancestor that is in the selected spans.

        Args:
            parent_span_id: The immediate parent span ID

        Returns:
            The span ID of the nearest selected ancestor, or None
        """
        current_id = parent_span_id
        visited: Set[str] = set()

        while current_id and current_id not in visited:
            visited.add(current_id)

            if current_id in self._span_set:
                return current_id

            # Look up parent
            parent_span = self._all_spans.get(current_id)
            if parent_span and parent_span.parentSpanId:
                current_id = parent_span.parentSpanId
            else:
                break

        return None

    def _generate_styles(self, spans: List[Span]) -> List[str]:
        """Generate Mermaid style definitions for spans.

        Applies different styles for:
        - Error spans (red)
        - Slow spans (orange)
        - Normal spans (blue)

        Args:
            spans: List of spans to style

        Returns:
            List of Mermaid style definition strings
        """
        styles: List[str] = []
        error_nodes: List[str] = []
        slow_nodes: List[str] = []
        normal_nodes: List[str] = []

        for span in spans:
            node_id = self._sanitize_node_id(span.spanId)

            if span.status.upper() == "ERROR":
                error_nodes.append(node_id)
            elif span.duration >= self._slow_threshold and self._slow_threshold > 0:
                slow_nodes.append(node_id)
            else:
                normal_nodes.append(node_id)

        # Generate class definitions
        if error_nodes or slow_nodes or normal_nodes:
            styles.append("")  # Empty line before styles
            styles.append("%% Style definitions")

        if error_nodes:
            styles.append(
                f"classDef errorStyle fill:{self.style.error_color},"
                f"stroke:{self.style.error_stroke},stroke-width:2px,color:#fff"
            )
            node_list = ",".join(error_nodes)
            styles.append(f"class {node_list} errorStyle")

        if slow_nodes:
            styles.append(
                f"classDef slowStyle fill:{self.style.slow_color},"
                f"stroke:{self.style.slow_stroke},stroke-width:2px,color:#000"
            )
            node_list = ",".join(slow_nodes)
            styles.append(f"class {node_list} slowStyle")

        if normal_nodes:
            styles.append(
                f"classDef normalStyle fill:{self.style.normal_color},"
                f"stroke:{self.style.normal_stroke},stroke-width:1px,color:#000"
            )
            node_list = ",".join(normal_nodes)
            styles.append(f"class {node_list} normalStyle")

        return styles

    def _escape_label(self, label: str) -> str:
        """Escape special characters in a Mermaid label.

        Args:
            label: The raw label text

        Returns:
            Escaped label safe for use in Mermaid
        """
        escaped = label
        for char, replacement in self.SPECIAL_CHARS.items():
            escaped = escaped.replace(char, replacement)
        return escaped

    def _sanitize_node_id(self, span_id: str) -> str:
        """Sanitize a span ID for use as a Mermaid node ID.

        Mermaid node IDs can only contain alphanumeric characters and underscores.

        Args:
            span_id: The original span ID

        Returns:
            Sanitized node ID safe for Mermaid
        """
        # Replace any non-alphanumeric characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', span_id)

        # Ensure it doesn't start with a number (add prefix if needed)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"span_{sanitized}"

        # Handle empty result
        if not sanitized:
            sanitized = "unknown_span"

        return sanitized

    def generate_with_metadata(
        self,
        spans: List[Span],
        trace: Trace,
        include_duration: bool = True,
        include_depth: bool = False
    ) -> str:
        """Generate a Mermaid diagram with additional metadata in labels.

        Args:
            spans: List of selected important spans
            trace: The complete trace object
            include_duration: Whether to include duration in labels
            include_depth: Whether to include depth in labels

        Returns:
            Mermaid flowchart diagram as a string
        """
        if not spans:
            return "flowchart TD\n    empty[No spans to display]"

        # Build lookup structures
        self._span_set = {span.spanId for span in spans}
        self._all_spans = {span.spanId: span for span in trace.spans}

        # Calculate slow threshold
        durations = sorted([s.duration for s in spans])
        idx = int(len(durations) * self.style.slow_threshold_percentile / 100)
        idx = min(idx, len(durations) - 1)
        self._slow_threshold = durations[idx] if durations else 0

        lines: List[str] = ["flowchart TD"]

        # Generate nodes with metadata
        for span in spans:
            node_id = self._sanitize_node_id(span.spanId)

            # Build label with optional metadata
            label_parts = [f"{span.service}: {span.operation}"]

            if include_duration:
                duration_str = self._format_duration(span.duration)
                label_parts.append(f"({duration_str})")

            if include_depth:
                label_parts.append(f"[depth: {span.depth}]")

            label = " ".join(label_parts)
            escaped_label = self._escape_label(label)

            lines.append(
                f'    {node_id}{self.style.node_shape_start}'
                f'"{escaped_label}"{self.style.node_shape_end}'
            )

        # Generate edges
        edges = self._generate_edges(spans)
        for edge in edges:
            lines.append(f"    {edge}")

        # Generate styles
        style_lines = self._generate_styles(spans)
        for style_line in style_lines:
            lines.append(f"    {style_line}")

        return "\n".join(lines)

    def _format_duration(self, duration_us: int) -> str:
        """Format duration in a human-readable way.

        Args:
            duration_us: Duration in microseconds

        Returns:
            Formatted duration string
        """
        if duration_us < 1000:
            return f"{duration_us}μs"
        elif duration_us < 1_000_000:
            return f"{duration_us / 1000:.1f}ms"
        else:
            return f"{duration_us / 1_000_000:.2f}s"

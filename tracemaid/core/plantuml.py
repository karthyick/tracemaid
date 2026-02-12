"""
tracemaid.core.plantuml - PlantUML diagram generation module.

This module provides functionality to convert selected important spans
into PlantUML activity diagram syntax for visualization.

Classes:
    PlantUMLGenerator: Generates PlantUML activity diagrams from spans
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from tracemaid.core.parser import Span, Trace


@dataclass
class PlantUMLStyle:
    """Style configuration for PlantUML diagram elements.

    Attributes:
        error_color: Fill color for error spans (default: Red)
        slow_color: Fill color for slow spans (default: Orange)
        normal_color: Fill color for normal spans (default: LightBlue)
        slow_threshold_percentile: Percentile above which spans are considered slow
        box_type: PlantUML box type for nodes (e.g., "rectangle", "round_rectangle")
        diagram_type: PlantUML diagram type (e.g., "activity", "sequence")
    """

    error_color: str = "Red"
    slow_color: str = "Orange"
    normal_color: str = "LightBlue"
    slow_threshold_percentile: float = 90.0
    box_type: str = ""  # For activity diagram, no explicit box type needed for default
    diagram_type: str = "activity"


class PlantUMLGenerator:
    """Generates PlantUML activity diagrams from trace spans.

    This class converts a list of selected important spans into valid
    PlantUML activity diagram syntax. It supports:
    - Node generation with service and operation labels
    - Edge generation for parent-child relationships
    - Styling for error and slow spans

    Example:
        >>> from tracemaid.core.plantuml import PlantUMLGenerator
        >>> from tracemaid.core.parser import Span, Trace
        >>>
        >>> generator = PlantUMLGenerator()
        >>> spans = [...]  # List of Span objects
        >>> trace = Trace(...)
        >>> diagram = generator.generate(spans, trace)
        >>> print(diagram)
        @startuml activity
            :service-a: get_user;
            :service-b: query_db;
            :service-a: get_user --> :service-b: query_db;
        @enduml
    """

    # Characters that need escaping in PlantUML labels
    # Using specific PlantUML escaping where available, otherwise generic
    SPECIAL_CHARS = {
        '"': r"\"",
        "\\": r"\\",  # Escape backslash
        ";": r"\;",
        "{": r"\{",
        "}": r"\}",
        "<": r"\<",  # Escape angle brackets
        ">": r"\>",
        "|": r"\|",
        "[": r"\[",  # Escape square brackets
        "]": r"\]",
    }

    def __init__(self, style: Optional[PlantUMLStyle] = None) -> None:
        """Initialize the PlantUMLGenerator.

        Args:
            style: Optional style configuration. Uses defaults if not provided.
        """
        self.style = style or PlantUMLStyle()
        self._span_set: Set[str] = set()
        self._all_spans: Dict[str, Span] = {}
        self._slow_threshold: float = 0.0

    def _initialize_generator_state(self, spans: List[Span], trace: Trace) -> None:
        """Initializes the generator's internal state for a new diagram generation.

        This includes building span lookup sets and calculating the slow threshold.
        """
        self._span_set = {span.spanId for span in spans}
        self._all_spans = {span.spanId: span for span in trace.spans}

        if len(spans) > 1:  # Only calculate meaningful slow threshold for multiple spans
            durations = sorted([s.duration for s in spans])
            idx = int(len(durations) * self.style.slow_threshold_percentile / 100)
            idx = min(idx, len(durations) - 1)
            self._slow_threshold = durations[idx]
        else:
            self._slow_threshold = 0.0

    def generate(
        self,
        spans: List[Span],
        trace: Trace,
        enable_styling: bool = True,
        title: Optional[str] = None,
    ) -> str:
        """Generate a PlantUML activity diagram from spans.

        Args:
            spans: List of selected important spans to include in diagram
            trace: The complete trace object for context
            enable_styling: Whether to apply styling for error/slow spans
            title: Optional title for the diagram

        Returns:
            PlantUML activity diagram as a string
        """
        if not spans:
            return f"@startuml {self.style.diagram_type}\n    :No spans to display;\n@enduml"

        self._initialize_generator_state(spans, trace)

        lines: List[str] = []

        # Header
        lines.append(f"@startuml {self.style.diagram_type}")

        # Optional title
        if title:
            # PlantUML activity diagram title is usually separate or a floating note
            # For simplicity, we'll add it as a top-level note for now.
            escaped_title = self._escape_label(title)
            lines.append(f"title {escaped_title}")

        # Generate nodes
        node_definitions = self._generate_nodes(spans, enable_styling)
        for node_def in node_definitions:
            lines.append(f"    {node_def}")

        # Generate edges
        edges = self._generate_edges(spans)
        for edge in edges:
            lines.append(f"    {edge}")

        # Generate styles if enabled
        if enable_styling:
            style_lines = self._generate_styles(spans)
            for style_line in style_lines:
                lines.append(f"    {style_line}")

        # Footer
        lines.append("@enduml")

        return "\n".join(lines)

    def _generate_nodes(self, spans: List[Span], enable_styling: bool) -> List[str]:
        """Generate PlantUML activity node definitions for spans.

        Creates nodes with format: :service: operation; as activity_id
        Special characters in labels are escaped. Colors are applied via skinparam.

        Args:
            spans: The spans to generate nodes for
            enable_styling: Whether to apply styling (colors) to nodes

        Returns:
            List of PlantUML node definition strings
        """
        node_lines: List[str] = []
        for span in spans:
            # For PlantUML activities, we use activity syntax with ID
            label = f"{span.service}: {span.operation}"
            escaped_label = self._escape_label(label)

            # Store node ID for edge generation
            node_id = self._sanitize_node_id(span.spanId)

            # PlantUML activity syntax: :label; as ID
            node_lines.append(f":{escaped_label}; as {node_id}")

        return node_lines

    def _generate_edges(self, spans: List[Span]) -> List[str]:
        """Generate PlantUML edge definitions for parent-child relationships.

        For PlantUML activity diagrams, we use activity flow syntax.
        This generates the sequential flow between activities.

        Args:
            spans: List of selected spans

        Returns:
            List of PlantUML edge definition strings
        """
        edges: List[str] = []

        # Sort spans by start time or depth to create logical flow
        # For simplicity, we'll use parent-child relationships
        span_map = {span.spanId: span for span in spans}

        for span in spans:
            if span.parentSpanId and span.parentSpanId in self._span_set:
                # Direct parent-child relationship
                parent = span_map.get(span.parentSpanId)
                if parent:
                    parent_id = self._sanitize_node_id(span.parentSpanId)
                    child_id = self._sanitize_node_id(span.spanId)
                    # PlantUML activity flow syntax
                    edges.append(f"{parent_id} --> {child_id}")
            elif span.parentSpanId:
                # Indirect relationship - find nearest ancestor
                ancestor_id = self._find_nearest_selected_ancestor(span.parentSpanId)
                if ancestor_id:
                    parent_id = self._sanitize_node_id(ancestor_id)
                    child_id = self._sanitize_node_id(span.spanId)
                    # Use dotted line for indirect relationship
                    edges.append(f"{parent_id} ..> {child_id}")

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

    def _get_span_color(self, span: Span, enable_styling: bool) -> Optional[str]:
        """Determine the color for a span based on its status and duration.

        Args:
            span: The span to check
            enable_styling: Whether to apply styling (colors)

        Returns:
            PlantUML color string or None if no specific color applies and styling is disabled
        """
        if not enable_styling:
            return None
        if span.status.upper() == "ERROR":
            return self.style.error_color
        elif self._slow_threshold > 0 and span.duration >= self._slow_threshold:
            return self.style.slow_color
        return self.style.normal_color  # Always return a color for normal spans too

    def _generate_styles(self, spans: List[Span]) -> List[str]:
        """Generate PlantUML style definitions using skinparam.

        PlantUML supports styling through skinparam statements that apply
        colors based on node IDs or stereotypes.

        Args:
            spans: List of spans to style

        Returns:
            List of PlantUML style definition strings
        """
        styles: List[str] = []

        # Generate skinparam for each span based on its type
        for span in spans:
            node_id = self._sanitize_node_id(span.spanId)
            color = self._get_span_color(span, True)
            if color:
                styles.append(f"skinparam activity{node_id}BackgroundColor {color}")

        return styles

    def _escape_label(self, label: str) -> str:
        """Escape special characters in a PlantUML label.

        Args:
            label: The raw label text

        Returns:
            Escaped label safe for use in PlantUML
        """
        escaped_chars = []
        for char in label:
            if char in self.SPECIAL_CHARS:
                escaped_chars.append(self.SPECIAL_CHARS[char])
            else:
                escaped_chars.append(char)
        return "".join(escaped_chars)

    def _sanitize_node_id(self, span_id: str) -> str:
        """Sanitize a span ID for use as a PlantUML node alias.

        PlantUML aliases can contain alphanumeric characters and underscores.
        They can't start with a digit.

        Args:
            span_id: The original span ID

        Returns:
            Sanitized node ID safe for PlantUML alias
        """
        # Replace any non-alphanumeric characters with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9]", "_", span_id)

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
        include_depth: bool = False,
        enable_styling: bool = True,
    ) -> str:
        """Generate a PlantUML diagram with additional metadata in labels.

        Args:
            spans: List of selected important spans
            trace: The complete trace object
            include_duration: Whether to include duration in labels
            include_depth: Whether to include depth in labels

        Returns:
            PlantUML activity diagram as a string
        """
        if not spans:
            return f"@startuml {self.style.diagram_type}\n    :No spans to display;\n@enduml"

        self._initialize_generator_state(spans, trace)

        lines: List[str] = [f"@startuml {self.style.diagram_type}"]

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

            # PlantUML activity syntax with ID
            lines.append(f"    :{escaped_label}; as {node_id}")

        # Generate edges
        edges = self._generate_edges(spans)
        for edge in edges:
            lines.append(f"    {edge}")

        # Generate styles
        if enable_styling:
            style_lines = self._generate_styles(spans)
            for style_line in style_lines:
                lines.append(f"    {style_line}")

        lines.append("@enduml")

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

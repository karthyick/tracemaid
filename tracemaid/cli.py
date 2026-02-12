"""
tracemaid.cli - Command-line interface for tracemaid.

This module provides a CLI for processing OpenTelemetry traces,
selecting important spans, and generating Mermaid diagrams.

Usage:
    tracemaid <input_file> [--output/-o <file>] [--max-spans/-n <count>] [--format <format>]

Examples:
    tracemaid trace.json
    tracemaid trace.json -o diagram.md
    tracemaid trace.json -n 15 --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from tracemaid import __version__
from tracemaid.core.parser import OTelParser, Span, Trace
from tracemaid.core.selector import SpanSelector
from tracemaid.core.mermaid import MermaidGenerator
from tracemaid.core.plantuml import PlantUMLGenerator
from tracemaid.core.advanced_filter import FilterCriterion, FilterOperator, apply_advanced_filter


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: List of arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="tracemaid",
        description="Analyze OpenTelemetry traces and generate Mermaid diagrams",
        epilog="Example: tracemaid trace.json -o diagram.md -n 10",
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input trace file (JSON format)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (defaults to stdout)",
    )

    parser.add_argument(
        "-n",
        "--max-spans",
        type=int,
        default=10,
        help="Maximum number of spans to include in the diagram (default: 10)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["mermaid", "json", "plantuml"],
        default="mermaid",
        help="Output format: mermaid diagram or json (default: mermaid)",
    )

    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Disable styling in Mermaid diagrams",
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Include duration and depth metadata in labels",
    )

    # Advanced filtering arguments
    parser.add_argument(
        "--filter-attribute",
        type=str,
        action="append",
        help="Attribute name to filter by. Can be specified multiple times for multiple criteria. E.g., --filter-attribute service --filter-operator eq --filter-value my-service",
    )
    parser.add_argument(
        "--filter-operator",
        type=str,
        action="append",
        choices=[op.value for op in FilterOperator],  # Use .value for actual string values
        help=f"Comparison operator for filtering. Must be specified with --filter-attribute and --filter-value. Available: {[op.value for op in FilterOperator]}",
    )
    parser.add_argument(
        "--filter-value",
        type=str,
        action="append",
        help="Value to compare against for filtering. Can be specified multiple times.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parsed_args = parser.parse_args(args)

    # Validate advanced filtering arguments
    filter_attrs = parsed_args.filter_attribute
    filter_ops = parsed_args.filter_operator
    filter_vals = parsed_args.filter_value

    if (filter_attrs is not None) != (filter_ops is not None) or (filter_attrs is not None) != (
        filter_vals is not None
    ):
        parser.error(
            "If using advanced filtering, --filter-attribute, --filter-operator, "
            "and --filter-value must all be provided an equal number of times."
        )

    if filter_attrs is not None and not (len(filter_attrs) == len(filter_ops) == len(filter_vals)):
        parser.error(
            "The number of --filter-attribute, --filter-operator, and --filter-value "
            "arguments must be equal."
        )

    return parsed_args


def load_trace(input_path: str) -> Trace:
    """Load and parse a trace from a JSON file.

    Args:
        input_path: Path to the input JSON file

    Returns:
        Parsed Trace object

    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If the trace format is invalid
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(path, "r", encoding="utf-8") as f:
        json_str = f.read()

    parser = OTelParser()
    return parser.parse_json(json_str)


def select_important_spans(trace: Trace, max_spans: int) -> List[Span]:
    """Select the most important spans from a trace.

    Args:
        trace: The trace to analyze
        max_spans: Maximum number of spans to select

    Returns:
        List of selected Span objects
    """
    selector = SpanSelector()
    return selector.select_from_trace(trace, max_spans=max_spans)


def generate_mermaid_output(
    spans: List[Span],
    trace: Trace,
    enable_styling: bool = True,
    include_metadata: bool = False,
) -> str:
    """Generate Mermaid diagram from selected spans.

    Args:
        spans: List of selected spans
        trace: The complete trace for context
        enable_styling: Whether to apply styling
        include_metadata: Whether to include duration/depth in labels

    Returns:
        Mermaid diagram as a string
    """
    generator = MermaidGenerator()

    if include_metadata:
        return generator.generate_with_metadata(
            spans, trace, include_duration=True, include_depth=True
        )
    else:
        return generator.generate(spans, trace, enable_styling=enable_styling)


def generate_plantuml_output(
    spans: List[Span],
    trace: Trace,
    enable_styling: bool = True,
    include_metadata: bool = False,
) -> str:
    """Generate PlantUML diagram from selected spans.

    Args:
        spans: List of selected spans
        trace: The complete trace for context
        enable_styling: Whether to apply styling
        include_metadata: Whether to include duration/depth in labels

    Returns:
        PlantUML diagram as a string
    """
    generator = PlantUMLGenerator()

    if include_metadata:
        return generator.generate_with_metadata(
            spans, trace, include_duration=True, include_depth=True, enable_styling=enable_styling
        )
    else:
        return generator.generate(spans, trace, enable_styling=enable_styling)


def generate_json_output(spans: List[Span], trace: Trace) -> str:
    """Generate JSON output with selected spans.

    Args:
        spans: List of selected spans
        trace: The complete trace for context

    Returns:
        JSON string with span information
    """
    output = {
        "traceId": trace.traceId,
        "totalSpans": len(trace.spans),
        "selectedSpans": len(spans),
        "spans": [
            {
                "spanId": span.spanId,
                "parentSpanId": span.parentSpanId,
                "service": span.service,
                "operation": span.operation,
                "duration": span.duration,
                "status": span.status,
                "depth": span.depth,
            }
            for span in spans
        ],
    }

    return json.dumps(output, indent=2)


def write_output(content: str, output_path: str | None) -> None:
    """Write content to output file or stdout.

    Args:
        content: The content to write
        output_path: Path to output file, or None for stdout
    """
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print(content)


def main(args: List[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        parsed_args = parse_args(args)

        if parsed_args.verbose:
            print(f"Loading trace from: {parsed_args.input_file}", file=sys.stderr)

        # Load and parse the trace
        trace = load_trace(parsed_args.input_file)

        if parsed_args.verbose:
            print(
                f"Parsed trace with {len(trace.spans)} spans",
                file=sys.stderr,
            )

        # Apply advanced filtering if arguments are provided
        initial_spans: List[Span] = trace.spans
        if parsed_args.filter_attribute:
            filter_criteria: List[FilterCriterion] = []
            for i in range(len(parsed_args.filter_attribute)):
                filter_criteria.append(
                    FilterCriterion(
                        attribute=parsed_args.filter_attribute[i],
                        operator=FilterOperator(
                            parsed_args.filter_operator[i]
                        ),  # Ensure operator is FilterOperator enum
                        value=parsed_args.filter_value[i],
                    )
                )

            if parsed_args.verbose:
                print(
                    f"Applying advanced filter with {len(filter_criteria)} criteria",
                    file=sys.stderr,
                )

            initial_spans = apply_advanced_filter(initial_spans, filter_criteria)
            trace.spans = initial_spans  # Update trace spans for further processing

            if parsed_args.verbose:
                print(
                    f"Advanced filtering resulted in {len(initial_spans)} spans",
                    file=sys.stderr,
                )

        # Select important spans
        selected_spans = select_important_spans(trace, parsed_args.max_spans)

        if parsed_args.verbose:
            print(
                f"Selected {len(selected_spans)} important spans",
                file=sys.stderr,
            )

        # Generate output based on format
        if parsed_args.format == "json":
            output = generate_json_output(selected_spans, trace)
        elif parsed_args.format == "plantuml":
            output = generate_plantuml_output(
                selected_spans,
                trace,
                enable_styling=not parsed_args.no_style,
                include_metadata=parsed_args.metadata,
            )
        else:  # Default to mermaid
            output = generate_mermaid_output(
                selected_spans,
                trace,
                enable_styling=not parsed_args.no_style,
                include_metadata=parsed_args.metadata,
            )

        # Write output
        write_output(output, parsed_args.output)

        if parsed_args.verbose and parsed_args.output:
            print(f"Output written to: {parsed_args.output}", file=sys.stderr)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 2

    except ValueError as e:
        print(f"Error: Invalid trace format: {e}", file=sys.stderr)
        return 3

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

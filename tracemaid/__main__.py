"""
tracemaid.__main__ - Entry point for running tracemaid as a module.

Usage:
    python -m tracemaid <input_file> [options]

This module enables running tracemaid using:
    python -m tracemaid trace.json
"""

from tracemaid.cli import main

if __name__ == "__main__":
    exit(main())

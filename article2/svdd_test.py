#!/usr/bin/env python3
"""Compatibility entry point for the unified direct-parameter SVDD pipeline."""

from src.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main() or 0)

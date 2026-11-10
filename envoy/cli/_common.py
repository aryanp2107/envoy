"""
Shared utilities for the envoy CLI: logging setup and error handling.

Each subcommand imports from here to get consistent behavior.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure logging format and level based on CLI verbosity flags.

    Default (neither flag): INFO level, no timestamps on console.
    --verbose: DEBUG level, full timestamps + module names.
    --quiet: WARN level, no timestamps.

    Called once at CLI entry point. All library logging routes through this.
    """
    if verbose:
        level = logging.DEBUG
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    elif quiet:
        level = logging.WARN
        fmt = "[%(levelname)s] %(message)s"
    else:
        level = logging.INFO
        fmt = "[%(levelname)s] %(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # keep stdout clean for pipeable output
    )


def handle_cli_error(err: Exception, verbose: bool = False) -> int:
    """
    Print a clean error message and return a nonzero exit code.

    In --verbose mode, re-raise the exception so Python shows the full
    traceback (useful for debugging). Otherwise print a single line and exit.

    Returns the exit code to pass to sys.exit().
    """
    if verbose:
        raise err
    # Classify common errors for better messages
    if isinstance(err, FileNotFoundError):
        print(f"envoy: file not found: {err.filename or err}", file=sys.stderr)
        return 2
    if isinstance(err, PermissionError):
        print(f"envoy: permission denied: {err.filename or err}", file=sys.stderr)
        return 2
    if isinstance(err, ValueError):
        print(f"envoy: invalid input: {err}", file=sys.stderr)
        return 2
    if isinstance(err, RuntimeError):
        print(f"envoy: {err}", file=sys.stderr)
        return 1
    # Unknown error: show the class name so bug reports are informative
    print(f"envoy: {type(err).__name__}: {err}", file=sys.stderr)
    print("       (run with --verbose for full traceback)", file=sys.stderr)
    return 1


def resolve_path(p: str | Path, must_exist: bool = False) -> Path:
    """
    Expand user (~) and resolve relative paths.

    If must_exist=True, raises FileNotFoundError when the path doesn't exist.
    This is used by subcommands to validate --input paths before kicking off
    work that would fail later anyway.
    """
    path = Path(p).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")
    return path


def add_common_flags(parser) -> None:
    """
    Add --verbose and --quiet to any subcommand parser.

    All subcommands get these flags. Mutually exclusive with each other
    (you wouldn't want both at once).
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging and show full tracebacks on error.",
    )
    group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Show only warnings and errors.",
    )
"""
ENVOY command-line interface.

This is a stub for Delivery 1 so `pip install -e .` can register the entry point.
Full subcommand dispatch lands in Delivery 4.
"""

import sys

from envoy import __version__


def main() -> int:
    """Stub entry point. Prints version and exits."""
    if len(sys.argv) > 1 and sys.argv[1] in ("-v", "--version"):
        print(f"envoy {__version__}")
        return 0
    print(f"envoy {__version__}")
    print("CLI subcommands land in Delivery 4. This is a Delivery 1 stub.")
    print("Available after Delivery 4: envoy qc | dedup | embed | cluster | figures | divergence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
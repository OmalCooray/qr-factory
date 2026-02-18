"""Unified entry point for qr-factory source code modules."""

import logging
import sys

log = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        logging.basicConfig(level=logging.INFO)
        log.error("Usage: python -m src <module> [args...]")
        log.error("Available modules:")
        log.error("  loader    - Data loading from MT5")
        sys.exit(1)
    
    module = sys.argv[1]
    
    if module == "loader":
        from src.loader._loader import main as loader_main
        loader_main(sys.argv[2:])  # Pass remaining args
    else:
        logging.basicConfig(level=logging.INFO)
        log.error(f"Unknown module: {module}")
        sys.exit(1)

if __name__ == "__main__":
    main()

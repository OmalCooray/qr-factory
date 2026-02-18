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
        log.error("  backtest  - Run deterministic backtest")
        sys.exit(1)
    
    module = sys.argv[1]
    
    if module == "loader":
        from src.loader._loader import main as loader_main
        loader_main(sys.argv[2:])  # Pass remaining args
    elif module == "backtest":
        import argparse
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
        p = argparse.ArgumentParser(description="Run deterministic backtest")
        p.add_argument("--config", required=True, help="Path to YAML config file")
        args = p.parse_args(sys.argv[2:])
        from src.engine.runner import run_backtest
        run_id = run_backtest(args.config)
        log.info("Finished — run_id: %s", run_id)
    else:
        logging.basicConfig(level=logging.INFO)
        log.error(f"Unknown module: {module}")
        sys.exit(1)

if __name__ == "__main__":
    main()

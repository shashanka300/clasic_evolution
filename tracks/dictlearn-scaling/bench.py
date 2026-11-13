"""Entry point alias for run_bench.py."""

from pathlib import Path
import sys

TRACK_DIR = Path(__file__).resolve().parent
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from run_bench import main

if __name__ == "__main__":
    sys.exit(main())

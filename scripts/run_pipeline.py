"""Compatibility wrapper for the old pipeline name.

The current project version is modeling-first and does not run ML training.
Use `python main.py` for the official workflow.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    main()

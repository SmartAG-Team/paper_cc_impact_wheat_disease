"""Run the manuscript workflow from any working directory."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper.workflow import main


if __name__ == "__main__":
    main()

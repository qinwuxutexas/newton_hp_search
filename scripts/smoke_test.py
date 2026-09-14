"""Run: python scripts/smoke_test.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newton_hpo.cli import smoke

if __name__ == "__main__":
    smoke()

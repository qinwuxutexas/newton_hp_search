"""Multi-dataset benchmark (breast cancer, wine, digits).

    python scripts/run_benchmark.py --newton-iter 12 --trials 30
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newton_hpo.cli import run_benchmark

if __name__ == "__main__":
    run_benchmark()

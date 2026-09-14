"""Compare Newton HPO vs random vs TPE on breast cancer.

    python scripts/run_single.py --newton-iter 8 --trials 20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newton_hpo.cli import run_single

if __name__ == "__main__":
    run_single()

"""Newton HPO ablations: damping, initialization, dimension.

    python scripts/run_ablations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from newton_hpo.cli import run_ablations

if __name__ == "__main__":
    run_ablations()

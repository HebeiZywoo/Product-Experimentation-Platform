from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    subprocess.run([sys.executable, script], cwd=ROOT, check=True)


def main() -> None:
    run("scripts/generate_data.py")
    run("scripts/run_analysis.py")
    run("scripts/run_sql_analysis.py")
    print("Pipeline complete")


if __name__ == "__main__":
    main()


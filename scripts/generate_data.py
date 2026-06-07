from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product_experiment_ds.data_generation import generate_all



def main() -> None:
    tables = generate_all(ROOT / "data" / "raw")
    for name, table in tables.items():
        print(f"{name}: {len(table):,} rows")


if __name__ == "__main__":
    main()

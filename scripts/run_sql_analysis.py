from __future__ import annotations

import sys
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
ANALYTICS_DIR = ROOT / "analytics"
SQL_PATH = ROOT / "sql" / "product_experiment_analysis.sql"


def main() -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ANALYTICS_DIR / "product_experimentation.duckdb"))

    required = ["user_features.csv", "uplift_scores.csv"]
    missing = [name for name in required if not (PROCESSED_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed outputs: {', '.join(missing)}. Run scripts/run_analysis.py first.")

    con.execute(
        """
        CREATE OR REPLACE TABLE user_features AS
        SELECT * FROM read_csv_auto(?);
        """,
        [str(PROCESSED_DIR / "user_features.csv")],
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE uplift_scores AS
        SELECT * FROM read_csv_auto(?);
        """,
        [str(PROCESSED_DIR / "uplift_scores.csv")],
    )
    con.execute(SQL_PATH.read_text())

    output_tables = [
        "user_funnel",
        "segment_experiment",
        "channel_cohorts",
        "campaign_observational_summary",
        "uplift_priority",
    ]
    for table in output_tables:
        output_path = PROCESSED_DIR / f"sql_{table}.csv"
        con.execute(f"COPY {table} TO ? (HEADER, DELIMITER ',')", [str(output_path)])
        print(f"Saved {output_path}")
    con.close()


if __name__ == "__main__":
    sys.exit(main())


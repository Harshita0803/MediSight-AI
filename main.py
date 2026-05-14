import argparse
import sys

from etl.pipeline import run, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="MediSight ETL pipeline")
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Path to Synthea CSV output directory (default: CSV_DIR env var or 'output/csv')",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="PostgreSQL connection string (default: DATABASE_URL env var)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    result = run(csv_dir=args.csv_dir, dsn=args.dsn)
    print("\nPipeline complete.")
    print(f"  Elapsed : {result['elapsed_seconds']}s")
    print(f"  Rows    : {result['row_counts']}")
    print(f"  QA      : {result['validation']['passed']} passed, "
          f"{result['validation']['warnings']} warnings, "
          f"{result['validation']['errors']} errors")
    sys.exit(0 if result["validation"]["errors"] == 0 else 1)


if __name__ == "__main__":
    main()

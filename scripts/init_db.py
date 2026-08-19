"""Create or explicitly rebuild the local relational telecom-support database."""

import argparse
import sys
from pathlib import Path

# Support the documented command: ``python scripts/init_db.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.persistence.telecom_seed import format_statistics, initialize_telecom_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Import IBM Telco churn data and deterministic demo support records.")
    parser.add_argument(
        "action",
        nargs="?",
        default="initialize",
        choices=("initialize", "seed", "reset"),
        help="initialize/create+seed safely, seed a blank telecom schema, or explicitly reset telecom data.",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Explicitly replace telecom tables; call snapshots are preserved."
    )
    parser.add_argument(
        "--database-url", default=settings.database_url, help="SQLModel database URL; defaults to DATABASE_URL."
    )
    args = parser.parse_args()
    stats = initialize_telecom_database(args.database_url, reset=args.reset or args.action == "reset")
    print("Database initialization successful\n")
    print(format_statistics(stats.items()))
    print("\nIntegrity checks: PASSED")


if __name__ == "__main__":
    main()

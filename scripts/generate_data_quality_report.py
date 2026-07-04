import logging
import os
from pathlib import Path

import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from ydata_profiling import ProfileReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

OUTPUT_PATH = Path("reports/data_quality_report.html")


def main() -> None:
    db_url = os.environ["DATABASE_URL"].replace("postgres://", "postgresql://")
    if "sslmode" not in db_url:
        db_url += "?sslmode=require"
    engine = sqlalchemy.create_engine(db_url)

    logger.info("Loading ml_encounter_features...")
    df = pd.read_sql("SELECT * FROM ml_encounter_features", engine)
    engine.dispose()
    logger.info("Loaded %d rows × %d columns", *df.shape)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile = ProfileReport(
        df,
        title="MediSight Encounter Features Quality Report",
        minimal=True,
    )
    profile.to_file(str(OUTPUT_PATH))
    logger.info("Report saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()

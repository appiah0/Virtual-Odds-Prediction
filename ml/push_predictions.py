import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import fetch_fixtures, get_client
from predict import predict_dataframe

MODEL_PATH = str(Path(__file__).resolve().parent / "instant_football_xgb.joblib")
PREDICTIONS_TABLE = "live_predictions"


def score_and_push_fixtures(scraped_rows: list[dict] | pd.DataFrame, history_df: pd.DataFrame | None = None):
    """Score scraped pre-match fixtures with XGBoost and upsert into Supabase."""
    upcoming_df = pd.DataFrame(scraped_rows) if isinstance(scraped_rows, list) else scraped_rows.copy()
    if upcoming_df.empty:
        return []

    client = get_client()
    if history_df is None:
        history_df = fetch_fixtures(client, settled=True)

    preds = predict_dataframe(upcoming_df, history_df=history_df, path=MODEL_PATH)
    records = preds.to_dict(orient="records")

    if records:
        client.table(PREDICTIONS_TABLE).upsert(records, on_conflict="fixture_key").execute()

    return records

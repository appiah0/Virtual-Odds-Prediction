import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import fetch_fixtures, get_client
from predict import predict_dataframe

MODEL_PATH = str(Path(__file__).resolve().parent / "instant_football_xgb.joblib")
PREDICTIONS_TABLE = "live_predictions"


def score_and_push_fixtures(
    scraped_rows: list[dict] | pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Score scraped pre-match fixtures with XGBoost and upsert into live_predictions."""
    upcoming_df = (
        pd.DataFrame(scraped_rows)
        if isinstance(scraped_rows, list)
        else scraped_rows.copy()
    )
    if upcoming_df.empty:
        return []

    client = get_client()
    if history_df is None:
        history_df = fetch_fixtures(client, settled=True)

    preds = predict_dataframe(upcoming_df, history_df=history_df, path=MODEL_PATH)

    # Rename dotted O/U keys from predict_dataframe to match live_predictions columns
    rename_map = {
        "under_1.5_pct": "under_15_pct",
        "over_1.5_pct": "over_15_pct",
        "under_2.5_pct": "under_25_pct",
        "over_2.5_pct": "over_25_pct",
        "under_3.5_pct": "under_35_pct",
        "over_3.5_pct": "over_35_pct",
    }
    preds = preds.rename(columns=rename_map)

    keep_cols = [
        "fixture_key",
        "round_no",
        "league",
        "home_team",
        "away_team",
        "home_win_pct",
        "draw_pct",
        "away_win_pct",
        "pick_1x2",
        "under_15_pct",
        "over_15_pct",
        "under_25_pct",
        "over_25_pct",
        "under_35_pct",
        "over_35_pct",
    ]
    preds = preds[[c for c in keep_cols if c in preds.columns]].copy()
    preds["updated_at"] = datetime.now(timezone.utc).isoformat()

    records = preds.to_dict(orient="records")
    if records:
        client.table(PREDICTIONS_TABLE).upsert(records, on_conflict="fixture_key").execute()

    return records

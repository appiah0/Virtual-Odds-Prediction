"""Predict probabilities for pre-match rows.

CLI:   python predict.py              # all unsettled rows in Supabase
       python predict.py --round 180  # one round only
API:   from predict import predict_dataframe
"""
import argparse

import joblib
import pandas as pd

from features import build_features, fetch_fixtures, get_client

ARTIFACT = "instant_football_xgb.joblib"
_bundle = None


def load_bundle(path=ARTIFACT):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def predict_dataframe(new_df, history_df=None, path=ARTIFACT):
    """new_df: pre-match rows (outcome/scores may be null).
    history_df: settled fixtures used for rolling form (fetched from Supabase if None)."""
    bundle = load_bundle(path)
    new_df = new_df.copy()
    if history_df is None:
        history_df = fetch_fixtures(get_client(), settled=True)
    history_df = history_df.copy()
    if "fixture_key" in new_df and "fixture_key" in history_df:
        history_df = history_df[~history_df["fixture_key"].isin(new_df["fixture_key"])]
    # Never let a new row's own result (if present) influence its features.
    new_df[["home_score", "away_score", "outcome"]] = None
    new_df["_new"], history_df["_new"] = True, False
    if "id" not in new_df:
        new_df["id"] = range(10**9, 10**9 + len(new_df))

    feats = build_features(pd.concat([history_df, new_df], ignore_index=True))
    feats = feats[feats["_new"].astype(bool)].reset_index(drop=True)
    X = feats[bundle["features"]]

    out = feats[[c for c in ("fixture_key", "round_no", "league", "home_team", "away_team")
                 if c in feats.columns]].copy()
    p = bundle["models"]["1x2"].predict_proba(X)
    out["home_win_pct"], out["draw_pct"], out["away_win_pct"] = (p * 100).round(1).T
    out["pick_1x2"] = pd.Series(p.argmax(1)).map({0: "Home", 1: "Draw", 2: "Away"})
    for k in ("1.5", "2.5", "3.5"):
        po = bundle["models"][f"ou_{k}"].predict_proba(X)[:, 1]
        out[f"under_{k}_pct"] = ((1 - po) * 100).round(1)
        out[f"over_{k}_pct"] = (po * 100).round(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None)
    args = ap.parse_args()
    upcoming = fetch_fixtures(get_client(), settled=False)
    if args.round is not None:
        upcoming = upcoming[upcoming["round_no"] == args.round]
    if upcoming.empty:
        print("No pre-match rows found.")
        return
    pd.set_option("display.width", 220, "display.max_columns", None)
    print(predict_dataframe(upcoming).to_string(index=False))


if __name__ == "__main__":
    main()


"""Predict probabilities for pre-match rows.

CLI:   python predict.py              # all unsettled rows in Supabase
       python predict.py --round 180  # one round only
API:   from predict import predict_dataframe
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from features import build_features, fetch_fixtures, get_client

ARTIFACT = "instant_football_xgb.joblib"
_bundle = None


def load_bundle(path=ARTIFACT):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def _blend_1x2(p_xgb, feats, alpha):
    if alpha is None or alpha >= 1.0:
        return p_xgb
    cols = ["imp_home", "imp_draw", "imp_away"]
    if not all(c in feats.columns for c in cols):
        return p_xgb
    p_book = feats[cols].to_numpy(dtype=float)
    out = p_xgb.copy()
    valid = ~np.isnan(p_book).any(axis=1)
    if np.any(valid):
        blended = alpha * p_xgb[valid] + (1.0 - alpha) * p_book[valid]
        row_sum = blended.sum(axis=1, keepdims=True)
        out[valid] = np.where(row_sum > 0, blended / row_sum, p_xgb[valid])
    return out


def _blend_binary(p_xgb, feats, col, alpha):
    if alpha is None or alpha >= 1.0 or col not in feats.columns:
        return p_xgb
    p_book = feats[col].to_numpy(dtype=float)
    out = p_xgb.copy()
    valid = ~np.isnan(p_book)
    if np.any(valid):
        out[valid] = alpha * p_xgb[valid] + (1.0 - alpha) * p_book[valid]
    return np.clip(out, 0.0, 1.0)


def predict_dataframe(new_df, history_df=None, path=ARTIFACT):
    """new_df: pre-match rows (outcome/scores may be null).
    history_df: settled fixtures used for rolling form (fetched from Supabase if None)."""
    bundle = load_bundle(path)
    alpha = bundle.get("blend_alpha")
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
    p_raw = bundle["models"]["1x2"].predict_proba(X)
    p = _blend_1x2(p_raw, feats, alpha)
    out["home_win_pct"], out["draw_pct"], out["away_win_pct"] = (p * 100).round(1).T
    out["pick_1x2"] = pd.Series(p.argmax(1)).map({0: "Home", 1: "Draw", 2: "Away"})
    for k in ("1.5", "2.5", "3.5"):
        tag = k.replace(".", "")
        po_raw = bundle["models"][f"ou_{k}"].predict_proba(X)[:, 1]
        po = _blend_binary(po_raw, feats, f"imp_over_{tag}", alpha)
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

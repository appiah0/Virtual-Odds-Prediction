#!/usr/bin/env python3
"""Compare pre-match XGBoost predictions (`live_predictions`) against
settled outcomes (`scraped_fixtures`) in Supabase.

Usage:
  python ml/accuracy_tracker.py                  # full report across all settled predictions
  python ml/accuracy_tracker.py --last-rounds 20 # report on the most recent 20 settled rounds
  python ml/accuracy_tracker.py --push-summary   # also upsert round summaries to `prediction_accuracy`
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from features import get_client  # noqa: E402

PREDICTIONS_TABLE = "live_predictions"
FIXTURES_TABLE = "scraped_fixtures"
SUMMARY_TABLE = "prediction_accuracy"
OUTCOME_TO_PICK = {"H": "Home", "D": "Draw", "A": "Away"}


def _ensure_env_key():
    if not os.environ.get("SUPABASE_SERVICE_ROLE_KEY") and os.environ.get("SUPABASE_KEY"):
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = os.environ["SUPABASE_KEY"]


def fetch_all_rows(client, table: str, settled_only: bool = False, page_size: int = 1000) -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        query = client.table(table).select("*")
        if settled_only:
            query = query.not_.is_("outcome", "null")
        batch = (
            query.order("round_no", desc=False)
            .order("fixture_key", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict:
    out = {"matches": int(len(df))}
    if df.empty:
        return out

    # --- 1X2 metrics ---
    actual_pick = df["outcome"].map(OUTCOME_TO_PICK)
    valid_1x2 = df["pick_1x2"].notna() & actual_pick.notna()
    if valid_1x2.any():
        sub = df[valid_1x2]
        out["acc_1x2"] = float((sub["pick_1x2"] == actual_pick[valid_1x2]).mean())

        prob_cols = ["home_win_pct", "draw_pct", "away_win_pct"]
        if all(c in sub.columns for c in prob_cols):
            probs = sub[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) / 100.0
            y_idx = sub["outcome"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
            valid_p = ~np.isnan(probs).any(axis=1) & ~np.isnan(y_idx)
            if valid_p.any():
                p_clean = np.clip(probs[valid_p], 1e-6, 1.0 - 1e-6)
                p_clean = p_clean / p_clean.sum(axis=1, keepdims=True)
                y_clean = y_idx[valid_p].astype(int)
                out["logloss_1x2"] = float(log_loss(y_clean, p_clean, labels=[0, 1, 2]))
                out["avg_conf_actual_1x2"] = float(p_clean[np.arange(len(y_clean)), y_clean].mean())

    # --- Over / Under metrics (1.5, 2.5, 3.5) ---
    total_goals = pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(
        df["away_score"], errors="coerce"
    )
    for line, thr in [("15", 2.0), ("25", 3.0), ("35", 4.0)]:
        over_col = f"over_{line}_pct"
        if over_col not in df.columns:
            continue
        p_over = pd.to_numeric(df[over_col], errors="coerce") / 100.0
        y_over = (total_goals >= thr).astype(float)
        mask = total_goals.notna() & p_over.notna()
        if mask.any():
            y_bin = y_over[mask].to_numpy(dtype=int)
            p_bin = np.clip(p_over[mask].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
            out[f"acc_ou_{line}"] = float(((p_bin >= 0.5).astype(int) == y_bin).mean())
            out[f"brier_ou_{line}"] = float(brier_score_loss(y_bin, p_bin))

    return out


def build_joined_settled(client) -> pd.DataFrame:
    preds = fetch_all_rows(client, PREDICTIONS_TABLE, settled_only=False)
    fixtures = fetch_all_rows(client, FIXTURES_TABLE, settled_only=True)
    if preds.empty or fixtures.empty:
        return pd.DataFrame()

    fixtures_slim = fixtures[
        [c for c in ("fixture_key", "home_score", "away_score", "outcome") if c in fixtures.columns]
    ].drop_duplicates(subset=["fixture_key"])

    merged = preds.merge(fixtures_slim, on="fixture_key", how="inner", suffixes=("", "_settled"))
    merged["round_no"] = pd.to_numeric(merged["round_no"], errors="coerce")
    merged = merged[merged["outcome"].isin(["H", "D", "A"])].copy()
    return merged.sort_values(["round_no", "league", "fixture_key"]).reset_index(drop=True)


def print_block(title: str, m: dict):
    if m.get("matches", 0) == 0:
        print(f"\n=== {title}: no settled predictions yet ===")
        return
    print(f"\n=== {title} (n={m['matches']} settled fixtures) ===")
    if "acc_1x2" in m:
        ll = f" | LogLoss {m['logloss_1x2']:.4f}" if "logloss_1x2" in m else ""
        conf = f" | Avg P(Actual) {m['avg_conf_actual_1x2']:.1%}" if "avg_conf_actual_1x2" in m else ""
        print(f"  1X2 Pick Acc : {m['acc_1x2']:.2%}{ll}{conf}")
    for line, label in [("15", "O/U 1.5"), ("25", "O/U 2.5"), ("35", "O/U 3.5")]:
        if f"acc_ou_{line}" in m:
            print(f"  {label:<9}  : Acc {m[f'acc_ou_{line}']:.2%} | Brier {m[f'brier_ou_{line}']:.4f}")


def build_round_summaries(merged: pd.DataFrame) -> list[dict]:
    records = []
    now_str = datetime.now(timezone.utc).isoformat()
    for (r_no, lg), grp in merged.groupby(["round_no", "league"], dropna=True):
        m = compute_metrics(grp)
        records.append({
            "summary_key": f"{int(r_no)}_{lg}",
            "round_no": int(r_no),
            "league": str(lg),
            "settled_fixtures": m.get("matches", 0),
            "acc_1x2": m.get("acc_1x2"),
            "logloss_1x2": m.get("logloss_1x2"),
            "acc_ou_15": m.get("acc_ou_15"),
            "brier_ou_15": m.get("brier_ou_15"),
            "acc_ou_25": m.get("acc_ou_25"),
            "brier_ou_25": m.get("brier_ou_25"),
            "acc_ou_35": m.get("acc_ou_35"),
            "brier_ou_35": m.get("brier_ou_35"),
            "updated_at": now_str,
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Track live XGBoost prediction accuracy.")
    parser.add_argument("--last-rounds", type=int, default=None, help="Filter to last N rounds")
    parser.add_argument("--push-summary", action="store_true", help="Upsert round stats to `prediction_accuracy`")
    args = parser.parse_args()

    _ensure_env_key()
    client = get_client()
    merged = build_joined_settled(client)

    if merged.empty:
        print("No overlapping settled rows found between `live_predictions` and `scraped_fixtures` yet.")
        return

    if args.last_rounds is not None and args.last_rounds > 0:
        recent_rounds = sorted(merged["round_no"].dropna().unique())[-args.last_rounds :]
        merged = merged[merged["round_no"].isin(recent_rounds)].copy()

    rounds_covered = sorted(merged["round_no"].dropna().astype(int).unique())
    print(f"Rounds evaluated: {rounds_covered[0]} .. {rounds_covered[-1]} ({len(rounds_covered)} rounds)")

    print_block("Overall", compute_metrics(merged))

    for lg, grp in merged.groupby("league", dropna=True):
        print_block(f"League: {lg}", compute_metrics(grp))

    if args.push_summary:
        recs = build_round_summaries(merged)
        if recs:
            client.table(SUMMARY_TABLE).upsert(recs, on_conflict="summary_key").execute()
            print(f"\nUpserted {len(recs)} round-league summary rows into `{SUMMARY_TABLE}`.")


if __name__ == "__main__":
    main()

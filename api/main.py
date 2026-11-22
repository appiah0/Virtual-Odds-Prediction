import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from features import FEATURES, OU_LINES

MODEL_PATH = ML_DIR / "instant_football_xgb.joblib"
bundle = joblib.load(MODEL_PATH)
models = bundle["models"]
feat_cols = bundle.get("features", FEATURES)

app = FastAPI(title="Instant Football XGBoost API")


class OddsInput(BaseModel):
    home: float
    draw: float
    away: float


class OverUnderInput(BaseModel):
    line: Optional[float] = None
    over: Optional[float] = None
    under: Optional[float] = None


class MatchInput(BaseModel):
    league: Optional[str] = "Spain"
    home: str
    away: str
    home_stars: Optional[float] = 3.0
    away_stars: Optional[float] = 3.0
    odds: OddsInput
    over_under: Optional[OverUnderInput] = None
    home_profile: Optional[Dict[str, Any]] = None
    away_profile: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    matches: List[MatchInput]


def _pick_num(prof: Optional[Dict[str, Any]], keys: List[str]) -> float:
    if not isinstance(prof, dict):
        return np.nan
    for k in keys:
        v = prof.get(k)
        if v is not None and v != "":
            try:
                f = float(v)
                if not np.isnan(f):
                    return f
            except (TypeError, ValueError):
                pass
    return np.nan


def _build_rows(matches: List[MatchInput]) -> pd.DataFrame:
    rows = []
    for m in matches:
        lg = (m.league or "Spain").strip()
        hs = float(m.home_stars) if m.home_stars is not None else np.nan
        as_ = float(m.away_stars) if m.away_stars is not None else np.nan
        oh, od, oa = float(m.odds.home), float(m.odds.draw), float(m.odds.away)

        raw_h = 1.0 / oh if oh > 1.0 else np.nan
        raw_d = 1.0 / od if od > 1.0 else np.nan
        raw_a = 1.0 / oa if oa > 1.0 else np.nan
        tot = (
            np.nansum([raw_h, raw_d, raw_a])
            if not any(np.isnan([raw_h, raw_d, raw_a]))
            else np.nan
        )

        imp_h = (raw_h / tot) if (tot and not np.isnan(tot)) else np.nan
        imp_d = (raw_d / tot) if (tot and not np.isnan(tot)) else np.nan
        imp_a = (raw_a / tot) if (tot and not np.isnan(tot)) else np.nan

        row: Dict[str, Any] = {
            "league": lg,
            "league_spain": 1 if lg.lower() == "spain" else 0,
            "home_stars": hs,
            "away_stars": as_,
            "star_diff": hs - as_ if not (np.isnan(hs) or np.isnan(as_)) else np.nan,
            "star_sum": hs + as_ if not (np.isnan(hs) or np.isnan(as_)) else np.nan,
            "odds_home": oh,
            "odds_draw": od,
            "odds_away": oa,
            "vig_1x2": (tot - 1.0) if not np.isnan(tot) else np.nan,
            "imp_home": imp_h,
            "imp_draw": imp_d,
            "imp_away": imp_a,
            "imp_home_away_diff": (
                (imp_h - imp_a) if not (np.isnan(imp_h) or np.isnan(imp_a)) else np.nan
            ),
            "slope_15_25": np.nan,
            "slope_25_35": np.nan,
            "imp_gg": np.nan,
        }

        for l in OU_LINES:
            row[f"imp_over_{l}"] = np.nan

        if m.over_under and m.over_under.over and m.over_under.under:
            oo, ou = float(m.over_under.over), float(m.over_under.under)
            if oo > 1.0 and ou > 1.0:
                ro, ru = 1.0 / oo, 1.0 / ou
                p_over = ro / (ro + ru)
                line_str = str(m.over_under.line or "")
                tag_map = {
                    "0.5": "05",
                    "1.5": "15",
                    "2.5": "25",
                    "3.5": "35",
                    "4.5": "45",
                    "5.5": "55",
                }
                tag = tag_map.get(line_str)
                if tag:
                    row[f"imp_over_{tag}"] = p_over

        for side, prof in (("home", m.home_profile), ("away", m.away_profile)):
            row[f"{side}_roll5_gf"] = _pick_num(
                prof, [f"{side}_roll5_gf", f"{side}_gf_ma5", "roll5_gf", "gf_ma5", "avg_gf"]
            )
            row[f"{side}_roll5_ga"] = _pick_num(
                prof, [f"{side}_roll5_ga", f"{side}_ga_ma5", "roll5_ga", "ga_ma5", "avg_ga"]
            )
            row[f"{side}_roll5_pts"] = _pick_num(
                prof, [f"{side}_roll5_pts", f"{side}_pts_ma5", "roll5_pts", "pts_ma5", "avg_pts"]
            )

        row["roll5_gf_diff"] = row["home_roll5_gf"] - row["away_roll5_gf"]
        row["roll5_ga_diff"] = row["home_roll5_ga"] - row["away_roll5_ga"]
        row["roll5_pts_diff"] = row["home_roll5_pts"] - row["away_roll5_pts"]
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in feat_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[feat_cols]


@app.get("/health")
def health():
    return {"status": "ok", "features": len(feat_cols)}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.matches:
        raise HTTPException(status_code=400, detail="No matches provided")
    X = _build_rows(req.matches)
    probs_1x2 = models["1x2"].predict_proba(X)
    pick_map = {0: "1", 1: "X", 2: "2"}

    ou_15 = models["ou_1.5"].predict_proba(X)[:, 1] * 100.0
    ou_25 = models["ou_2.5"].predict_proba(X)[:, 1] * 100.0
    ou_35 = models["ou_3.5"].predict_proba(X)[:, 1] * 100.0

    results = []
    for i in range(len(req.matches)):
        p = probs_1x2[i] * 100.0
        best_idx = int(np.argmax(p))
        results.append({
            "probs": [round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)],
            "pick": pick_map[best_idx],
            "confidence": round(float(p[best_idx]), 1),
            "over_15_pct": round(float(ou_15[i]), 1),
            "under_15_pct": round(float(100.0 - ou_15[i]), 1),
            "over_25_pct": round(float(ou_25[i]), 1),
            "under_25_pct": round(float(100.0 - ou_25[i]), 1),
            "over_35_pct": round(float(ou_35[i]), 1),
            "under_35_pct": round(float(100.0 - ou_35[i]), 1),
        })
    return {"predictions": results}

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
from pathlib import Path

_API_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _API_DIR.parent
for _candidate in (_REPO_ROOT, _REPO_ROOT / "ml", _API_DIR):
    _path_str = str(_candidate)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from features import FEATURES, OU_TARGETS, build_features

ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "instant_football_xgb.joblib")

app = FastAPI(title="Virtual Odds XGBoost Inference API")


class MatchInput(BaseModel):
    league: Optional[str] = "Spain"
    home: str
    away: str
    home_stars: Optional[float] = 3.0
    away_stars: Optional[float] = 3.0
    odds: Dict[str, float]
    over_under: Optional[Dict[str, Any]] = None
    home_profile: Optional[Dict[str, Any]] = None
    away_profile: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    matches: List[MatchInput]


def _log_odds_margin(probs_3col: np.ndarray) -> np.ndarray:
    p = np.asarray(probs_3col, dtype=float).copy()
    bad = ~np.isfinite(p) | (p <= 1e-6)
    p[bad] = 1.0 / 3.0
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(np.clip(p, 1e-6, 1.0))


def _blend_1x2(p_xgb: np.ndarray, p_book: np.ndarray, alpha: float) -> np.ndarray:
    out = p_xgb.copy()
    valid = ~np.isnan(p_book).any(axis=1)
    if np.any(valid):
        blended = alpha * p_xgb[valid] + (1.0 - alpha) * p_book[valid]
        row_sum = blended.sum(axis=1, keepdims=True)
        out[valid] = np.where(row_sum > 0, blended / row_sum, p_xgb[valid])
    return out


def _blend_binary(p_xgb: np.ndarray, p_book: np.ndarray, alpha: float) -> np.ndarray:
    out = p_xgb.copy()
    valid = ~np.isnan(p_book)
    if np.any(valid):
        out[valid] = alpha * p_xgb[valid] + (1.0 - alpha) * p_book[valid]
    return np.clip(out, 0.0, 1.0)


def _load_bundle():
    if not os.path.exists(ARTIFACT_PATH):
        raise RuntimeError(f"Model artifact not found at {ARTIFACT_PATH}")
    bundle = joblib.load(ARTIFACT_PATH)
    models = bundle.get("models", {})
    x1x2_feats = bundle.get("1x2_features") or bundle.get("features") or list(FEATURES)
    ou_feats = bundle.get("ou_features") or x1x2_feats
    blend_alphas = bundle.get("blend_alphas") or {
        "1x2": bundle.get("blend_alpha", 0.25),
        "ou_1.5": 0.35,
        "ou_2.5": 0.35,
        "ou_3.5": 0.35,
    }
    draw_mult = float(bundle.get("draw_mult", 0.85))
    return models, x1x2_feats, ou_feats, blend_alphas, draw_mult


MODELS, X1X2_FEATS, OU_FEATS, BLEND_ALPHAS, DRAW_MULT = _load_bundle()


def _matches_to_feature_df(matches: List[MatchInput]) -> pd.DataFrame:
    rows = []
    for idx, m in enumerate(matches):
        hp = m.home_profile or {}
        ap = m.away_profile or {}
        ou = m.over_under or {}
        line_key = str(ou.get("line", "2.5")).replace(".", "")

        row: Dict[str, Any] = {
            "id": idx,
            "round_no": int(hp.get("round_no") or ap.get("round_no") or 9999),
            "league": m.league or "Spain",
            "home_team": m.home,
            "away_team": m.away,
            "home_stars": m.home_stars if m.home_stars is not None else 3.0,
            "away_stars": m.away_stars if m.away_stars is not None else 3.0,
            "home_score": np.nan,
            "away_score": np.nan,
            "odds_home": m.odds.get("home"),
            "odds_draw": m.odds.get("draw"),
            "odds_away": m.odds.get("away"),
            "home_roll5_gf": hp.get("home_gf_ma5", hp.get("gf_ma5", np.nan)),
            "home_roll5_ga": hp.get("home_ga_ma5", hp.get("ga_ma5", np.nan)),
            "home_roll5_pts": hp.get("pts_ma5", np.nan),
            "home_roll5_gd": hp.get("gd_ma5", np.nan),
            "away_roll5_gf": ap.get("away_gf_ma5", ap.get("gf_ma5", np.nan)),
            "away_roll5_ga": ap.get("away_ga_ma5", ap.get("ga_ma5", np.nan)),
            "away_roll5_pts": ap.get("pts_ma5", np.nan),
            "away_roll5_gd": ap.get("gd_ma5", np.nan),
        }
        if line_key in ("05", "15", "25", "35", "45", "55"):
            row[f"odds_over_{line_key}"] = ou.get("over")
            row[f"odds_under_{line_key}"] = ou.get("under")
        rows.append(row)

    raw_df = pd.DataFrame(rows)
    feat_df = build_features(raw_df)
    return feat_df.sort_values("id").reset_index(drop=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifact": ARTIFACT_PATH,
        "blend_alphas": BLEND_ALPHAS,
        "draw_mult": DRAW_MULT,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.matches:
        raise HTTPException(status_code=400, detail="No matches provided.")

    m1x2 = MODELS.get("1x2")
    if m1x2 is None:
        raise HTTPException(status_code=500, detail="1X2 model missing from artifact.")

    df = _matches_to_feature_df(req.matches)
    for col in set(X1X2_FEATS) | set(OU_FEATS):
        if col not in df.columns:
            df[col] = np.nan

    imp_1x2 = df[["imp_home", "imp_draw", "imp_away"]].to_numpy(dtype=float)
    margin = _log_odds_margin(imp_1x2)

    X_1x2 = df[X1X2_FEATS]
    try:
        p_xgb = m1x2.predict_proba(X_1x2, base_margin=margin)
    except TypeError:
        p_xgb = m1x2.predict_proba(X_1x2)

    a_1x2 = float(BLEND_ALPHAS.get("1x2", 0.25))
    p_final = _blend_1x2(p_xgb, imp_1x2, a_1x2)

    adj_picks = p_final.copy()
    adj_picks[:, 1] = adj_picks[:, 1] * DRAW_MULT
    pick_indices = adj_picks.argmax(axis=1)
    pick_labels = ["1", "X", "2"]

    ou_probs: Dict[str, Optional[np.ndarray]] = {}
    for k in OU_TARGETS:
        tag = k.replace(".", "")
        key = f"ou_{k}"
        m_ou = MODELS.get(key)
        if m_ou is not None:
            po_xgb = m_ou.predict_proba(df[OU_FEATS])[:, 1]
            imp_ou = df[f"imp_over_{tag}"].to_numpy(dtype=float) if f"imp_over_{tag}" in df.columns else np.full(len(df), np.nan)
            a_ou = float(BLEND_ALPHAS.get(key, 0.35))
            ou_probs[k] = _blend_binary(po_xgb, imp_ou, a_ou) * 100.0
        else:
            ou_probs[k] = None

    predictions = []
    for i in range(len(df)):
        h_pct = float(p_final[i, 0] * 100.0)
        d_pct = float(p_final[i, 1] * 100.0)
        a_pct = float(p_final[i, 2] * 100.0)
        pick_code = pick_labels[int(pick_indices[i])]
        conf = max(h_pct, d_pct, a_pct)

        item: Dict[str, Any] = {
            "home_win_pct": round(h_pct, 2),
            "draw_pct": round(d_pct, 2),
            "away_win_pct": round(a_pct, 2),
            "probs": [round(h_pct, 2), round(d_pct, 2), round(a_pct, 2)],
            "pick_1x2": pick_code,
            "confidence": round(conf, 2),
        }

        for k in ("1.5", "2.5", "3.5"):
            arr = ou_probs.get(k)
            if arr is not None and np.isfinite(arr[i]):
                over_v = float(np.clip(arr[i], 0.0, 100.0))
                under_v = float(np.clip(100.0 - over_v, 0.0, 100.0))
                tag = k.replace(".", "_")
                item[f"over_{tag}_pct"] = round(over_v, 2)
                item[f"under_{tag}_pct"] = round(under_v, 2)

        predictions.append(item)

    return {"predictions": predictions}

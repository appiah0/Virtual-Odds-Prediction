"""Train + evaluate XGBoost 1X2 and O/U models on Supabase data."""
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from features import FEATURES, OU_TARGETS, add_targets, build_features, fetch_fixtures, get_client

ARTIFACT = "instant_football_xgb.joblib"
SEED = 42
MIN_EDGE = 0.04
DRAW_MULT = 0.85  # dampens draw probability slightly when comparing 1X2 argmax picks

BLEND_ALPHAS = {
    "1x2": 0.25,
    "ou_1.5": 0.35,
    "ou_2.5": 0.35,
    "ou_3.5": 0.35,
}

PREFERRED_1X2 = [
    "league_spain",
    "home_stars",
    "away_stars",
    "star_diff",
    "star_share_home",
    "star_vs_imp_home",
    "star_vs_imp_away",
    "odds_home",
    "odds_draw",
    "odds_away",
    "imp_home",
    "imp_draw",
    "imp_away",
    "vig_1x2",
    "imp_home_away_diff",
    "imp_fav_max",
    "imp_fav_underdog_gap",
    "imp_fav_vs_draw",
    "imp_over_15",
    "imp_over_25",
    "imp_over_35",
    "slope_15_25",
    "slope_25_35",
    "imp_gg",
    "home_roll5_gf",
    "home_roll5_ga",
    "home_roll5_pts",
    "away_roll5_gf",
    "away_roll5_ga",
    "away_roll5_pts",
    "roll5_pts_diff",
    "roll5_net_gd_diff",
]

X1X2_FEATURES = [c for c in PREFERRED_1X2 if c in FEATURES] or list(FEATURES)
OU_FEATURES = [c for c in FEATURES if c in PREFERRED_1X2] or list(FEATURES)

BASE_1X2 = dict(
    n_estimators=180,
    max_depth=2,
    learning_rate=0.02,
    subsample=0.75,
    colsample_bytree=0.7,
    min_child_weight=25,
    reg_alpha=2.0,
    reg_lambda=10.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

BASE_OU = dict(
    n_estimators=220,
    max_depth=2,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.65,
    min_child_weight=20,
    reg_alpha=2.0,
    reg_lambda=10.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

BASE = BASE_1X2

def _log_odds_margin(probs_3col):
    p = np.asarray(probs_3col, dtype=float).copy()
    bad = ~np.isfinite(p) | (p <= 1e-6)
    p[bad] = 1.0 / 3.0
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(np.clip(p, 1e-6, 1.0))


def _weighted_argmax_1x2(probs, draw_mult=1.0):
    adj = np.asarray(probs, dtype=float).copy()
    adj[:, 1] = adj[:, 1] * draw_mult
    return adj.argmax(axis=1)


def calibrated(estimator):
    tscv = TimeSeriesSplit(n_splits=5)
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=tscv)


def _blend_1x2(p_xgb, p_book, alpha):
    p_xgb = np.asarray(p_xgb, dtype=float)
    p_book = np.asarray(p_book, dtype=float)
    out = p_xgb.copy()
    valid = ~np.isnan(p_book).any(axis=1)
    if np.any(valid):
        blended = alpha * p_xgb[valid] + (1.0 - alpha) * p_book[valid]
        row_sum = blended.sum(axis=1, keepdims=True)
        out[valid] = np.where(row_sum > 0, blended / row_sum, p_xgb[valid])
    return out


def _blend_binary(p_xgb, p_book, alpha):
    p_xgb = np.asarray(p_xgb, dtype=float)
    p_book = np.asarray(p_book, dtype=float)
    out = p_xgb.copy()
    valid = ~np.isnan(p_book)
    if np.any(valid):
        out[valid] = alpha * p_xgb[valid] + (1.0 - alpha) * p_book[valid]
    return np.clip(out, 0.0, 1.0)


def main():
    raw = fetch_fixtures(get_client(), settled=True)
    print(f"Fetched {len(raw)} settled rows, {raw['round_no'].nunique()} rounds")
    df = add_targets(build_features(raw))
    df = df[df["y_1x2"].notna()].reset_index(drop=True)
    print(f"1X2 features: {len(X1X2_FEATURES)}/{len(FEATURES)} | draw_mult={DRAW_MULT}")

    rounds = np.sort(df["round_no"].unique())
    cut = rounds[int(len(rounds) * 0.8)]
    tr, te = df[df.round_no < cut], df[df.round_no >= cut]
    print(f"Train rounds < {cut}: {len(tr)} rows | Test rounds >= {cut}: {len(te)} rows\n")

    models, metrics = {}, {}

    # ---- Model 1: Raw vs Calibrated 1X2 with log-odds base_margin ----
    a_1x2 = BLEND_ALPHAS["1x2"]
    margin_tr = _log_odds_margin(tr[["imp_home", "imp_draw", "imp_away"]].values)
    margin_te = _log_odds_margin(te[["imp_home", "imp_draw", "imp_away"]].values)

    raw_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        **BASE_1X2,
    )
    raw_xgb.fit(
        tr[X1X2_FEATURES],
        tr["y_1x2"].astype(int).values,
        base_margin=margin_tr,
    )
    p_raw = raw_xgb.predict_proba(te[X1X2_FEATURES], base_margin=margin_te)

    cal_xgb = calibrated(
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **BASE_1X2,
        )
    )
    try:
        cal_xgb.fit(tr[X1X2_FEATURES], tr["y_1x2"].astype(int).values, base_margin=margin_tr)
        p_cal = cal_xgb.predict_proba(te[X1X2_FEATURES])
    except TypeError:
        cal_xgb.fit(tr[X1X2_FEATURES], tr["y_1x2"].astype(int).values)
        p_cal = cal_xgb.predict_proba(te[X1X2_FEATURES])

    y = te["y_1x2"].astype(int).values
    b = te[["imp_home", "imp_draw", "imp_away"]].values
    ok = ~np.isnan(b).any(axis=1)

    p_blend = _blend_1x2(p_raw, b, a_1x2)
    raw_pick = p_raw[ok].argmax(1)
    raw_adj_pick = _weighted_argmax_1x2(p_raw[ok], DRAW_MULT)
    cal_pick = p_cal[ok].argmax(1)
    blend_pick = p_blend[ok].argmax(1)
    book_pick = b[ok].argmax(1)

    disagree = raw_pick != book_pick
    edge_1x2 = np.max(p_blend[ok] - b[ok], axis=1)
    high_edge = edge_1x2 >= MIN_EDGE

    metrics["1x2"] = {
        "blend_alpha": a_1x2,
        "draw_mult": DRAW_MULT,
        "xgb_raw_logloss": float(log_loss(y[ok], p_raw[ok], labels=[0, 1, 2])),
        "xgb_cal_logloss": float(log_loss(y[ok], p_cal[ok], labels=[0, 1, 2])),
        "book_logloss": float(log_loss(y[ok], b[ok], labels=[0, 1, 2])),
        "xgb_acc": float(accuracy_score(y[ok], raw_pick)),
        "xgb_adj_acc": float(accuracy_score(y[ok], raw_adj_pick)),
        "xgb_cal_acc": float(accuracy_score(y[ok], cal_pick)),
        "blend_acc": float(accuracy_score(y[ok], blend_pick)),
        "book_acc": float(accuracy_score(y[ok], book_pick)),
        "disagree_n": int(disagree.sum()),
        "disagree_xgb_acc": float(accuracy_score(y[ok][disagree], raw_pick[disagree])) if disagree.any() else None,
        "disagree_book_acc": float(accuracy_score(y[ok][disagree], book_pick[disagree])) if disagree.any() else None,
        "high_edge_n": int(high_edge.sum()),
        "high_edge_blend_acc": float(accuracy_score(y[ok][high_edge], blend_pick[high_edge])) if high_edge.any() else None,
    }
    models["1x2"] = raw_xgb
    models["1x2_cal"] = cal_xgb
    r = metrics["1x2"]
    print("=== 1X2 (out-of-sample) ===")
    print(f"Log-loss  XGB-Raw {r['xgb_raw_logloss']:.4f} | XGB-Cal {r['xgb_cal_logloss']:.4f} | Bookmaker {r['book_logloss']:.4f}")
    print(
        f"Top-pick acc  XGB-Raw {r['xgb_acc']:.3%} | XGB-DrawAdj({DRAW_MULT}) {r['xgb_adj_acc']:.3%} | "
        f"XGB-Cal {r['xgb_cal_acc']:.3%} | Blend({a_1x2:.2f}) {r['blend_acc']:.3%} | Bookmaker {r['book_acc']:.3%}"
    )
    if r["disagree_n"] > 0:
        print(f"Disagreements (n={r['disagree_n']}): XGB-Raw {r['disagree_xgb_acc']:.3%} | Bookmaker {r['disagree_book_acc']:.3%}")
    if r["high_edge_n"] > 0:
        print(f"Edge >= {MIN_EDGE:.0%} (n={r['high_edge_n']}): Blend acc {r['high_edge_blend_acc']:.3%}\n")
    else:
        print()

    # ---- Model 2: Over/Under ----
    for k in OU_TARGETS:
        tag = k.replace(".", "")
        key = f"ou_{k}"
        a_ou = BLEND_ALPHAS.get(key, 0.35)
        target_col = f"y_over_{k}"
        tr_sub = tr[tr[target_col].notna()]
        te_sub = te[te[target_col].notna()]

        m = calibrated(XGBClassifier(objective="binary:logistic", eval_metric="logloss", **BASE_OU))
        m.fit(tr_sub[OU_FEATURES], tr_sub[target_col].astype(int))

        po = m.predict_proba(te_sub[OU_FEATURES])[:, 1]
        yo = te_sub[target_col].astype(int).values
        bo = te_sub[f"imp_over_{tag}"].values
        po_blend = _blend_binary(po, bo, a_ou)
        ok_ou = ~np.isnan(bo)

        metrics[key] = {
            "blend_alpha": a_ou,
            "xgb_brier": float(brier_score_loss(yo[ok_ou], po[ok_ou])),
            "blend_brier": float(brier_score_loss(yo[ok_ou], po_blend[ok_ou])),
            "book_brier": float(brier_score_loss(yo[ok_ou], bo[ok_ou])),
            "xgb_acc": float(accuracy_score(yo[ok_ou], po[ok_ou] > 0.5)),
            "blend_acc": float(accuracy_score(yo[ok_ou], po_blend[ok_ou] > 0.5)),
            "book_acc": float(accuracy_score(yo[ok_ou], bo[ok_ou] > 0.5)),
        }
        models[key] = m
        r = metrics[key]
        print(f"=== O/U {k} (n={ok_ou.sum()}, alpha={a_ou:.2f}) ===")
        print(f"Brier  XGB {r['xgb_brier']:.4f} | Blend {r['blend_brier']:.4f} | Bookmaker {r['book_brier']:.4f}")
        print(f"Top-pick accuracy  XGB {r['xgb_acc']:.3%} | Blend {r['blend_acc']:.3%} | Bookmaker {r['book_acc']:.3%}\n")

    joblib.dump(
        {
            "models": models,
            "features": X1X2_FEATURES,
            "1x2_features": X1X2_FEATURES,
            "ou_features": OU_FEATURES,
            "blend_alpha": a_1x2,
            "draw_mult": DRAW_MULT,
            "blend_alphas": BLEND_ALPHAS,
            "metrics": metrics,
            "test_start_round": int(cut),
        },
        ARTIFACT,
    )
    print(f"Saved -> {ARTIFACT}")


if __name__ == "__main__":
    main()

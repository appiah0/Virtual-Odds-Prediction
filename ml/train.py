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
MIN_EDGE = 0.04  # 4 percentage points above bookmaker implied probability

BLEND_ALPHAS = {
    "1x2": 0.25,
    "ou_1.5": 0.35,
    "ou_2.5": 0.35,
    "ou_3.5": 0.35,
}

PREFERRED_LEAN = [
    "league_spain",
    "home_stars",
    "away_stars",
    "star_diff",
    "odds_home",
    "odds_draw",
    "odds_away",
    "imp_home",
    "imp_draw",
    "imp_away",
    "vig_1x2",
    "imp_home_away_diff",
    "imp_over_05",
    "imp_over_15",
    "imp_over_25",
    "imp_over_35",
    "slope_15_25",
    "slope_25_35",
    "imp_gg",
    "home_roll5_gf",
    "home_roll5_ga",
    "away_roll5_gf",
    "away_roll5_ga",
    "roll5_pts_diff",
]

LEAN_FEATURES = [c for c in PREFERRED_LEAN if c in FEATURES] or list(FEATURES)

BASE = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)


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
    print(f"Using {len(LEAN_FEATURES)}/{len(FEATURES)} lean features | alphas={BLEND_ALPHAS}")

    rounds = np.sort(df["round_no"].unique())
    cut = rounds[int(len(rounds) * 0.8)]
    tr, te = df[df.round_no < cut], df[df.round_no >= cut]
    print(f"Train rounds < {cut}: {len(tr)} rows | Test rounds >= {cut}: {len(te)} rows\n")

    models, metrics = {}, {}

    # ---- Model 1: 1X2 ----
    a_1x2 = BLEND_ALPHAS["1x2"]
    m1 = calibrated(
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **BASE,
        )
    )
    m1.fit(tr[LEAN_FEATURES], tr["y_1x2"].astype(int))

    p = m1.predict_proba(te[LEAN_FEATURES])
    y = te["y_1x2"].astype(int).values
    b = te[["imp_home", "imp_draw", "imp_away"]].values
    p_blend = _blend_1x2(p, b, a_1x2)
    ok = ~np.isnan(b).any(axis=1)

    xgb_pick = p[ok].argmax(1)
    blend_pick = p_blend[ok].argmax(1)
    book_pick = b[ok].argmax(1)
    disagree = blend_pick != book_pick

    edge_1x2 = np.max(p_blend[ok] - b[ok], axis=1)
    high_edge = edge_1x2 >= MIN_EDGE

    metrics["1x2"] = {
        "blend_alpha": a_1x2,
        "xgb_logloss": float(log_loss(y[ok], p[ok], labels=[0, 1, 2])),
        "blend_logloss": float(log_loss(y[ok], p_blend[ok], labels=[0, 1, 2])),
        "book_logloss": float(log_loss(y[ok], b[ok], labels=[0, 1, 2])),
        "xgb_acc": float(accuracy_score(y[ok], xgb_pick)),
        "blend_acc": float(accuracy_score(y[ok], blend_pick)),
        "book_acc": float(accuracy_score(y[ok], book_pick)),
        "disagree_n": int(disagree.sum()),
        "disagree_blend_acc": float(accuracy_score(y[ok][disagree], blend_pick[disagree])) if disagree.any() else None,
        "disagree_book_acc": float(accuracy_score(y[ok][disagree], book_pick[disagree])) if disagree.any() else None,
        "high_edge_n": int(high_edge.sum()),
        "high_edge_blend_acc": float(accuracy_score(y[ok][high_edge], blend_pick[high_edge])) if high_edge.any() else None,
    }
    models["1x2"] = m1
    r = metrics["1x2"]
    print(f"=== 1X2 (out-of-sample, alpha={a_1x2:.2f}) ===")
    print(f"Log-loss  XGB {r['xgb_logloss']:.4f} | Blend {r['blend_logloss']:.4f} | Bookmaker {r['book_logloss']:.4f}")
    print(f"Top-pick accuracy  XGB {r['xgb_acc']:.3%} | Blend {r['blend_acc']:.3%} | Bookmaker {r['book_acc']:.3%}")
    if r["disagree_n"] > 0:
        print(
            f"Disagreements (n={r['disagree_n']}): "
            f"Blend {r['disagree_blend_acc']:.3%} | Bookmaker {r['disagree_book_acc']:.3%}"
        )
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

        m = calibrated(XGBClassifier(objective="binary:logistic", eval_metric="logloss", **BASE))
        m.fit(tr_sub[LEAN_FEATURES], tr_sub[target_col].astype(int))

        po = m.predict_proba(te_sub[LEAN_FEATURES])[:, 1]
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
            "features": LEAN_FEATURES,
            "blend_alpha": a_1x2,
            "blend_alphas": BLEND_ALPHAS,
            "metrics": metrics,
            "test_start_round": int(cut),
        },
        ARTIFACT,
    )
    print(f"Saved -> {ARTIFACT}")


if __name__ == "__main__":
    main()

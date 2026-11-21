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


def main():
    raw = fetch_fixtures(get_client(), settled=True)
    print(f"Fetched {len(raw)} settled rows, {raw['round_no'].nunique()} rounds")
    df = add_targets(build_features(raw))
    df = df[df["y_1x2"].notna()].reset_index(drop=True)

    # Strict chronological split on round_no (first 80% of rounds = train)
    rounds = np.sort(df["round_no"].unique())
    cut = rounds[int(len(rounds) * 0.8)]
    tr, te = df[df.round_no < cut], df[df.round_no >= cut]
    print(f"Train rounds < {cut}: {len(tr)} rows | Test rounds >= {cut}: {len(te)} rows\n")

    models, metrics = {}, {}

    # ---- Model 1: 1X2 ----
    Xtr_1x2 = tr[FEATURES]
    ytr_1x2 = tr["y_1x2"].astype(int)
    m1 = calibrated(XGBClassifier(objective="multi:softprob", num_class=3,
                                  eval_metric="mlogloss", **BASE))
    m1.fit(Xtr_1x2, ytr_1x2)

    p = m1.predict_proba(te[FEATURES])
    y = te["y_1x2"].astype(int).values
    b = te[["imp_home", "imp_draw", "imp_away"]].values
    ok = ~np.isnan(b).any(axis=1)
    metrics["1x2"] = {
        "xgb_logloss": float(log_loss(y[ok], p[ok], labels=[0, 1, 2])),
        "book_logloss": float(log_loss(y[ok], b[ok], labels=[0, 1, 2])),
        "xgb_acc": float(accuracy_score(y[ok], p[ok].argmax(1))),
        "book_acc": float(accuracy_score(y[ok], b[ok].argmax(1))),
    }
    models["1x2"] = m1
    r = metrics["1x2"]
    print("=== 1X2 (out-of-sample) ===")
    print(f"Log-loss  XGB {r['xgb_logloss']:.4f} | Bookmaker {r['book_logloss']:.4f}")
    print(f"Top-pick accuracy  XGB {r['xgb_acc']:.3%} | Bookmaker {r['book_acc']:.3%}\n")

    # ---- Model 2: Over/Under ----
    for k in OU_TARGETS:
        tag = k.replace(".", "")
        target_col = f"y_over_{k}"
        tr_sub = tr[tr[target_col].notna()]
        te_sub = te[te[target_col].notna()]

        m = calibrated(XGBClassifier(objective="binary:logistic", eval_metric="logloss", **BASE))
        m.fit(tr_sub[FEATURES], tr_sub[target_col].astype(int))

        po = m.predict_proba(te_sub[FEATURES])[:, 1]
        yo = te_sub[target_col].astype(int).values
        bo = te_sub[f"imp_over_{tag}"].values
        ok = ~np.isnan(bo)
        metrics[f"ou_{k}"] = {
            "xgb_brier": float(brier_score_loss(yo[ok], po[ok])),
            "book_brier": float(brier_score_loss(yo[ok], bo[ok])),
            "xgb_acc": float(accuracy_score(yo[ok], po[ok] > 0.5)),
            "book_acc": float(accuracy_score(yo[ok], bo[ok] > 0.5)),
        }
        models[f"ou_{k}"] = m
        r = metrics[f"ou_{k}"]
        print(f"=== O/U {k} (n={ok.sum()}) ===")
        print(f"Brier  XGB {r['xgb_brier']:.4f} | Bookmaker {r['book_brier']:.4f}")
        print(f"Top-pick accuracy  XGB {r['xgb_acc']:.3%} | Bookmaker {r['book_acc']:.3%}\n")

    joblib.dump(
        {"models": models, "features": FEATURES, "metrics": metrics, "test_start_round": int(cut)},
        ARTIFACT,
    )
    print(f"Saved -> {ARTIFACT}")


if __name__ == "__main__":
    main()

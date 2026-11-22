"""Inspect out-of-sample divergence between XGBoost and bookmaker implied odds."""
import numpy as np
import pandas as pd

from features import OU_TARGETS, add_targets, build_features, fetch_fixtures, get_client
from train import BASE, calibrated


def main():
    raw = fetch_fixtures(get_client(), settled=True)
    df = add_targets(build_features(raw))
    df = df[df["y_1x2"].notna()].reset_index(drop=True)

    rounds = np.sort(df["round_no"].unique())
    cut = rounds[int(len(rounds) * 0.8)]
    tr, te = df[df.round_no < cut].copy(), df[df.round_no >= cut].copy()

    # Use feature columns available in training
    from features import FEATURES
    from xgboost import XGBClassifier

    m1 = calibrated(
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **BASE,
        )
    )
    m1.fit(tr[FEATURES], tr["y_1x2"].astype(int))
    p_1x2 = m1.predict_proba(te[FEATURES])

    te["xgb_h"] = p_1x2[:, 0]
    te["xgb_d"] = p_1x2[:, 1]
    te["xgb_a"] = p_1x2[:, 2]
    te["xgb_pick_1x2"] = pd.Series(p_1x2.argmax(1), index=te.index).map({0: "H", 1: "D", 2: "A"})
    te["book_pick_1x2"] = pd.Series(
        te[["imp_home", "imp_draw", "imp_away"]].values.argmax(1), index=te.index
    ).map({0: "H", 1: "D", 2: "A"})

    te["max_diff_1x2"] = np.maximum.reduce([
        (te["xgb_h"] - te["imp_home"]).abs(),
        (te["xgb_d"] - te["imp_draw"]).abs(),
        (te["xgb_a"] - te["imp_away"]).abs(),
    ])
    te["xgb_won_1x2"] = (te["xgb_pick_1x2"] == te["outcome"]).astype(int)
    te["book_won_1x2"] = (te["book_pick_1x2"] == te["outcome"]).astype(int)

    cols = [
        "round_no", "league", "home_team", "away_team", "outcome",
        "xgb_pick_1x2", "book_pick_1x2", "xgb_won_1x2", "book_won_1x2",
        "max_diff_1x2", "xgb_h", "imp_home", "xgb_d", "imp_draw", "xgb_a", "imp_away",
    ]
    print("=== Top 15 Largest 1X2 Probability Divergences (Out-of-Sample) ===")
    print(te.sort_values("max_diff_1x2", ascending=False)[cols].head(15).to_string(index=False))

    disagree = te[te["xgb_pick_1x2"] != te["book_pick_1x2"]]
    if len(disagree) > 0:
        print(
            f"\n1X2 Disagreements (n={len(disagree)}): "
            f"XGB acc={disagree['xgb_won_1x2'].mean():.3%} | "
            f"Book acc={disagree['book_won_1x2'].mean():.3%}"
        )

    for k in OU_TARGETS:
        tag = k.replace(".", "")
        target_col = f"y_over_{k}"
        imp_col = f"imp_over_{tag}"
        sub = te[te[target_col].notna() & te[imp_col].notna()].copy()
        tr_sub = tr[tr[target_col].notna()]
        m_ou = calibrated(
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", **BASE)
        )
        m_ou.fit(tr_sub[FEATURES], tr_sub[target_col].astype(int))
        p_over = m_ou.predict_proba(sub[FEATURES])[:, 1]
        sub["xgb_over"] = p_over
        sub["diff_ou"] = (sub["xgb_over"] - sub[imp_col]).abs()
        sub["xgb_pick_ou"] = np.where(sub["xgb_over"] > 0.5, "OVER", "UNDER")
        sub["book_pick_ou"] = np.where(sub[imp_col] > 0.5, "OVER", "UNDER")
        sub["actual_ou"] = np.where(sub[target_col] == 1.0, "OVER", "UNDER")
        sub["xgb_won_ou"] = (sub["xgb_pick_ou"] == sub["actual_ou"]).astype(int)
        sub["book_won_ou"] = (sub["book_pick_ou"] == sub["actual_ou"]).astype(int)

        ou_cols = [
            "round_no", "league", "home_team", "away_team", "actual_ou",
            "xgb_pick_ou", "book_pick_ou", "xgb_won_ou", "book_won_ou",
            "diff_ou", "xgb_over", imp_col,
        ]
        print(f"\n=== Top 10 Largest O/U {k} Divergences ===")
        print(sub.sort_values("diff_ou", ascending=False)[ou_cols].head(10).to_string(index=False))

    bins = [0.0, 0.45, 0.55, 0.65, 1.01]
    labels = ["<45%", "45-55%", "55-65%", "65%+"]

    te["xgb_conf_1x2"] = te[["xgb_h", "xgb_d", "xgb_a"]].max(axis=1)
    te["conf_bucket"] = pd.cut(te["xgb_conf_1x2"], bins=bins, labels=labels, right=False)
    print("\n=== 1X2 Accuracy by Model Confidence Bucket ===")
    print(
        te.groupby("conf_bucket", observed=False)
        .agg(n=("xgb_won_1x2", "count"), xgb_acc=("xgb_won_1x2", "mean"), book_acc=("book_won_1x2", "mean"))
        .to_string()
    )

    for k in OU_TARGETS:
        tag = k.replace(".", "")
        target_col = f"y_over_{k}"
        imp_col = f"imp_over_{tag}"
        sub = te[te[target_col].notna() & te[imp_col].notna()].copy()
        tr_sub = tr[tr[target_col].notna()]
        m_ou = calibrated(
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", **BASE)
        )
        m_ou.fit(tr_sub[FEATURES], tr_sub[target_col].astype(int))
        p_over = m_ou.predict_proba(sub[FEATURES])[:, 1]
        sub["ou_conf"] = np.maximum(p_over, 1.0 - p_over)
        sub["conf_bucket"] = pd.cut(sub["ou_conf"], bins=bins, labels=labels, right=False)
        sub["xgb_pick_ou"] = np.where(p_over > 0.5, "OVER", "UNDER")
        sub["book_pick_ou"] = np.where(sub[imp_col] > 0.5, "OVER", "UNDER")
        sub["actual_ou"] = np.where(sub[target_col] == 1.0, "OVER", "UNDER")
        sub["xgb_won_ou"] = (sub["xgb_pick_ou"] == sub["actual_ou"]).astype(int)
        sub["book_won_ou"] = (sub["book_pick_ou"] == sub["actual_ou"]).astype(int)

        print(f"\n=== O/U {k} Accuracy by Model Confidence Bucket ===")
        print(
            sub.groupby("conf_bucket", observed=False)
            .agg(n=("xgb_won_ou", "count"), xgb_acc=("xgb_won_ou", "mean"), book_acc=("book_won_ou", "mean"))
            .to_string()
        )

if __name__ == "__main__":
    main()

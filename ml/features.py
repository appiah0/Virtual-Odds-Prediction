"""Shared data access + leakage-free feature engineering for Instant Football."""
import os

import numpy as np
import pandas as pd

TABLE = "scraped_fixtures"
OU_LINES = ["05", "15", "25", "35", "45", "55"]

ODDS_COLS = (
    ["odds_home", "odds_draw", "odds_away"]
    + [f"odds_{s}_{l}" for l in OU_LINES for s in ("over", "under")]
    + ["odds_1x", "odds_12", "odds_2x", "odds_gg", "odds_ng"]
)
NUM_COLS = ["round_no", "home_stars", "away_stars", "home_score", "away_score"] + ODDS_COLS

FEATURES = (
    [
        "league_spain",
        "home_stars",
        "away_stars",
        "star_diff",
        "star_sum",
        "star_share_home",
        "star_odds_gap_home",
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
    ]
    + [f"imp_over_{l}" for l in OU_LINES]
    + ["slope_15_25", "slope_25_35", "imp_gg", "imp_1x", "imp_12", "imp_2x"]
    + [f"{side}_roll5_{m}" for side in ("home", "away") for m in ("gf", "ga", "pts", "gd")]
    + ["roll5_gf_diff", "roll5_ga_diff", "roll5_pts_diff", "roll5_gd_diff", "roll5_net_gd_diff"]
)

OU_TARGETS = {"1.5": 2, "2.5": 3, "3.5": 4}  # over if total goals >= threshold
LABEL_MAP = {"H": 0, "D": 1, "A": 2}


def get_client():
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


def fetch_fixtures(client, settled=True, page_size=1000, table=TABLE):
    """Paginate with .range() so we get past Supabase's 1,000-row cap."""
    rows, start = [], 0
    while True:
        q = client.table(table).select("*")
        q = q.not_.is_("outcome", "null") if settled else q.is_("outcome", "null")
        batch = q.order("id").range(start, start + page_size - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return pd.DataFrame(rows)


def _two_way_prob(a, b):
    """Vig-free P(first side) from a pair of decimal odds; NaN if either missing."""
    a = a.where(a > 1.0)
    b = b.where(b > 1.0)
    ra, rb = 1.0 / a, 1.0 / b
    return ra / (ra + rb)


def build_features(df):
    df = df.copy()
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    # Sort and reset index FIRST so row IDs align cleanly with df.index
    df = df.sort_values(["round_no", "id"], kind="mergesort").reset_index(drop=True)

    df["league_spain"] = (df["league"] == "Spain").astype(int)
    df["star_diff"] = df["home_stars"] - df["away_stars"]
    df["star_sum"] = df["home_stars"] + df["away_stars"]
    df["star_share_home"] = np.where(
        df["star_sum"] > 0, df["home_stars"] / df["star_sum"], np.nan
    )
    away_share = np.where(
        df["star_sum"] > 0, df["away_stars"] / df["star_sum"], np.nan
    )

    # 1X2 vig-free probabilities
    o = df[["odds_home", "odds_draw", "odds_away"]].where(lambda x: x > 1.0)
    raw = 1.0 / o
    tot = raw.sum(axis=1, min_count=3)
    df["vig_1x2"] = tot - 1.0
    df["imp_home"] = raw["odds_home"] / tot
    df["imp_draw"] = raw["odds_draw"] / tot
    df["imp_away"] = raw["odds_away"] / tot
    df["imp_home_away_diff"] = df["imp_home"] - df["imp_away"]
    df["star_odds_gap_home"] = df["star_share_home"] - df["imp_home"]
    df["star_vs_imp_home"] = df["star_share_home"] - df["imp_home"]
    df["star_vs_imp_away"] = away_share - df["imp_away"]

    imp_1x2_arr = df[["imp_home", "imp_draw", "imp_away"]].to_numpy()
    df["imp_fav_max"] = np.nanmax(imp_1x2_arr, axis=1)
    df["imp_fav_underdog_gap"] = np.nanmax(imp_1x2_arr, axis=1) - np.nanmin(imp_1x2_arr, axis=1)
    df["imp_fav_vs_draw"] = np.nanmax(imp_1x2_arr, axis=1) - df["imp_draw"]

    # Over/Under vig-free probabilities (NaN preserved for XGBoost)
    for l in OU_LINES:
        df[f"imp_over_{l}"] = _two_way_prob(df[f"odds_over_{l}"], df[f"odds_under_{l}"])
    df["slope_15_25"] = df["imp_over_15"] - df["imp_over_25"]
    df["slope_25_35"] = df["imp_over_25"] - df["imp_over_35"]
    df["imp_gg"] = _two_way_prob(df["odds_gg"], df["odds_ng"])

    # Double-chance implied probabilities from odds_1x, odds_12, odds_2x
    for dc_col, out_col in [("odds_1x", "imp_1x"), ("odds_12", "imp_12"), ("odds_2x", "imp_2x")]:
        dc_odds = df[dc_col].where(df[dc_col] > 1.0)
        df[out_col] = 1.0 / dc_odds

    # Rolling 5-round form per (league, team); shift(1) => current match never leaks
    def side(team_col, gf, ga, tag):
        return pd.DataFrame({
            "row": df.index,
            "league": df["league"],
            "team": df[team_col],
            "round_no": df["round_no"],
            "gf": df[gf],
            "ga": df[ga],
            "side": tag,
        })

    long = pd.concat(
        [
            side("home_team", "home_score", "away_score", "home"),
            side("away_team", "away_score", "home_score", "away"),
        ],
        ignore_index=True,
    )
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3.0, 1.0], default=0.0)
    long["gd"] = long.gf - long.ga
    long.loc[long.gf.isna() | long.ga.isna(), ["pts", "gd"]] = np.nan
    long = long.sort_values(["league", "team", "round_no", "row"], kind="mergesort")
    grp = long.groupby(["league", "team"], sort=False)
    for m in ("gf", "ga", "pts", "gd"):
        long[f"roll5_{m}"] = grp[m].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    for tag in ("home", "away"):
        sub = long[long.side == tag].set_index("row")
        for m in ("gf", "ga", "pts", "gd"):
            df[f"{tag}_roll5_{m}"] = sub[f"roll5_{m}"].reindex(df.index)
    for m in ("gf", "ga", "pts", "gd"):
        df[f"roll5_{m}_diff"] = df[f"home_roll5_{m}"] - df[f"away_roll5_{m}"]
    df["roll5_net_gd_diff"] = (
        (df["home_roll5_gf"] - df["home_roll5_ga"])
        - (df["away_roll5_gf"] - df["away_roll5_ga"])
    )
    return df


def add_targets(df):
    df = df.copy()
    df["y_1x2"] = df["outcome"].map(LABEL_MAP)
    total = df["home_score"] + df["away_score"]
    for k, thr in OU_TARGETS.items():
        df[f"y_over_{k}"] = np.where(total.isna(), np.nan, (total >= thr).astype(float))
    return df

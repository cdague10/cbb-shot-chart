import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

SHOT_TYPES = {"DunkShot", "JumpShot", "LayUpShot", "TipShot"}
RAW_COLUMNS = [
    "game_id",
    "clock",
    "text",
    "home_team",
    "away_team",
    "team",
    "play_type",
    "score",
    "miss",
    "coord_x",
    "coord_y",
]
OUTPUT_BASE_COLUMNS = [
    "game_id",
    "home_team",
    "away_team",
    "team",
    "player",
    "coord_x",
    "coord_y",
    "is_made",
    "is_three_point",
    "points",
]
FEATURE_COLUMNS = [
    "coord_x",
    "coord_y",
    "distance",
    "angle",
    "is_three_point_int",
    "player_fg_loo",
    "shooter_local_fg",
    "log_player_attempts",
    "log_local_attempts",
]


def _extract_player_from_text(text):
    text = str(text)
    made_or_miss = re.match(r"^(.+?)\s+(?:made?|makes?|missed?|misses?)\b", text, flags=re.IGNORECASE)
    if made_or_miss:
        return made_or_miss.group(1).strip()

    # ESPN-style rows that omit make/miss verbs.
    verbless = re.match(
        r"^(.+?)\s+(?:Two|Three|Free\s+Throw|Jump\s+Shot|Dunk|Lay.?[Uu]p|Tip\s+Shot|Hook\s+Shot)",
        text,
        flags=re.IGNORECASE,
    )
    return verbless.group(1).strip() if verbless else ""


def _prepare_shot_chunk(chunk):
    if "play_type" not in chunk.columns:
        return pd.DataFrame()

    shots = chunk[chunk["play_type"].isin(SHOT_TYPES)].copy()
    if shots.empty:
        return shots

    shots["coord_x"] = pd.to_numeric(shots["coord_x"], errors="coerce")
    shots["coord_y"] = pd.to_numeric(shots["coord_y"], errors="coerce")
    shots = shots[shots["coord_x"].notna() & shots["coord_y"].notna()]
    if shots.empty:
        return shots

    score_col = shots["score"].fillna("").astype(str).str.strip()
    miss_col = shots["miss"].fillna("").astype(str).str.strip()
    shot_text = shots["text"].fillna("").astype(str).str.lower()

    score_nonempty = score_col.ne("")
    miss_nonempty = miss_col.ne("")
    text_says_made = shot_text.str.contains(r"\bmade\b|\bmakes?\b", regex=True, na=False)
    text_says_missed = shot_text.str.contains(r"\bmissed\b|\bmisses?\b", regex=True, na=False)

    shots["is_made"] = score_nonempty | (text_says_made & ~(miss_nonempty | text_says_missed))
    shots["is_three_point"] = shot_text.str.contains("three point", na=False)

    shots["points"] = 0
    shots.loc[shots["is_made"] & shots["is_three_point"], "points"] = 3
    shots.loc[shots["is_made"] & ~shots["is_three_point"], "points"] = 2

    shots["player"] = (score_col + miss_col).str.strip()
    blank_player = shots["player"].eq("")
    if blank_player.any():
        fallback_players = shots.loc[blank_player, "text"].map(_extract_player_from_text)
        shots.loc[blank_player, "player"] = fallback_players

        # Rows that include shooter + shot type but no explicit miss are makes.
        verbless_no_miss = blank_player & shots["player"].ne("") & ~shot_text.str.contains(r"\bmiss", regex=True, na=False)
        shots.loc[verbless_no_miss, "is_made"] = True

        shots.loc[verbless_no_miss & shots["is_three_point"], "points"] = 3
        shots.loc[verbless_no_miss & ~shots["is_three_point"], "points"] = 2

    keep_cols = [
        "game_id",
        "clock",
        "text",
        "home_team",
        "away_team",
        "team",
        "player",
        "coord_x",
        "coord_y",
        "is_made",
        "is_three_point",
        "points",
    ]
    for col in keep_cols:
        if col not in shots.columns:
            shots[col] = pd.NA

    return shots[keep_cols]


def load_shot_attempts(csv_path, chunksize=200000, sample_frac=None, random_state=7):
    chunk_frames = []
    collected_rows = 0
    for idx, chunk in enumerate(
        pd.read_csv(
            csv_path,
            on_bad_lines="skip",
            usecols=lambda c: c in RAW_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        prepared = _prepare_shot_chunk(chunk)
        if prepared.empty:
            continue

        if sample_frac is not None:
            if sample_frac <= 0 or sample_frac > 1:
                raise ValueError("sample_frac must be between 0 and 1.")
            prepared = prepared.sample(frac=sample_frac, random_state=random_state + idx)
            if prepared.empty:
                continue

        chunk_frames.append(prepared)
        collected_rows += len(prepared)

        if idx % 10 == 0:
            print(f"Processed {idx} chunks. Shot rows collected so far: {collected_rows:,}")

    if not chunk_frames:
        raise ValueError("No valid shot attempts were found in the input file.")

    shots = pd.concat(chunk_frames, ignore_index=True)
    shots = shots.drop_duplicates(subset=["game_id", "clock", "text", "team"], keep="first")

    shots["player"] = shots["player"].fillna("").astype(str).str.strip()
    shots["team"] = shots["team"].fillna("").astype(str).str.strip()
    shots = shots[shots["player"].ne("")]

    shots["is_made"] = shots["is_made"].fillna(False).astype(bool)
    shots["is_three_point"] = shots["is_three_point"].fillna(False).astype(bool)
    shots["points"] = pd.to_numeric(shots["points"], errors="coerce").fillna(0).astype(int)

    return shots


def add_model_features(shots, bin_size=3.0, prior_weight=20.0):
    feature_df = shots.copy()

    hoop_x, hoop_y = 25.0, 0.0
    feature_df["distance"] = np.sqrt((feature_df["coord_x"] - hoop_x) ** 2 + (feature_df["coord_y"] - hoop_y) ** 2)
    feature_df["angle"] = np.degrees(
        np.arctan2(np.abs(feature_df["coord_y"] - hoop_y), np.abs(feature_df["coord_x"] - hoop_x) + 1e-6)
    )

    feature_df["x_bin"] = np.floor(feature_df["coord_x"] / bin_size).astype(int)
    feature_df["y_bin"] = np.floor(feature_df["coord_y"] / bin_size).astype(int)

    global_fg = feature_df.groupby("is_three_point")["is_made"].mean().to_dict()
    global_default = float(feature_df["is_made"].mean())
    feature_df["shot_type_prior_fg"] = feature_df["is_three_point"].map(
        {
            False: float(global_fg.get(False, global_default)),
            True: float(global_fg.get(True, global_default)),
        }
    )

    player_agg = (
        feature_df.groupby(["player", "is_three_point"], as_index=False)
        .agg(player_attempts=("is_made", "size"), player_makes=("is_made", "sum"))
    )
    feature_df = feature_df.merge(player_agg, on=["player", "is_three_point"], how="left")

    player_attempts_loo = (feature_df["player_attempts"] - 1).clip(lower=0)
    player_makes_loo = (feature_df["player_makes"] - feature_df["is_made"].astype(int)).clip(lower=0)

    feature_df["player_fg_loo"] = (
        player_makes_loo + prior_weight * feature_df["shot_type_prior_fg"]
    ) / (player_attempts_loo + prior_weight)
    feature_df["player_attempts_loo"] = player_attempts_loo

    grid = (
        feature_df.groupby(["player", "is_three_point", "x_bin", "y_bin"], as_index=False)
        .agg(bin_attempts=("is_made", "size"), bin_makes=("is_made", "sum"))
    )

    neighborhood_parts = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = grid.copy()
            shifted["x_bin"] = shifted["x_bin"] + dx
            shifted["y_bin"] = shifted["y_bin"] + dy
            neighborhood_parts.append(shifted)

    neighborhood = (
        pd.concat(neighborhood_parts, ignore_index=True)
        .groupby(["player", "is_three_point", "x_bin", "y_bin"], as_index=False)
        .agg(local_attempts=("bin_attempts", "sum"), local_makes=("bin_makes", "sum"))
    )

    feature_df = feature_df.merge(
        neighborhood,
        on=["player", "is_three_point", "x_bin", "y_bin"],
        how="left",
    )

    local_attempts_loo = (feature_df["local_attempts"] - 1).clip(lower=0)
    local_makes_loo = (feature_df["local_makes"] - feature_df["is_made"].astype(int)).clip(lower=0)

    feature_df["shooter_local_fg"] = (
        local_makes_loo + prior_weight * feature_df["shot_type_prior_fg"]
    ) / (local_attempts_loo + prior_weight)
    feature_df["shooter_local_attempts"] = local_attempts_loo

    feature_df["is_three_point_int"] = feature_df["is_three_point"].astype(int)
    feature_df["log_player_attempts"] = np.log1p(feature_df["player_attempts_loo"])
    feature_df["log_local_attempts"] = np.log1p(feature_df["shooter_local_attempts"])

    return feature_df


def train_expected_points_model(feature_df, random_state=7):
    X = feature_df[FEATURE_COLUMNS].copy()
    y = feature_df["is_made"].astype(int)
    groups = feature_df["game_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    train_frame = feature_df.iloc[train_idx]
    valid_frame = feature_df.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=100,
        l2_regularization=0.1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    valid_make_prob = np.clip(model.predict_proba(X_valid)[:, 1], 0, 1)
    valid_shot_value = np.where(valid_frame["is_three_point"], 3.0, 2.0)
    valid_expected_points = valid_make_prob * valid_shot_value
    valid_actual_points = valid_frame["points"].astype(float).to_numpy()

    # Baseline: shot-type make rate on training split (2PT vs 3PT).
    baseline_make_rate_by_type = train_frame.groupby("is_three_point")["is_made"].mean().to_dict()
    default_make_rate = float(train_frame["is_made"].mean())
    baseline_make_prob = valid_frame["is_three_point"].map(
        {
            False: float(baseline_make_rate_by_type.get(False, default_make_rate)),
            True: float(baseline_make_rate_by_type.get(True, default_make_rate)),
        }
    ).to_numpy()
    baseline_expected_points = baseline_make_prob * valid_shot_value

    auc_value = float("nan")
    if y_valid.nunique() > 1:
        auc_value = float(roc_auc_score(y_valid, valid_make_prob))

    metrics = {
        "validation_brier": float(brier_score_loss(y_valid, valid_make_prob)),
        "validation_log_loss": float(log_loss(y_valid, valid_make_prob, labels=[0, 1])),
        "validation_auc": auc_value,
        "validation_expected_points_rmse": float(np.sqrt(mean_squared_error(valid_actual_points, valid_expected_points))),
        "baseline_expected_points_rmse": float(np.sqrt(mean_squared_error(valid_actual_points, baseline_expected_points))),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
    }
    return model, metrics


def add_shot_plus_scores(feature_df, model):
    scored = feature_df.copy()
    scored["make_probability_model"] = np.clip(model.predict_proba(scored[FEATURE_COLUMNS])[:, 1], 0, 1)
    shot_value = np.where(scored["is_three_point"], 3.0, 2.0)
    scored["expected_points_model"] = scored["make_probability_model"] * shot_value

    league_avg_expected = float(scored["expected_points_model"].mean())
    if league_avg_expected <= 0:
        league_avg_expected = 1.0

    scored["shot_plus"] = 100.0 * (scored["expected_points_model"] / league_avg_expected)
    scored["shot_plus"] = scored["shot_plus"].clip(60, 140)

    scored["shot_value_added"] = scored["points"] - scored["expected_points_model"]
    scored["result_plus"] = (100 + 20 * scored["shot_value_added"]).clip(50, 150)

    scored["shot_grade"] = pd.cut(
        scored["shot_plus"],
        bins=[-np.inf, 85, 95, 105, 115, np.inf],
        labels=["D", "C", "B", "A", "A+"],
    ).astype(str)

    scored["result_grade"] = pd.cut(
        scored["result_plus"],
        bins=[-np.inf, 80, 95, 105, 120, np.inf],
        labels=["D", "C", "B", "A", "A+"],
    ).astype(str)

    return scored, league_avg_expected


def run_pipeline(
    input_csv,
    output_scored,
    output_model,
    output_metrics,
    chunksize,
    bin_size,
    prior_weight,
    sample_frac,
    random_state,
):
    print(f"Reading shot attempts from {input_csv}...")
    shots = load_shot_attempts(
        input_csv,
        chunksize=chunksize,
        sample_frac=sample_frac,
        random_state=random_state,
    )
    print(f"Loaded {len(shots):,} shot attempts from {shots['game_id'].nunique():,} games.")
    if sample_frac is not None:
        print(f"Sampled approximately {sample_frac:.2%} of rows while loading.")

    print("Building model features (location + shooter efficiency at nearby spots)...")
    feature_df = add_model_features(shots, bin_size=bin_size, prior_weight=prior_weight)

    print("Training expected-points model...")
    model, metrics = train_expected_points_model(feature_df, random_state=random_state)

    print("Scoring shots and generating Shot+ grades...")
    scored, league_avg_expected = add_shot_plus_scores(feature_df, model)

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "bin_size": bin_size,
        "prior_weight": prior_weight,
        "league_avg_expected_points": league_avg_expected,
        "metrics": metrics,
    }

    keep_cols = OUTPUT_BASE_COLUMNS + [
        "make_probability_model",
        "expected_points_model",
        "shot_plus",
        "shot_grade",
        "shot_value_added",
        "result_plus",
        "result_grade",
    ]
    output_df = scored[keep_cols].copy()

    output_scored.parent.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_parquet(output_scored, index=False)
    with open(output_model, "wb") as f:
        pickle.dump(model_bundle, f)

    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved scored shots to {output_scored} ({len(output_df):,} rows).")
    print(f"Saved model bundle to {output_model}.")
    print(f"Saved metrics to {output_metrics}.")
    print("Validation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a Shot+ model on cbb_pbp shot attempts and score every shot using "
            "location + shooter efficiency from nearby spots."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("cbb_pbp.csv"), help="Input cbb play-by-play CSV path")
    parser.add_argument(
        "--output-scored",
        type=Path,
        default=Path("cbb_pbp_shot_plus.parquet"),
        help="Output parquet containing per-shot Shot+ grades",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("shot_plus_model.pkl"),
        help="Serialized model bundle path",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("shot_plus_metrics.json"),
        help="Validation metrics JSON path",
    )
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV read chunk size")
    parser.add_argument("--bin-size", type=float, default=3.0, help="Spatial bin size in feet")
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=20.0,
        help="Empirical Bayes prior weight for shooter efficiency smoothing",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional sample fraction in (0, 1] for faster experimentation",
    )
    parser.add_argument("--random-state", type=int, default=7, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        input_csv=args.input,
        output_scored=args.output_scored,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
        chunksize=args.chunksize,
        bin_size=args.bin_size,
        prior_weight=args.prior_weight,
        sample_frac=args.sample_frac,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

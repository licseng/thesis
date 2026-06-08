from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
MATCHING_VARIABLE_DIR = SCRIPT_DIR / "matching_variables" / "matching_variable_tables_output"
OUTPUT_DIR = SCRIPT_DIR / "matched_cohort_output"

EXPOSED_COHORT = "MHH_psychotic"
CONTROL_COHORT = "only_MHC0"

MHH_INPUT = MATCHING_VARIABLE_DIR / "MHH_psychotic_matching_variables.parquet"
MHC0_INPUT = MATCHING_VARIABLE_DIR / "only_MHC0_matching_variables.parquet"

K_NEIGHBORS = 100
MIN_COSINE_SIMILARITY = 0.80
ELIXHAUSER_CALIPER = 5
ALLOW_RELAXED_ELIXHAUSER = True
RELAXED_ELIXHAUSER_CALIPER = 10
RANDOM_SEED = 42

REQUIRED_COLUMNS = {
    "cohort",
    "subject_id",
    "hadm_id",
    "embedding_row_id",
    "embedding_file",
    "age_at_admission",
    "sex",
    "elixhauser_score",
}

UNMATCHED_REASONS = [
    "no_same_sex_age_bin_candidates",
    "no_available_controls",
    "below_similarity_threshold",
    "elixhauser_caliper",
]


def load_matching_variables(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing matching-variable table: {path}")
    return pd.read_parquet(path)


def validate_required_columns(df: pd.DataFrame, path: Path) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")


def make_age_bin(age: object) -> str:
    if pd.isna(age):
        return "unknown"
    try:
        age_float = float(age)
    except (TypeError, ValueError):
        return "unknown"

    if not 18 <= age_float <= 120:
        return "unknown"
    if age_float < 30:
        return "18-29"
    if age_float < 40:
        return "30-39"
    if age_float < 50:
        return "40-49"
    if age_float < 60:
        return "50-59"
    if age_float < 70:
        return "60-69"
    if age_float < 80:
        return "70-79"
    return "80+"


def prepare_matching_table(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    validate_required_columns(df, path)
    before_count = len(df)

    prepared = df.copy()
    prepared["sex"] = prepared["sex"].astype("string").str.strip().str.upper()
    invalid_sex = prepared["sex"].notna() & ~prepared["sex"].isin(["F", "M"])
    if invalid_sex.any():
        raise ValueError(f"{path} has {invalid_sex.sum():,} rows with sex outside F/M.")

    prepared["age_at_admission"] = pd.to_numeric(
        prepared["age_at_admission"], errors="coerce"
    )
    invalid_age = prepared["age_at_admission"].notna() & ~prepared[
        "age_at_admission"
    ].between(18, 120)
    if invalid_age.any():
        raise ValueError(
            f"{path} has {invalid_age.sum():,} rows with age outside 18-120."
        )

    prepared["elixhauser_score"] = pd.to_numeric(
        prepared["elixhauser_score"], errors="coerce"
    )
    prepared["embedding_row_id"] = pd.to_numeric(
        prepared["embedding_row_id"], errors="coerce"
    )
    prepared["age_bin"] = prepared["age_at_admission"].map(make_age_bin)

    required_match_columns = [
        "sex",
        "age_at_admission",
        "age_bin",
        "elixhauser_score",
        "embedding_row_id",
        "embedding_file",
    ]
    missing_match_data = prepared[required_match_columns].isna().any(axis=1)
    missing_match_data |= prepared["age_bin"].eq("unknown")
    after = prepared.loc[~missing_match_data].copy()
    after["embedding_row_id"] = after["embedding_row_id"].astype(np.int64)
    after["elixhauser_score"] = after["elixhauser_score"].astype(float)

    print(
        f"{path.name}: {before_count:,} rows before validation, "
        f"{len(after):,} rows matchable",
        flush=True,
    )
    return after.reset_index(drop=True), before_count


def load_embeddings_for_rows(
    df: pd.DataFrame,
    embedding_cache: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    vectors = []

    for embedding_file, group in df.groupby("embedding_file", sort=False):
        embedding_path = Path(embedding_file)
        if embedding_file not in embedding_cache:
            if not embedding_path.exists():
                raise FileNotFoundError(f"Missing embedding matrix: {embedding_path}")
            embedding_cache[embedding_file] = np.load(embedding_path, mmap_mode="r")

        matrix = embedding_cache[embedding_file]
        row_ids = group["embedding_row_id"].to_numpy(dtype=np.int64)
        out_of_bounds = (row_ids < 0) | (row_ids >= matrix.shape[0])
        if out_of_bounds.any():
            bad_id = row_ids[out_of_bounds][0]
            raise ValueError(
                f"embedding_row_id {bad_id} is outside matrix bounds for "
                f"{embedding_path}: {matrix.shape[0]} rows"
            )

        rows.append(group)
        vectors.append(np.asarray(matrix[row_ids], dtype=np.float32))

    aligned_df = pd.concat(rows, ignore_index=True)
    aligned_embeddings = np.vstack(vectors).astype(np.float32)
    return aligned_df, aligned_embeddings


def build_stratified_indexes(
    controls: pd.DataFrame,
    control_embeddings: np.ndarray,
) -> dict[tuple[str, str], dict[str, object]]:
    indexes = {}
    for stratum, row_indexes in controls.groupby(["sex", "age_bin"]).groups.items():
        positions = np.array(list(row_indexes), dtype=np.int64)
        stratum_embeddings = control_embeddings[positions]
        model = NearestNeighbors(metric="cosine")
        model.fit(stratum_embeddings)
        indexes[stratum] = {
            "metadata": controls.iloc[positions].reset_index(drop=True),
            "embeddings": stratum_embeddings,
            "model": model,
        }
        print(
            f"Built index for {stratum}: {len(positions):,} controls",
            flush=True,
        )
    return indexes


def compute_candidate_pool_sizes(
    exposed: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    pool_sizes = (
        controls.groupby(["sex", "age_bin"])
        .size()
        .rename("candidate_pool_size")
        .reset_index()
    )
    exposed = exposed.merge(pool_sizes, on=["sex", "age_bin"], how="left")
    exposed["candidate_pool_size"] = (
        exposed["candidate_pool_size"].fillna(0).astype(np.int64)
    )
    return exposed


def choose_best_candidate(
    candidates: pd.DataFrame,
    caliper: float,
) -> pd.Series | None:
    eligible = candidates.loc[
        (candidates["cosine_similarity"] >= MIN_COSINE_SIMILARITY)
        & (candidates["abs_elixhauser_difference"] <= caliper)
    ].copy()
    if eligible.empty:
        return None

    eligible = eligible.sort_values(
        ["cosine_similarity", "abs_elixhauser_difference"],
        ascending=[False, True],
    )
    return eligible.iloc[0]


def find_match_for_row(
    mhh_row: pd.Series,
    mhh_embedding: np.ndarray,
    indexes: dict[tuple[str, str], dict[str, object]],
    used_control_keys: set[tuple[int, int]],
) -> tuple[dict[str, object] | None, str | None]:
    stratum = (mhh_row["sex"], mhh_row["age_bin"])
    if stratum not in indexes:
        return None, "no_same_sex_age_bin_candidates"

    index = indexes[stratum]
    controls = index["metadata"]
    model = index["model"]
    n_neighbors = min(K_NEIGHBORS, len(controls))
    if n_neighbors == 0:
        return None, "no_same_sex_age_bin_candidates"

    distances, neighbor_positions = model.kneighbors(
        mhh_embedding.reshape(1, -1),
        n_neighbors=n_neighbors,
        return_distance=True,
    )
    candidate_rows = controls.iloc[neighbor_positions[0]].copy()
    candidate_rows["embedding_distance"] = distances[0]
    candidate_rows["cosine_similarity"] = 1.0 - candidate_rows["embedding_distance"]
    candidate_rows["abs_elixhauser_difference"] = (
        candidate_rows["elixhauser_score"] - mhh_row["elixhauser_score"]
    ).abs()

    candidate_rows["control_key"] = list(
        zip(candidate_rows["subject_id"], candidate_rows["hadm_id"])
    )
    available = candidate_rows.loc[
        ~candidate_rows["control_key"].isin(used_control_keys)
    ].copy()
    if available.empty:
        return None, "no_available_controls"

    similarity_filtered = available.loc[
        available["cosine_similarity"] >= MIN_COSINE_SIMILARITY
    ].copy()
    if similarity_filtered.empty:
        return None, "below_similarity_threshold"

    strict = choose_best_candidate(similarity_filtered, ELIXHAUSER_CALIPER)
    if strict is not None:
        return {"control": strict, "match_type": "strict_elixhauser"}, None

    if ALLOW_RELAXED_ELIXHAUSER:
        relaxed = choose_best_candidate(
            similarity_filtered,
            RELAXED_ELIXHAUSER_CALIPER,
        )
        if relaxed is not None:
            return {"control": relaxed, "match_type": "relaxed_elixhauser"}, None

    return None, "elixhauser_caliper"


def pair_record(
    pair_id: int,
    mhh_row: pd.Series,
    control_row: pd.Series,
    match_type: str,
) -> dict[str, object]:
    cosine_similarity = float(control_row["cosine_similarity"])
    return {
        "pair_id": pair_id,
        "mhh_subject_id": mhh_row["subject_id"],
        "mhh_hadm_id": mhh_row["hadm_id"],
        "mhh_chief_complaint_raw": mhh_row.get("chief_complaint_raw"),
        "mhh_chief_complaint_normalized": mhh_row.get("chief_complaint_normalized"),
        "mhh_sex": mhh_row["sex"],
        "mhh_age_at_admission": mhh_row["age_at_admission"],
        "mhh_age_bin": mhh_row["age_bin"],
        "mhh_elixhauser_score": mhh_row["elixhauser_score"],
        "mhc0_subject_id": control_row["subject_id"],
        "mhc0_hadm_id": control_row["hadm_id"],
        "mhc0_chief_complaint_raw": control_row.get("chief_complaint_raw"),
        "mhc0_chief_complaint_normalized": control_row.get(
            "chief_complaint_normalized"
        ),
        "mhc0_sex": control_row["sex"],
        "mhc0_age_at_admission": control_row["age_at_admission"],
        "mhc0_age_bin": control_row["age_bin"],
        "mhc0_elixhauser_score": control_row["elixhauser_score"],
        "cosine_similarity": cosine_similarity,
        "embedding_distance": 1.0 - cosine_similarity,
        "abs_elixhauser_difference": float(
            control_row["abs_elixhauser_difference"]
        ),
        "match_type": match_type,
        "candidate_pool_size": int(mhh_row["candidate_pool_size"]),
    }


def match_cohorts(
    exposed: pd.DataFrame,
    exposed_embeddings: np.ndarray,
    controls: pd.DataFrame,
    control_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = build_stratified_indexes(controls, control_embeddings)
    exposed = compute_candidate_pool_sizes(exposed, controls)
    exposed = exposed.assign(_embedding_position=np.arange(len(exposed)))
    exposed = exposed.sort_values(
        ["candidate_pool_size", "elixhauser_score"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    matched_pairs = []
    unmatched_rows = []
    used_control_keys: set[tuple[int, int]] = set()

    for position, mhh_row in exposed.iterrows():
        if (position + 1) % 500 == 0:
            print(
                f"Processed {position + 1:,} of {len(exposed):,} MHH rows; "
                f"matched {len(matched_pairs):,}",
                flush=True,
            )

        embedding = exposed_embeddings[int(mhh_row["_embedding_position"])]
        match, reason = find_match_for_row(
            mhh_row,
            embedding,
            indexes,
            used_control_keys,
        )
        if match is None:
            unmatched = mhh_row.drop(labels=["_embedding_position"]).to_dict()
            unmatched["unmatched_reason"] = reason
            unmatched_rows.append(unmatched)
            continue

        control = match["control"]
        control_key = (int(control["subject_id"]), int(control["hadm_id"]))
        used_control_keys.add(control_key)
        matched_pairs.append(
            pair_record(
                len(matched_pairs) + 1,
                mhh_row,
                control,
                match["match_type"],
            )
        )

    return pd.DataFrame(matched_pairs), pd.DataFrame(unmatched_rows)


def build_summary(
    matched_pairs: pd.DataFrame,
    unmatched: pd.DataFrame,
    total_mhh_before_filtering: int,
    total_mhc0_before_filtering: int,
    total_mhh_matchable: int,
    total_mhc0_matchable: int,
) -> pd.DataFrame:
    n_matched = len(matched_pairs)
    n_unmatched = len(unmatched)
    cosine_q1 = matched_pairs["cosine_similarity"].quantile(0.25) if n_matched else np.nan
    cosine_q3 = matched_pairs["cosine_similarity"].quantile(0.75) if n_matched else np.nan

    row = {
        "total_mhh_before_filtering": total_mhh_before_filtering,
        "total_mhc0_before_filtering": total_mhc0_before_filtering,
        "total_mhh_matchable": total_mhh_matchable,
        "total_mhc0_matchable": total_mhc0_matchable,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "pct_matched": 100.0 * n_matched / total_mhh_matchable
        if total_mhh_matchable
        else np.nan,
        "median_cosine_similarity": matched_pairs["cosine_similarity"].median()
        if n_matched
        else np.nan,
        "iqr_cosine_similarity": cosine_q3 - cosine_q1 if n_matched else np.nan,
        "min_cosine_similarity": matched_pairs["cosine_similarity"].min()
        if n_matched
        else np.nan,
        "median_embedding_distance": matched_pairs["embedding_distance"].median()
        if n_matched
        else np.nan,
        "median_abs_elixhauser_difference": matched_pairs[
            "abs_elixhauser_difference"
        ].median()
        if n_matched
        else np.nan,
        "n_strict_elixhauser_matches": int(
            (matched_pairs.get("match_type", pd.Series(dtype=str)) == "strict_elixhauser").sum()
        ),
        "n_relaxed_elixhauser_matches": int(
            (
                matched_pairs.get("match_type", pd.Series(dtype=str))
                == "relaxed_elixhauser"
            ).sum()
        ),
    }

    reason_counts = (
        unmatched["unmatched_reason"].value_counts().to_dict()
        if not unmatched.empty
        else {}
    )
    for reason in UNMATCHED_REASONS:
        row[f"n_unmatched_{reason}"] = int(reason_counts.get(reason, 0))

    return pd.DataFrame([row])


def quality_checks(matched_pairs: pd.DataFrame) -> None:
    if matched_pairs.empty:
        return

    assert not matched_pairs.duplicated(["mhc0_subject_id", "mhc0_hadm_id"]).any()
    assert not matched_pairs.duplicated(["mhh_subject_id", "mhh_hadm_id"]).any()
    assert (matched_pairs["mhh_sex"] == matched_pairs["mhc0_sex"]).all()
    assert (matched_pairs["mhh_age_bin"] == matched_pairs["mhc0_age_bin"]).all()

    below_similarity = matched_pairs["cosine_similarity"] < MIN_COSINE_SIMILARITY
    if below_similarity.any():
        warnings.warn(
            f"{below_similarity.sum():,} matched pairs are below "
            f"MIN_COSINE_SIMILARITY={MIN_COSINE_SIMILARITY}.",
            RuntimeWarning,
            stacklevel=2,
        )

    strict_over = (
        matched_pairs["match_type"].eq("strict_elixhauser")
        & (matched_pairs["abs_elixhauser_difference"] > ELIXHAUSER_CALIPER)
    )
    if strict_over.any():
        warnings.warn(
            f"{strict_over.sum():,} strict matches exceed ELIXHAUSER_CALIPER.",
            RuntimeWarning,
            stacklevel=2,
        )

    relaxed_over = (
        matched_pairs["match_type"].eq("relaxed_elixhauser")
        & (matched_pairs["abs_elixhauser_difference"] > RELAXED_ELIXHAUSER_CALIPER)
    )
    if relaxed_over.any():
        warnings.warn(
            f"{relaxed_over.sum():,} relaxed matches exceed "
            "RELAXED_ELIXHAUSER_CALIPER.",
            RuntimeWarning,
            stacklevel=2,
        )


def write_outputs(
    matched_pairs: pd.DataFrame,
    unmatched: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    matched_pairs.to_parquet(OUTPUT_DIR / "matched_pairs.parquet", index=False)
    matched_pairs.to_csv(OUTPUT_DIR / "matched_pairs.csv", index=False)
    unmatched.to_parquet(OUTPUT_DIR / "unmatched_MHH_psychotic.parquet", index=False)
    summary.to_csv(OUTPUT_DIR / "matching_summary.csv", index=False)
    print(f"Saved matched cohort outputs to: {OUTPUT_DIR}", flush=True)


def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("Loading matching-variable tables", flush=True)
    mhh_raw = load_matching_variables(MHH_INPUT)
    mhc0_raw = load_matching_variables(MHC0_INPUT)

    mhh, total_mhh_before = prepare_matching_table(mhh_raw, MHH_INPUT)
    mhc0, total_mhc0_before = prepare_matching_table(mhc0_raw, MHC0_INPUT)

    embedding_cache: dict[str, np.ndarray] = {}
    print("Loading MHH embeddings", flush=True)
    mhh, mhh_embeddings = load_embeddings_for_rows(mhh, embedding_cache)
    print("Loading MHC0 embeddings", flush=True)
    mhc0, mhc0_embeddings = load_embeddings_for_rows(mhc0, embedding_cache)

    print("Starting greedy 1:1 matching", flush=True)
    matched_pairs, unmatched = match_cohorts(
        mhh,
        mhh_embeddings,
        mhc0,
        mhc0_embeddings,
    )
    quality_checks(matched_pairs)

    summary = build_summary(
        matched_pairs,
        unmatched,
        total_mhh_before,
        total_mhc0_before,
        len(mhh),
        len(mhc0),
    )
    print("Matching summary:", flush=True)
    print(summary.to_string(index=False), flush=True)

    write_outputs(matched_pairs, unmatched, summary)


if __name__ == "__main__":
    main()

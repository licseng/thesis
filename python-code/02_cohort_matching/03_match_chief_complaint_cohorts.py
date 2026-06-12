"""Create a 1:1 matched cohort for the diagnostic overshadowing analysis.

This script matches each exposed `MHH_psychotic` admission to at most one
`only_MHC0` control admission without replacement. The primary matching signal
is chief-complaint embedding similarity. Sex, age bin, QuickUMLS concept overlap,
and Elixhauser score are used as candidate restrictions or calipers before the
final control is selected.

Matching order:
    1. Load admission-level matching-variable tables.
    2. Validate sex, age, Elixhauser, embedding row IDs, and embedding paths.
    3. Build control indexes within sex + age-bin strata.
    4. For each exposed admission, search same-sex controls in the same age bin
       plus neighboring bins.
    5. Optionally restrict candidate controls to those sharing QuickUMLS terms.
    6. Retrieve nearest controls by BERT embedding cosine distance.
    7. Remove already-used controls.
    8. Apply cosine-similarity and Elixhauser calipers.
    9. Choose the best remaining control, prioritizing chief-complaint
       similarity and using age/Elixhauser closeness as tie-breakers.

Inputs:
    02_matching_variables/matching_variable_tables_output/MHH_psychotic_matching_variables.parquet
    02_matching_variables/matching_variable_tables_output/only_MHC0_matching_variables.parquet

Outputs:
    matched_cohort_output/matched_pairs.parquet
    matched_cohort_output/matched_pairs.csv
    matched_cohort_output/unmatched_MHH_psychotic.parquet
    matched_cohort_output/matching_summary.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


SCRIPT_DIR = Path(__file__).resolve().parent
MATCHING_VARIABLE_DIR = SCRIPT_DIR / "02_matching_variables" / "matching_variable_tables_output"
OUTPUT_DIR = SCRIPT_DIR / "matched_cohort_output"

EXPOSED_COHORT = "MHH_psychotic"
CONTROL_COHORT = "only_MHC0"

MHH_INPUT = MATCHING_VARIABLE_DIR / "MHH_psychotic_matching_variables.parquet"
MHC0_INPUT = MATCHING_VARIABLE_DIR / "only_MHC0_matching_variables.parquet"

# Matching configuration.
K_NEIGHBORS = 100
MIN_COSINE_SIMILARITY = 0.90
COSINE_TIE_TOLERANCE = 0.01
AGE_BIN_NEIGHBOR_RADIUS = 1

# Optional coarse clinical-concept screen before embedding-nearest-neighbor
# search. With the current values, controls must share at least one QuickUMLS
# term with the exposed admission. If no such candidates exist, fallback keeps
# the admission matchable using the older sex/nearby-age-bin embedding search.
USE_QUICKUMLS_CANDIDATE_FILTER = True
MIN_SHARED_QUICKUMLS_TERMS = 1
MIN_QUICKUMLS_JACCARD = 0.0
FALL_BACK_TO_NO_QUICKUMLS_FILTER = True

# Elixhauser score calipers. Strict matches use the primary caliper; if enabled,
# relaxed matches can use a wider caliper.
ELIXHAUSER_CALIPER = 5
ALLOW_RELAXED_ELIXHAUSER = True
RELAXED_ELIXHAUSER_CALIPER = 10
RANDOM_SEED = 42

# Age bins used during candidate restriction. `AGE_BIN_NEIGHBOR_RADIUS = 1`
# means same age bin plus one lower and one higher neighboring bin.
AGE_BIN_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

# Minimal schema expected from the matching-variable table generation step.
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
    "no_same_sex_nearby_age_bin_candidates",
    "no_quickumls_overlap_candidates",
    "no_available_controls",
    "below_similarity_threshold",
    "elixhauser_caliper",
]


# Load one cohort's matching-variable table.
def load_matching_variables(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing matching-variable table: {path}")
    return pd.read_parquet(path)


# Validate that the matching-variable table has the required columns for this
# matching step.
def validate_required_columns(df: pd.DataFrame, path: Path) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")


# Convert numeric age to the matching age-bin labels.
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


# Return the age bins eligible for one exposed admission. The radius controls how
# many neighboring bins are included on each side of the exposed admission's bin.
def nearby_age_bins(age_bin: str) -> list[str]:
    if age_bin not in AGE_BIN_ORDER:
        return []

    index = AGE_BIN_ORDER.index(age_bin)
    start = max(0, index - AGE_BIN_NEIGHBOR_RADIUS)
    end = min(len(AGE_BIN_ORDER), index + AGE_BIN_NEIGHBOR_RADIUS + 1)
    return AGE_BIN_ORDER[start:end]


# Parse pipe-separated QuickUMLS term strings into lowercase sets for overlap
# and Jaccard calculations.
def split_quickumls_terms(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        term.strip().lower()
        for term in str(value).split("|")
        if term.strip()
    }


# Calculate shared QuickUMLS term count and Jaccard similarity between one
# exposed admission and one candidate control.
def quickumls_overlap_stats(
    exposed_terms: set[str],
    control_terms: set[str],
) -> tuple[int, float]:
    if not exposed_terms or not control_terms:
        return 0, 0.0

    shared_terms = exposed_terms & control_terms
    union_terms = exposed_terms | control_terms
    jaccard = len(shared_terms) / len(union_terms) if union_terms else 0.0
    return len(shared_terms), jaccard


# Return True if a control passes the optional QuickUMLS candidate screen.
def passes_quickumls_candidate_filter(
    exposed_terms: set[str],
    control_terms: set[str],
) -> bool:
    shared_count, jaccard = quickumls_overlap_stats(exposed_terms, control_terms)
    return (
        shared_count >= MIN_SHARED_QUICKUMLS_TERMS
        and jaccard >= MIN_QUICKUMLS_JACCARD
    )


# Validate and prepare one cohort table for matching. Rows missing required
# matching data are dropped before embedding matrices are loaded.
def prepare_matching_table(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    validate_required_columns(df, path)
    before_count = len(df)

    prepared = df.copy()
    if USE_QUICKUMLS_CANDIDATE_FILTER and "quickumls_terms" not in prepared.columns:
        raise ValueError(
            f"{path} is missing quickumls_terms, required when "
            "USE_QUICKUMLS_CANDIDATE_FILTER=True."
        )

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
    if "quickumls_terms" in after.columns:
        after["_quickumls_term_set"] = after["quickumls_terms"].map(split_quickumls_terms)
    else:
        after["_quickumls_term_set"] = [set() for _ in range(len(after))]

    print(
        f"{path.name}: {before_count:,} rows before validation, "
        f"{len(after):,} rows matchable",
        flush=True,
    )
    return after.reset_index(drop=True), before_count


# Load embedding vectors for the rows in a matching table, using `embedding_row_id`
# to align metadata rows to the correct row in each `.npy` embedding matrix.
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


# Build a nearest-neighbor index for each control sex + age-bin stratum. The
# final matching search combines the exposed admission's same/neighboring age
# strata, while keeping sex exact.
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


# Count the size of each exposed admission's same-sex + nearby-age-bin candidate
# pool. This lets the greedy matcher process harder-to-match admissions first.
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
    pool_lookup = {
        (row.sex, row.age_bin): int(row.candidate_pool_size)
        for row in pool_sizes.itertuples(index=False)
    }

    exposed = exposed.copy()
    exposed["candidate_pool_size"] = exposed.apply(
        lambda row: sum(
            pool_lookup.get((row["sex"], age_bin), 0)
            for age_bin in nearby_age_bins(row["age_bin"])
        ),
        axis=1,
    )
    return exposed


# Select the best candidate after cosine and Elixhauser filtering. Candidates
# within `COSINE_TIE_TOLERANCE` of the best cosine similarity are tie-broken by
# closer age, same age bin, and closer Elixhauser score.
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

    max_similarity = eligible["cosine_similarity"].max()
    eligible = eligible.loc[
        eligible["cosine_similarity"] >= max_similarity - COSINE_TIE_TOLERANCE
    ].copy()
    eligible = eligible.sort_values(
        [
            "abs_age_difference",
            "same_age_bin",
            "cosine_similarity",
            "abs_elixhauser_difference",
        ],
        ascending=[True, False, False, True],
    )
    return eligible.iloc[0]


# Fit a small temporary KNN model and retrieve nearest controls from a filtered
# metadata/embedding subset. This is used after QuickUMLS term filtering.
def query_nearest_controls(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> pd.DataFrame:
    n_neighbors = min(K_NEIGHBORS, len(metadata))
    if n_neighbors == 0:
        return pd.DataFrame()

    model = NearestNeighbors(metric="cosine")
    model.fit(embeddings)
    distances, neighbor_positions = model.kneighbors(
        query_embedding.reshape(1, -1),
        n_neighbors=n_neighbors,
        return_distance=True,
    )
    candidates = metadata.iloc[neighbor_positions[0]].copy()
    candidates["embedding_distance"] = distances[0]
    candidates["cosine_similarity"] = 1.0 - candidates["embedding_distance"]
    return candidates


# Query an already-built control stratum index. This is used for the normal
# unfiltered search path and for fallback when QuickUMLS overlap filtering finds
# no candidate controls.
def query_existing_index(
    metadata: pd.DataFrame,
    model: NearestNeighbors,
    query_embedding: np.ndarray,
) -> pd.DataFrame:
    n_neighbors = min(K_NEIGHBORS, len(metadata))
    if n_neighbors == 0:
        return pd.DataFrame()

    distances, neighbor_positions = model.kneighbors(
        query_embedding.reshape(1, -1),
        n_neighbors=n_neighbors,
        return_distance=True,
    )
    candidates = metadata.iloc[neighbor_positions[0]].copy()
    candidates["embedding_distance"] = distances[0]
    candidates["cosine_similarity"] = 1.0 - candidates["embedding_distance"]
    return candidates


# Find one control match for one exposed admission. This function applies the
# same-sex/nearby-age-bin restriction, optional QuickUMLS overlap screen,
# embedding nearest-neighbor retrieval, no-reuse rule, cosine threshold, and
# Elixhauser caliper.
def find_match_for_row(
    mhh_row: pd.Series,
    mhh_embedding: np.ndarray,
    indexes: dict[tuple[str, str], dict[str, object]],
    used_control_keys: set[tuple[int, int]],
) -> tuple[dict[str, object] | None, str | None]:
    age_bins = nearby_age_bins(mhh_row["age_bin"])
    candidate_parts = []
    fallback_strata = []
    used_quickumls_candidate_filter = USE_QUICKUMLS_CANDIDATE_FILTER
    used_quickumls_filter_fallback = False

    for age_bin in age_bins:
        stratum = (mhh_row["sex"], age_bin)
        if stratum not in indexes:
            continue

        index = indexes[stratum]
        controls = index["metadata"]
        embeddings = index["embeddings"]
        model = index["model"]
        if len(controls) == 0:
            continue

        fallback_strata.append((controls, model))

        if USE_QUICKUMLS_CANDIDATE_FILTER:
            exposed_terms = mhh_row["_quickumls_term_set"]
            quickumls_filter = controls["_quickumls_term_set"].map(
                lambda control_terms: passes_quickumls_candidate_filter(
                    exposed_terms,
                    control_terms,
                )
            )
            filtered_controls = controls.loc[quickumls_filter].reset_index(drop=True)
            if filtered_controls.empty:
                continue

            filtered_embeddings = embeddings[np.flatnonzero(quickumls_filter.to_numpy())]
            stratum_candidates = query_nearest_controls(
                filtered_controls,
                filtered_embeddings,
                mhh_embedding,
            )
        else:
            stratum_candidates = query_existing_index(controls, model, mhh_embedding)

        candidate_parts.append(stratum_candidates)

    if not candidate_parts:
        if not fallback_strata:
            return None, "no_same_sex_nearby_age_bin_candidates"
        if not FALL_BACK_TO_NO_QUICKUMLS_FILTER:
            return None, "no_quickumls_overlap_candidates"

        candidate_parts = [
            query_existing_index(controls, model, mhh_embedding)
            for controls, model in fallback_strata
        ]
        used_quickumls_candidate_filter = False
        used_quickumls_filter_fallback = True

    candidate_rows = pd.concat(candidate_parts, ignore_index=True)
    candidate_rows["abs_elixhauser_difference"] = (
        candidate_rows["elixhauser_score"] - mhh_row["elixhauser_score"]
    ).abs()
    candidate_rows["same_age_bin"] = candidate_rows["age_bin"].eq(mhh_row["age_bin"])
    candidate_rows["abs_age_difference"] = (
        candidate_rows["age_at_admission"] - mhh_row["age_at_admission"]
    ).abs()
    age_bin_position = AGE_BIN_ORDER.index(mhh_row["age_bin"])
    candidate_rows["age_bin_distance"] = candidate_rows["age_bin"].map(
        lambda value: abs(AGE_BIN_ORDER.index(value) - age_bin_position)
    )
    overlap_stats = candidate_rows["_quickumls_term_set"].map(
        lambda control_terms: quickumls_overlap_stats(
            mhh_row["_quickumls_term_set"],
            control_terms,
        )
    )
    candidate_rows["quickumls_shared_term_count"] = overlap_stats.map(lambda item: item[0])
    candidate_rows["quickumls_jaccard"] = overlap_stats.map(lambda item: item[1])
    candidate_rows["used_quickumls_candidate_filter"] = used_quickumls_candidate_filter
    candidate_rows["used_quickumls_filter_fallback"] = used_quickumls_filter_fallback

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


# Convert one matched exposed/control pair into a flat output record.
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
        "mhh_quickumls_terms": mhh_row.get("quickumls_terms"),
        "mhh_quickumls_extracted_text": mhh_row.get("quickumls_extracted_text"),
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
        "mhc0_quickumls_terms": control_row.get("quickumls_terms"),
        "mhc0_quickumls_extracted_text": control_row.get("quickumls_extracted_text"),
        "mhc0_sex": control_row["sex"],
        "mhc0_age_at_admission": control_row["age_at_admission"],
        "mhc0_age_bin": control_row["age_bin"],
        "mhc0_elixhauser_score": control_row["elixhauser_score"],
        "same_age_bin": bool(control_row["same_age_bin"]),
        "age_bin_distance": int(control_row["age_bin_distance"]),
        "abs_age_difference": float(control_row["abs_age_difference"]),
        "cosine_similarity": cosine_similarity,
        "embedding_distance": 1.0 - cosine_similarity,
        "abs_elixhauser_difference": float(
            control_row["abs_elixhauser_difference"]
        ),
        "quickumls_shared_term_count": int(control_row["quickumls_shared_term_count"]),
        "quickumls_jaccard": float(control_row["quickumls_jaccard"]),
        "used_quickumls_candidate_filter": bool(
            control_row["used_quickumls_candidate_filter"]
        ),
        "used_quickumls_filter_fallback": bool(
            control_row["used_quickumls_filter_fallback"]
        ),
        "match_type": match_type,
        "candidate_pool_size": int(mhh_row["candidate_pool_size"]),
    }


# Greedy 1:1 matching without replacement. Exposed admissions with smaller
# candidate pools and higher Elixhauser scores are handled first.
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
            drop_labels = [
                label
                for label in ["_embedding_position", "_quickumls_term_set"]
                if label in mhh_row.index
            ]
            unmatched = mhh_row.drop(labels=drop_labels).to_dict()
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


# Build a one-row matching summary for review and methods reporting.
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
        "median_abs_age_difference": matched_pairs["abs_age_difference"].median()
        if n_matched
        else np.nan,
        "n_same_age_bin_matches": int(matched_pairs["same_age_bin"].sum())
        if n_matched
        else 0,
        "pct_same_age_bin_matches": 100.0 * matched_pairs["same_age_bin"].mean()
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
        "quickumls_candidate_filter_enabled": USE_QUICKUMLS_CANDIDATE_FILTER,
        "min_shared_quickumls_terms": MIN_SHARED_QUICKUMLS_TERMS,
        "min_quickumls_jaccard": MIN_QUICKUMLS_JACCARD,
        "n_matches_using_quickumls_candidate_filter": int(
            matched_pairs.get(
                "used_quickumls_candidate_filter",
                pd.Series(dtype=bool),
            ).sum()
        ),
        "n_matches_using_quickumls_filter_fallback": int(
            matched_pairs.get(
                "used_quickumls_filter_fallback",
                pd.Series(dtype=bool),
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


# Sanity checks for no reused controls, one match per exposed admission, exact
# sex matching, and configured caliper/threshold expectations.
def quality_checks(matched_pairs: pd.DataFrame) -> None:
    if matched_pairs.empty:
        return

    assert not matched_pairs.duplicated(["mhc0_subject_id", "mhc0_hadm_id"]).any()
    assert not matched_pairs.duplicated(["mhh_subject_id", "mhh_hadm_id"]).any()
    assert (matched_pairs["mhh_sex"] == matched_pairs["mhc0_sex"]).all()

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


# Write matched pairs, unmatched exposed admissions, and summary outputs.
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


# Script entry point: load covariates/embeddings, run matching, validate outputs,
# and write the matched cohort files.
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

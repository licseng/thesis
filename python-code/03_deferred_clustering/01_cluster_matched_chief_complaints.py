"""Cluster matched chief-complaint admissions for post-matching quality control.

This script is a validation and interpretability step only. It does not create,
modify, or filter the matched cohort. The matched cohort is already defined by
`02_cohort_matching/03_match_chief_complaint_cohorts.py`.

The script converts matched pairs into one row per admission, attaches the
already-computed Bio_ClinicalBERT embedding for each admission, and runs three
independent clustering routes:

1. BERT route:
   - cluster matched-admission BERT embeddings with UMAP + HDBSCAN
   - create a 2D UMAP projection for plotting

2. TF-IDF route:
   - vectorize normalized chief complaints with word n-gram TF-IDF
   - reduce with TruncatedSVD
   - cluster with UMAP + HDBSCAN
   - create a 2D UMAP projection for plotting

3. Agglomerative route:
   - cluster matched-admission Bio_ClinicalBERT embeddings directly with cosine
     agglomerative clustering across several candidate k values
   - produce broad, reviewable candidate complaint groups for downstream
     workup/lab comparison

The agglomerative route is not intended to prove individual pair validity. It is
for clinically interpretable candidate grouping; final complaint groups will
likely need manual review and possibly manual merging.

Outputs are local QC artifacts under:
    03_deferred_clustering/chief_complaint_cluster_qc_output/
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MATCHED_PAIRS_PATH = (
    PROJECT_DIR / "02_cohort_matching" / "matched_cohort_output" / "matched_pairs.parquet"
)
EMBEDDING_DIR = (
    PROJECT_DIR
    / "01_discharge_note_preprocessing"
    / "02_chief_complaint"
    / "02_embedding"
    / "chief_complaint_embeddings"
)
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_cluster_qc_output"

RANDOM_STATE = 42

# Default clustering route. You can override this from the command line:
#     python 01_cluster_matched_chief_complaints.py --route bert
#     python 01_cluster_matched_chief_complaints.py --route tfidf
#     python 01_cluster_matched_chief_complaints.py --route both
#
# Accepted values:
#   "bert"  - Bio_ClinicalBERT embeddings only
#   "tfidf" - normalized chief complaint TF-IDF only
#   "both"  - run both routes
#   "agglomerative" - Bio_ClinicalBERT agglomerative clustering only
#   "lexical_hierarchy" - frequency-aware parent complaint candidates
#   "all" - run BERT, TF-IDF, agglomerative, and lexical hierarchy routes
CLUSTERING_ROUTE = "both"

# UMAP/HDBSCAN starting parameters.
UMAP_N_NEIGHBORS = 30
UMAP_CLUSTER_MIN_DIST = 0.0
UMAP_PLOT_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_CLUSTER_COMPONENTS = 10
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_MIN_CLUSTER_SIZE = 5

# TF-IDF route starting parameters.
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.80
SVD_MAX_COMPONENTS = 50

# Agglomerative route candidate cluster counts. This route is meant to create
# broad candidate complaint groups for workup/lab comparison, then support manual
# review and manual merging into clinically meaningful groups.
AGGLOMERATIVE_N_CLUSTERS_LIST = [20, 30, 40, 50, 75]

# Lexical hierarchy route settings. This route is not standard clustering; it
# extracts frequent parent/core complaint phrases and their longer child
# complaints for manual review into broad workup/lab comparison groups.
LEXICAL_NGRAM_RANGE = (1, 5)
LEXICAL_MIN_PARENT_FREQUENCY = 20
LEXICAL_MIN_CHILD_COMPLAINTS = 2
LEXICAL_MAX_EXAMPLE_CHILDREN = 20
LEXICAL_SINGLE_TOKEN_PARENT_TERMS = {
    "fall",
    "fever",
    "fracture",
    "seizure",
    "syncope",
    "trauma",
}

# Human-readable cluster summaries.
TOP_COMPLAINTS_N = 10
TOP_TERMS_N = 20

# UMAP plot styling. Pair links connect the two admissions from the same
# matched pair; low alpha keeps the plot readable even with thousands of pairs.
DRAW_PAIR_LINES = True
PAIR_LINE_ALPHA = 0.08
PAIR_LINE_WIDTH = 0.35
POINT_SIZE = 8
POINT_ALPHA = 0.80

MATCHED_PAIR_REQUIRED_COLUMNS = {
    "pair_id",
    "mhh_subject_id",
    "mhh_hadm_id",
    "mhh_chief_complaint_normalized",
    "mhc0_subject_id",
    "mhc0_hadm_id",
    "mhc0_chief_complaint_normalized",
}

EMBEDDING_METADATA_REQUIRED_COLUMNS = {
    "embedding_row_id",
    "cohort",
    "subject_id",
    "hadm_id",
    "chief_complaint_normalized",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Cluster matched chief-complaint admissions for post-matching QC."
        )
    )
    parser.add_argument(
        "--route",
        default=CLUSTERING_ROUTE,
        choices=[
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "bert",
            "tfidf",
            "both",
            "agglomerative",
            "lexical_hierarchy",
            "all",
        ],
        help=(
            "Clustering route to run. Use 1/bert for BERT, 2/tfidf for TF-IDF, "
            "3/both for both HDBSCAN routes, 4/agglomerative for cosine "
            "agglomerative clustering, 6/lexical_hierarchy for parent complaint "
            "candidates, or 5/all for all routes. Default comes from "
            "CLUSTERING_ROUTE."
        ),
    )
    return parser.parse_args()


def normalize_route_value(route_value: str) -> str:
    """Map numeric route choices to readable route names."""
    route = route_value.strip().lower()
    numeric_map = {
        "1": "bert",
        "2": "tfidf",
        "3": "both",
        "4": "agglomerative",
        "5": "all",
        "6": "lexical_hierarchy",
    }
    return numeric_map.get(route, route)


def selected_routes(route_value: str) -> set[str]:
    """Return the configured clustering routes."""
    route = normalize_route_value(route_value)
    if route == "both":
        return {"bert", "tfidf"}
    if route == "all":
        return {"bert", "tfidf", "agglomerative", "lexical_hierarchy"}
    if route in {"bert", "tfidf", "agglomerative", "lexical_hierarchy"}:
        return {route}
    raise ValueError(
        "CLUSTERING_ROUTE must be one of: 'bert', 'tfidf', 'both', "
        "'agglomerative', 'lexical_hierarchy', or 'all'. "
        f"Current value: {route_value!r}"
    )


def require_clustering_dependencies() -> tuple[Any, Any]:
    """Import UMAP and HDBSCAN with a useful error if they are missing."""
    numba_cache_dir = OUTPUT_DIR / ".numba_cache"
    matplotlib_cache_dir = OUTPUT_DIR / ".matplotlib_cache"
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `umap-learn`. Install it in thesis_env with: "
            "pip install umap-learn"
        ) from exc

    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `hdbscan`. Install it in thesis_env with: "
            "pip install hdbscan"
        ) from exc

    return umap, hdbscan


def load_matched_pairs() -> pd.DataFrame:
    """Load matched pairs produced by the matching step."""
    if not MATCHED_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing matched pairs: {MATCHED_PAIRS_PATH}")

    matched_pairs = pd.read_parquet(MATCHED_PAIRS_PATH)
    missing = sorted(MATCHED_PAIR_REQUIRED_COLUMNS - set(matched_pairs.columns))
    if missing:
        raise ValueError(f"matched_pairs.parquet is missing columns: {missing}")

    return matched_pairs


def create_admission_rows(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert pair-level matched data into one row per admission."""
    pair_metric_columns = [
        "cosine_similarity",
        "embedding_distance",
        "abs_elixhauser_difference",
        "quickumls_shared_term_count",
        "quickumls_jaccard",
        "match_type",
    ]
    pair_metric_columns = [
        column for column in pair_metric_columns if column in matched_pairs.columns
    ]

    mhh_columns = {
        "mhh_subject_id": "subject_id",
        "mhh_hadm_id": "hadm_id",
        "mhh_chief_complaint_normalized": "chief_complaint_normalized",
    }
    mhc0_columns = {
        "mhc0_subject_id": "subject_id",
        "mhc0_hadm_id": "hadm_id",
        "mhc0_chief_complaint_normalized": "chief_complaint_normalized",
    }

    optional_prefix_columns = [
        "chief_complaint_raw",
        "quickumls_terms",
        "quickumls_extracted_text",
        "sex",
        "age_at_admission",
        "age_bin",
        "elixhauser_score",
    ]

    for suffix in optional_prefix_columns:
        mhh_column = f"mhh_{suffix}"
        mhc0_column = f"mhc0_{suffix}"
        if mhh_column in matched_pairs.columns:
            mhh_columns[mhh_column] = suffix
        if mhc0_column in matched_pairs.columns:
            mhc0_columns[mhc0_column] = suffix

    base_columns = ["pair_id", *pair_metric_columns]
    mhh = matched_pairs[base_columns + list(mhh_columns)].rename(columns=mhh_columns)
    mhc0 = matched_pairs[base_columns + list(mhc0_columns)].rename(columns=mhc0_columns)
    mhh["cohort"] = "MHH1_psychotic"
    mhc0["cohort"] = "MHC0"

    admissions = pd.concat([mhh, mhc0], ignore_index=True)
    admissions["subject_id"] = admissions["subject_id"].astype("int64")
    admissions["hadm_id"] = admissions["hadm_id"].astype("int64")
    admissions["chief_complaint_normalized"] = (
        admissions["chief_complaint_normalized"].fillna("").astype(str)
    )
    return admissions


def load_embedding_metadata() -> pd.DataFrame:
    """Load all embedding metadata files and add each row's embedding file path."""
    metadata_frames = []
    for metadata_path in sorted(EMBEDDING_DIR.glob("*_embedding_metadata.parquet")):
        metadata = pd.read_parquet(metadata_path)
        missing = sorted(EMBEDDING_METADATA_REQUIRED_COLUMNS - set(metadata.columns))
        if missing:
            raise ValueError(f"{metadata_path.name} is missing columns: {missing}")

        embedding_path = metadata_path.with_name(
            metadata_path.name.replace("_embedding_metadata.parquet", "_embeddings.npy")
        )
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embedding matrix: {embedding_path}")

        metadata = metadata.copy()
        metadata["embedding_file"] = str(embedding_path)
        metadata_frames.append(metadata)

    if not metadata_frames:
        raise FileNotFoundError(f"No embedding metadata files found in {EMBEDDING_DIR}")

    return pd.concat(metadata_frames, ignore_index=True)


def attach_embeddings(
    admissions: pd.DataFrame, embedding_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach embedding row IDs and return metadata aligned to an embedding matrix."""
    join_columns = ["cohort", "subject_id", "hadm_id"]
    join_metadata = embedding_metadata[
        join_columns + ["embedding_row_id", "embedding_file"]
    ].copy()
    join_metadata["subject_id"] = join_metadata["subject_id"].astype("int64")
    join_metadata["hadm_id"] = join_metadata["hadm_id"].astype("int64")

    admissions = admissions.merge(
        join_metadata,
        on=join_columns,
        how="left",
        validate="many_to_one",
    )

    missing_embedding = admissions["embedding_row_id"].isna()
    if missing_embedding.any():
        missing_rows = admissions.loc[
            missing_embedding, ["cohort", "subject_id", "hadm_id"]
        ].head(10)
        raise ValueError(
            "Some matched admissions are missing embedding metadata. "
            f"First missing rows:\n{missing_rows.to_string(index=False)}"
        )

    embedding_cache: dict[str, np.ndarray] = {}
    embeddings = []
    for row in admissions.itertuples(index=False):
        embedding_file = str(row.embedding_file)
        embedding_row_id = int(row.embedding_row_id)

        if embedding_file not in embedding_cache:
            embedding_cache[embedding_file] = np.load(embedding_file)

        matrix = embedding_cache[embedding_file]
        if embedding_row_id < 0 or embedding_row_id >= len(matrix):
            raise IndexError(
                f"embedding_row_id {embedding_row_id} is outside bounds for "
                f"{embedding_file} with {len(matrix)} rows"
            )
        embeddings.append(matrix[embedding_row_id])

    admissions = admissions.reset_index(drop=True)
    admissions["admission_cluster_row_id"] = np.arange(len(admissions))
    return admissions, np.vstack(embeddings)


def hdbscan_min_cluster_size(n_admissions: int) -> int:
    """Set the HDBSCAN cluster-size floor from the requested rule."""
    return max(20, int(0.02 * n_admissions))


def run_umap_hdbscan(
    features: np.ndarray,
    route_name: str,
    umap_module: Any,
    hdbscan_module: Any,
) -> dict[str, np.ndarray]:
    """Run UMAP reduction, HDBSCAN clustering, and a separate 2D projection."""
    n_admissions = features.shape[0]
    cluster_components = min(UMAP_CLUSTER_COMPONENTS, max(2, n_admissions - 2))

    cluster_reducer = umap_module.UMAP(
        n_components=cluster_components,
        n_neighbors=min(UMAP_N_NEIGHBORS, max(2, n_admissions - 1)),
        min_dist=UMAP_CLUSTER_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    reduced_for_clustering = cluster_reducer.fit_transform(features)

    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size(n_admissions),
        min_samples=HDBSCAN_MIN_SAMPLES,
    )
    labels = clusterer.fit_predict(reduced_for_clustering)

    plot_reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=min(UMAP_N_NEIGHBORS, max(2, n_admissions - 1)),
        min_dist=UMAP_PLOT_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    projection_2d = plot_reducer.fit_transform(features)

    print(
        f"{route_name}: {len(set(labels) - {-1})} clusters, "
        f"{100.0 * np.mean(labels == -1):.1f}% noise"
    )
    return {
        "labels": labels,
        "projection_2d": projection_2d,
        "reduced_for_clustering": reduced_for_clustering,
    }


def run_bert_clustering(
    embeddings: np.ndarray, umap_module: Any, hdbscan_module: Any
) -> dict[str, np.ndarray]:
    """Cluster Bio_ClinicalBERT embeddings."""
    return run_umap_hdbscan(embeddings, "BERT route", umap_module, hdbscan_module)


def run_tfidf_clustering(
    texts: pd.Series, umap_module: Any, hdbscan_module: Any
) -> dict[str, Any]:
    """Create TF-IDF/SVD features and cluster them with UMAP + HDBSCAN."""
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        lowercase=True,
        strip_accents="unicode",
    )
    tfidf = vectorizer.fit_transform(texts.fillna("").astype(str))

    if tfidf.shape[1] < 2:
        raise ValueError(
            "TF-IDF route produced fewer than two features. "
            "Lower TFIDF_MIN_DF or check chief_complaint_normalized."
        )

    n_components = min(SVD_MAX_COMPONENTS, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    svd_features = svd.fit_transform(tfidf)

    results = run_umap_hdbscan(
        svd_features, "TF-IDF route", umap_module, hdbscan_module
    )
    results["vectorizer"] = vectorizer
    results["svd"] = svd
    results["tfidf_matrix"] = tfidf
    return results


def fit_agglomerative_cosine(n_clusters: int) -> AgglomerativeClustering:
    """Create cosine agglomerative clustering across sklearn API versions."""
    try:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="cosine",
            linkage="average",
        )


def run_agglomerative_clustering(embeddings: np.ndarray) -> dict[int, np.ndarray]:
    """Cluster Bio_ClinicalBERT embeddings for several candidate k values."""
    results = {}
    for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
        if n_clusters < 2:
            raise ValueError("Agglomerative clustering requires at least 2 clusters.")
        if n_clusters > len(embeddings):
            raise ValueError(
                f"Cannot run agglomerative k={n_clusters} with only "
                f"{len(embeddings)} admissions."
            )
        clusterer = fit_agglomerative_cosine(n_clusters)
        labels = clusterer.fit_predict(embeddings)
        results[n_clusters] = labels
        print(f"Agglomerative route k={n_clusters}: {len(set(labels))} clusters")
    return results


def split_terms(value: Any) -> list[str]:
    """Split pipe-separated QuickUMLS terms for optional summaries."""
    if pd.isna(value):
        return []
    return [term.strip() for term in str(value).split("|") if term.strip()]


def top_normalized_complaints(texts: pd.Series, n: int = TOP_COMPLAINTS_N) -> str:
    """Return the most frequent normalized complaints in one compact string."""
    counts = Counter(texts.dropna().astype(str))
    return " | ".join(f"{text} ({count})" for text, count in counts.most_common(n))


def top_text_terms(texts: pd.Series, n: int = TOP_TERMS_N) -> str:
    """Return top word n-grams within a cluster for interpretability."""
    texts = texts.fillna("").astype(str)
    non_empty = texts[texts.str.strip().ne("")]
    if len(non_empty) == 0:
        return ""

    min_df = 1 if len(non_empty) < 3 else 2
    vectorizer = CountVectorizer(
        ngram_range=(1, 3),
        min_df=min_df,
        stop_words="english",
        strip_accents="unicode",
    )
    try:
        matrix = vectorizer.fit_transform(non_empty)
    except ValueError:
        return ""

    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(counts)[::-1][:n]
    return " | ".join(f"{terms[i]} ({int(counts[i])})" for i in order)


def complaint_tokens(text: Any) -> tuple[str, ...]:
    """Tokenize a normalized chief complaint for lexical hierarchy matching."""
    return tuple(str(text or "").lower().split())


def token_ngrams(tokens: tuple[str, ...], min_n: int, max_n: int) -> set[tuple[str, ...]]:
    """Return unique contiguous token n-grams from one token tuple."""
    ngrams = set()
    upper = min(max_n, len(tokens))
    for n in range(min_n, upper + 1):
        for start in range(0, len(tokens) - n + 1):
            ngrams.add(tokens[start : start + n])
    return ngrams


def contains_token_sequence(
    child_tokens: tuple[str, ...],
    parent_tokens: tuple[str, ...],
) -> bool:
    """Return True when parent_tokens occur contiguously inside child_tokens."""
    if not parent_tokens or len(parent_tokens) > len(child_tokens):
        return False
    width = len(parent_tokens)
    return any(
        child_tokens[start : start + width] == parent_tokens
        for start in range(0, len(child_tokens) - width + 1)
    )


def is_meaningful_parent_phrase(parent_tokens: tuple[str, ...]) -> bool:
    """Filter away generic lexical parent candidates."""
    if not parent_tokens:
        return False
    if all(token in ENGLISH_STOP_WORDS for token in parent_tokens):
        return False
    if len(parent_tokens) == 1:
        return parent_tokens[0] in LEXICAL_SINGLE_TOKEN_PARENT_TERMS
    if parent_tokens[0] in ENGLISH_STOP_WORDS or parent_tokens[-1] in ENGLISH_STOP_WORDS:
        return False
    return any(token not in ENGLISH_STOP_WORDS for token in parent_tokens)


def prune_redundant_lexical_parents(
    candidates: pd.DataFrame,
    assignments: pd.DataFrame,
    overlap_threshold: float = 0.85,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop shorter lexical fragments nearly covered by longer parent phrases."""
    if candidates.empty or assignments.empty:
        return candidates, assignments

    parent_to_keys = {
        parent: set(zip(group["cohort"], group["subject_id"], group["hadm_id"]))
        for parent, group in assignments.groupby("parent_phrase")
    }
    parent_to_tokens = {
        parent: complaint_tokens(parent)
        for parent in candidates["parent_phrase"]
    }
    parent_to_exact_count = candidates.set_index("parent_phrase")[
        "parent_exact_count"
    ].to_dict()

    redundant_parents = set()
    parents = list(parent_to_tokens)
    for parent in parents:
        parent_tokens = parent_to_tokens[parent]
        parent_keys = parent_to_keys.get(parent, set())
        if not parent_keys or parent_to_exact_count.get(parent, 0) > 0:
            continue

        for longer_parent in parents:
            if parent == longer_parent:
                continue
            longer_tokens = parent_to_tokens[longer_parent]
            if len(longer_tokens) <= len(parent_tokens):
                continue
            if not contains_token_sequence(longer_tokens, parent_tokens):
                continue

            longer_keys = parent_to_keys.get(longer_parent, set())
            overlap = len(parent_keys & longer_keys) / len(parent_keys)
            if overlap >= overlap_threshold:
                redundant_parents.add(parent)
                break

    if not redundant_parents:
        return candidates, assignments

    candidates = candidates.loc[
        ~candidates["parent_phrase"].isin(redundant_parents)
    ].copy()
    assignments = assignments.loc[
        ~assignments["parent_phrase"].isin(redundant_parents)
    ].copy()
    return candidates, assignments


def build_lexical_parent_complaint_candidates(
    admissions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build frequency-aware parent complaint candidates and row assignments.

    This route extracts frequent short/core phrases that appear as exact
    complaints and/or as token subsequences inside longer complaints. It is a
    review aid for candidate workup/lab comparison groups, not a formal
    validation of individual matching.
    """
    working = admissions.copy()
    working["chief_complaint_normalized"] = (
        working["chief_complaint_normalized"].fillna("").astype(str).str.strip()
    )
    working = working.loc[working["chief_complaint_normalized"].ne("")].copy()
    working["complaint_tokens"] = working["chief_complaint_normalized"].map(
        complaint_tokens
    )

    exact_counts = Counter(working["chief_complaint_normalized"])
    complaint_to_tokens = (
        working.drop_duplicates("chief_complaint_normalized")
        .set_index("chief_complaint_normalized")["complaint_tokens"]
        .to_dict()
    )

    phrase_admission_counts: Counter[tuple[str, ...]] = Counter()
    for tokens in working["complaint_tokens"]:
        for ngram in token_ngrams(tokens, *LEXICAL_NGRAM_RANGE):
            phrase_admission_counts[ngram] += 1

    candidate_rows = []
    assignment_rows = []
    for parent_tokens, phrase_count in phrase_admission_counts.items():
        if phrase_count < LEXICAL_MIN_PARENT_FREQUENCY:
            continue
        if not is_meaningful_parent_phrase(parent_tokens):
            continue

        parent_phrase = " ".join(parent_tokens)
        child_complaints = []
        for complaint, tokens in complaint_to_tokens.items():
            if contains_token_sequence(tokens, parent_tokens):
                child_complaints.append(complaint)

        if len(child_complaints) < LEXICAL_MIN_CHILD_COMPLAINTS:
            continue
        if not any(len(complaint_to_tokens[child]) > len(parent_tokens) for child in child_complaints):
            continue

        group = working.loc[
            working["chief_complaint_normalized"].isin(child_complaints)
        ].copy()
        if group.empty:
            continue

        frequent_children = Counter(group["chief_complaint_normalized"]).most_common(
            LEXICAL_MAX_EXAMPLE_CHILDREN
        )
        exact_count = exact_counts.get(parent_phrase, 0)
        candidate_rows.append(
            {
                "parent_phrase": parent_phrase,
                "parent_exact_count": int(exact_count),
                "total_group_admissions": len(group),
                "n_unique_child_complaints": len(child_complaints),
                "n_MHH1_psychotic": int(group["cohort"].eq("MHH1_psychotic").sum()),
                "n_MHC0": int(group["cohort"].eq("MHC0").sum()),
                "pct_MHH1_psychotic": 100.0
                * group["cohort"].eq("MHH1_psychotic").mean(),
                "example_child_complaints": " | ".join(
                    sorted(child_complaints)[:LEXICAL_MAX_EXAMPLE_CHILDREN]
                ),
                "most_frequent_child_complaints": " | ".join(
                    f"{complaint} ({count})"
                    for complaint, count in frequent_children
                ),
                "suggested_manual_label": "",
                "include_for_workup_analysis": "",
                "reviewer_notes": "",
            }
        )

        for row in group.itertuples(index=False):
            assignment_rows.append(
                {
                    "parent_phrase": parent_phrase,
                    "is_exact_parent_complaint": bool(
                        row.chief_complaint_normalized == parent_phrase
                    ),
                    "pair_id": row.pair_id,
                    "cohort": row.cohort,
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "chief_complaint_normalized": row.chief_complaint_normalized,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["total_group_admissions", "parent_exact_count", "parent_phrase"],
            ascending=[False, False, True],
        )

    assignments = pd.DataFrame(assignment_rows)
    if not assignments.empty:
        assignments = assignments.sort_values(
            ["parent_phrase", "pair_id", "cohort", "subject_id", "hadm_id"]
        )

    return prune_redundant_lexical_parents(candidates, assignments)



def build_cluster_summary(
    admissions: pd.DataFrame,
    cluster_column: str,
    route_name: str,
) -> pd.DataFrame:
    """Summarize cluster size, cohort mix, and readable text labels."""
    rows = []
    for label, cluster_df in admissions.groupby(cluster_column, dropna=False):
        rows.append(
            {
                "route": route_name,
                "cluster_label": int(label),
                "is_noise": bool(label == -1),
                "n_admissions": len(cluster_df),
                "n_MHH1_psychotic": int(
                    cluster_df["cohort"].eq("MHH1_psychotic").sum()
                ),
                "n_MHC0": int(cluster_df["cohort"].eq("MHC0").sum()),
                "pct_MHH1_psychotic": 100.0
                * cluster_df["cohort"].eq("MHH1_psychotic").mean(),
                "most_frequent_normalized_chief_complaints": top_normalized_complaints(
                    cluster_df["chief_complaint_normalized"]
                ),
                "top_terms": top_text_terms(cluster_df["chief_complaint_normalized"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["is_noise", "n_admissions", "cluster_label"],
        ascending=[True, False, True],
    )


def build_pair_agreement(admissions: pd.DataFrame, routes: set[str]) -> pd.DataFrame:
    """Report whether each matched pair falls into the same non-noise cluster."""
    required = {
        "pair_id",
        "cohort",
        "cosine_similarity",
        "embedding_distance",
    }
    if "bert" in routes:
        required.add("bert_cluster")
    if "tfidf" in routes:
        required.add("tfidf_cluster")
    if "agglomerative" in routes:
        for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
            required.add(f"agglomerative_k{n_clusters}_cluster")

    missing = sorted(required - set(admissions.columns))
    if missing:
        raise ValueError(f"Cannot build pair agreement; missing columns: {missing}")

    pair_rows = []
    for pair_id, pair_df in admissions.groupby("pair_id"):
        if len(pair_df) != 2:
            raise ValueError(f"pair_id {pair_id} has {len(pair_df)} admission rows")

        by_cohort = pair_df.set_index("cohort")
        mhh = by_cohort.loc["MHH1_psychotic"]
        mhc0 = by_cohort.loc["MHC0"]

        row = {
            "pair_id": pair_id,
            "mhh_subject_id": mhh["subject_id"],
            "mhh_hadm_id": mhh["hadm_id"],
            "mhc0_subject_id": mhc0["subject_id"],
            "mhc0_hadm_id": mhc0["hadm_id"],
            "pair_cosine_similarity": mhh["cosine_similarity"],
            "pair_embedding_distance": mhh["embedding_distance"],
        }

        if "bert" in routes:
            bert_either_noise = bool(
                mhh["bert_cluster"] == -1 or mhc0["bert_cluster"] == -1
            )
            row.update(
                {
                    "bert_mhh_cluster": int(mhh["bert_cluster"]),
                    "bert_mhc0_cluster": int(mhc0["bert_cluster"]),
                    "bert_either_noise": bert_either_noise,
                    "bert_same_cluster": bool(
                        not bert_either_noise
                        and mhh["bert_cluster"] == mhc0["bert_cluster"]
                    ),
                }
            )

        if "tfidf" in routes:
            tfidf_either_noise = bool(
                mhh["tfidf_cluster"] == -1 or mhc0["tfidf_cluster"] == -1
            )
            row.update(
                {
                    "tfidf_mhh_cluster": int(mhh["tfidf_cluster"]),
                    "tfidf_mhc0_cluster": int(mhc0["tfidf_cluster"]),
                    "tfidf_either_noise": tfidf_either_noise,
                    "tfidf_same_cluster": bool(
                        not tfidf_either_noise
                        and mhh["tfidf_cluster"] == mhc0["tfidf_cluster"]
                    ),
                }
            )

        if "agglomerative" in routes:
            for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
                cluster_column = f"agglomerative_k{n_clusters}_cluster"
                mhh_cluster = int(mhh[cluster_column])
                mhc0_cluster = int(mhc0[cluster_column])
                row.update(
                    {
                        f"agglomerative_k{n_clusters}_mhh_cluster": mhh_cluster,
                        f"agglomerative_k{n_clusters}_mhc0_cluster": mhc0_cluster,
                        f"agglomerative_k{n_clusters}_same_cluster": bool(
                            mhh_cluster == mhc0_cluster
                        ),
                    }
                )

        pair_rows.append(row)

    return pd.DataFrame(pair_rows)


def percent_true(values: pd.Series) -> float:
    """Convert a boolean series mean to a percent."""
    if len(values) == 0:
        return 0.0
    return 100.0 * values.mean()


def build_overall_summary(
    admissions: pd.DataFrame, pair_agreement: pd.DataFrame, routes: set[str]
) -> pd.DataFrame:
    """Create one-row summary of clustering QC results."""
    summary = {
        "n_admissions": len(admissions),
        "n_matched_pairs": pair_agreement["pair_id"].nunique(),
        "median_pair_cosine_similarity": pair_agreement[
            "pair_cosine_similarity"
        ].median(),
    }

    if "bert" in routes:
        bert_non_noise = admissions["bert_cluster"].ne(-1)
        summary.update(
            {
                "bert_n_clusters_excluding_noise": admissions.loc[
                    bert_non_noise, "bert_cluster"
                ].nunique(),
                "bert_pct_noise": 100.0 * admissions["bert_cluster"].eq(-1).mean(),
                "bert_pct_pairs_same_non_noise_cluster": percent_true(
                    pair_agreement["bert_same_cluster"]
                ),
                "bert_pct_pairs_either_noise": percent_true(
                    pair_agreement["bert_either_noise"]
                ),
            }
        )

    if "tfidf" in routes:
        tfidf_non_noise = admissions["tfidf_cluster"].ne(-1)
        summary.update(
            {
                "tfidf_n_clusters_excluding_noise": admissions.loc[
                    tfidf_non_noise, "tfidf_cluster"
                ].nunique(),
                "tfidf_pct_noise": 100.0 * admissions["tfidf_cluster"].eq(-1).mean(),
                "tfidf_pct_pairs_same_non_noise_cluster": percent_true(
                    pair_agreement["tfidf_same_cluster"]
                ),
                "tfidf_pct_pairs_either_noise": percent_true(
                    pair_agreement["tfidf_either_noise"]
                ),
            }
        )
    if "agglomerative" in routes:
        for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
            same_column = f"agglomerative_k{n_clusters}_same_cluster"
            summary[f"agglomerative_k{n_clusters}_pct_pairs_same_cluster"] = (
                percent_true(pair_agreement[same_column])
            )
    return pd.DataFrame([summary])


def add_clustering_outputs_to_admissions(
    admissions: pd.DataFrame,
    bert_results: dict[str, np.ndarray] | None,
    tfidf_results: dict[str, Any] | None,
    agglomerative_results: dict[int, np.ndarray] | None,
) -> pd.DataFrame:
    """Attach cluster labels and 2D projection coordinates to admission rows."""
    admissions = admissions.copy()
    if bert_results is not None:
        admissions["bert_cluster"] = bert_results["labels"]
        admissions["bert_umap_x"] = bert_results["projection_2d"][:, 0]
        admissions["bert_umap_y"] = bert_results["projection_2d"][:, 1]
    if tfidf_results is not None:
        admissions["tfidf_cluster"] = tfidf_results["labels"]
        admissions["tfidf_umap_x"] = tfidf_results["projection_2d"][:, 0]
        admissions["tfidf_umap_y"] = tfidf_results["projection_2d"][:, 1]
    if agglomerative_results is not None:
        for n_clusters, labels in agglomerative_results.items():
            admissions[f"agglomerative_k{n_clusters}_cluster"] = labels
    return admissions


def maybe_write_umap_plots(admissions: pd.DataFrame, routes: set[str]) -> list[Path]:
    """Write optional UMAP scatter plots if matplotlib imports cleanly."""
    route_plot_columns = {
        "bert": ("bert_umap_x", "bert_umap_y", "bert_cluster"),
        "tfidf": ("tfidf_umap_x", "tfidf_umap_y", "tfidf_cluster"),
    }
    plot_routes = routes & set(route_plot_columns)
    if not plot_routes:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping UMAP plots because matplotlib is unavailable: {exc}")
        return []

    plot_paths = []
    for route in sorted(plot_routes):
        x_column, y_column, cluster_column = route_plot_columns[route]
        fig, ax = plt.subplots(figsize=(9, 7))

        if DRAW_PAIR_LINES:
            for _, pair_df in admissions.groupby("pair_id", sort=False):
                if len(pair_df) != 2:
                    continue
                ax.plot(
                    pair_df[x_column],
                    pair_df[y_column],
                    color="black",
                    alpha=PAIR_LINE_ALPHA,
                    linewidth=PAIR_LINE_WIDTH,
                    zorder=1,
                )

        scatter = ax.scatter(
            admissions[x_column],
            admissions[y_column],
            c=admissions[cluster_column],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            cmap="tab20",
            zorder=2,
        )
        ax.set_title(f"{route.upper()} UMAP projection by HDBSCAN cluster")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        fig.colorbar(scatter, ax=ax, label="cluster")
        fig.tight_layout()

        path = OUTPUT_DIR / f"{route}_umap_clusters.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        plot_paths.append(path)

    return plot_paths


def build_bert_different_cluster_normalized_pairs(
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Return normalized pairs in different non-noise BERT clusters.

    Noise cases are exported separately by `build_bert_noise_admissions` and
    `build_bert_noise_pairs`, so this review file stays focused on pairs where
    both admissions received a concrete HDBSCAN cluster label.
    """
    required_columns = {
        "pair_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "chief_complaint_normalized",
        "bert_cluster",
    }
    missing = sorted(required_columns - set(admissions.columns))
    if missing:
        raise ValueError(
            "Cannot build BERT different-cluster review; missing columns: "
            f"{missing}"
        )

    rows = []
    for pair_id, pair_df in admissions.groupby("pair_id", sort=True):
        if len(pair_df) != 2:
            continue
        by_cohort = pair_df.set_index("cohort")
        if "MHH1_psychotic" not in by_cohort.index or "MHC0" not in by_cohort.index:
            continue

        mhh = by_cohort.loc["MHH1_psychotic"]
        mhc0 = by_cohort.loc["MHC0"]
        mhh_cluster = int(mhh["bert_cluster"])
        mhc0_cluster = int(mhc0["bert_cluster"])
        if mhh_cluster == -1 or mhc0_cluster == -1:
            continue
        if mhh_cluster == mhc0_cluster:
            continue

        rows.append(
            {
                "pair_id": pair_id,
                "mhh_subject_id": mhh["subject_id"],
                "mhh_hadm_id": mhh["hadm_id"],
                "mhh_bert_cluster": mhh_cluster,
                "mhh_chief_complaint_normalized": mhh[
                    "chief_complaint_normalized"
                ],
                "mhc0_subject_id": mhc0["subject_id"],
                "mhc0_hadm_id": mhc0["hadm_id"],
                "mhc0_bert_cluster": mhc0_cluster,
                "mhc0_chief_complaint_normalized": mhc0[
                    "chief_complaint_normalized"
                ],
            }
        )

    return pd.DataFrame(rows)


def build_bert_noise_admissions(admissions: pd.DataFrame) -> pd.DataFrame:
    """Return admission-level rows assigned to BERT HDBSCAN noise."""
    required_columns = {
        "pair_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "chief_complaint_normalized",
        "bert_cluster",
    }
    missing = sorted(required_columns - set(admissions.columns))
    if missing:
        raise ValueError(
            "Cannot build BERT noise admissions review; missing columns: "
            f"{missing}"
        )

    columns = [
        "pair_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "bert_cluster",
        "chief_complaint_normalized",
    ]
    optional_columns = [
        "chief_complaint_raw",
        "quickumls_terms",
        "quickumls_extracted_text",
        "cosine_similarity",
        "embedding_distance",
    ]
    columns.extend(column for column in optional_columns if column in admissions.columns)
    return (
        admissions.loc[admissions["bert_cluster"].eq(-1), columns]
        .sort_values(["pair_id", "cohort"])
        .reset_index(drop=True)
    )


def build_bert_noise_pairs(admissions: pd.DataFrame) -> pd.DataFrame:
    """Return matched pairs where either admission is BERT HDBSCAN noise."""
    rows = []
    for pair_id, pair_df in admissions.groupby("pair_id", sort=True):
        if len(pair_df) != 2:
            continue
        by_cohort = pair_df.set_index("cohort")
        if "MHH1_psychotic" not in by_cohort.index or "MHC0" not in by_cohort.index:
            continue

        mhh = by_cohort.loc["MHH1_psychotic"]
        mhc0 = by_cohort.loc["MHC0"]
        mhh_cluster = int(mhh["bert_cluster"])
        mhc0_cluster = int(mhc0["bert_cluster"])
        if mhh_cluster != -1 and mhc0_cluster != -1:
            continue

        rows.append(
            {
                "pair_id": pair_id,
                "mhh_subject_id": mhh["subject_id"],
                "mhh_hadm_id": mhh["hadm_id"],
                "mhh_bert_cluster": mhh_cluster,
                "mhh_chief_complaint_normalized": mhh[
                    "chief_complaint_normalized"
                ],
                "mhc0_subject_id": mhc0["subject_id"],
                "mhc0_hadm_id": mhc0["hadm_id"],
                "mhc0_bert_cluster": mhc0_cluster,
                "mhc0_chief_complaint_normalized": mhc0[
                    "chief_complaint_normalized"
                ],
                "bert_both_noise": bool(mhh_cluster == -1 and mhc0_cluster == -1),
            }
        )

    return pd.DataFrame(rows)


def build_agglomerative_pair_agreement_summary(
    admissions: pd.DataFrame,
    pair_agreement: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize pair agreement and cohort balance for each agglomerative k."""
    rows = []
    for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
        cluster_column = f"agglomerative_k{n_clusters}_cluster"
        same_column = f"agglomerative_k{n_clusters}_same_cluster"
        cluster_sizes = admissions.groupby(cluster_column).size()
        cohort_counts = (
            admissions.groupby([cluster_column, "cohort"]).size().unstack(fill_value=0)
        )
        has_mhh = cohort_counts.get("MHH1_psychotic", 0).gt(0)
        has_mhc0 = cohort_counts.get("MHC0", 0).gt(0)
        has_both = has_mhh & has_mhc0

        rows.append(
            {
                "k": n_clusters,
                "n_clusters": admissions[cluster_column].nunique(),
                "n_admissions": len(admissions),
                "n_matched_pairs": pair_agreement["pair_id"].nunique(),
                "pct_pairs_same_cluster": percent_true(pair_agreement[same_column]),
                "median_cluster_size": cluster_sizes.median(),
                "min_cluster_size": cluster_sizes.min(),
                "max_cluster_size": cluster_sizes.max(),
                "n_clusters_with_both_cohorts": int(has_both.sum()),
                "pct_clusters_with_both_cohorts": 100.0 * has_both.mean(),
                "n_clusters_mhh_only": int((has_mhh & ~has_mhc0).sum()),
                "n_clusters_mhc0_only": int((~has_mhh & has_mhc0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_agglomerative_cluster_review_candidates(
    cluster_summaries: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create review-friendly agglomerative cluster rows for manual labeling."""
    review_frames = []
    for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
        route_name = f"agglomerative_k{n_clusters}"
        summary = cluster_summaries.get(route_name)
        if summary is None:
            continue
        review = summary.loc[
            :,
            [
                "cluster_label",
                "n_admissions",
                "n_MHH1_psychotic",
                "n_MHC0",
                "most_frequent_normalized_chief_complaints",
                "top_terms",
            ],
        ].copy()
        review.insert(0, "k", n_clusters)
        review["suggested_manual_label"] = ""
        review["include_for_workup_analysis"] = ""
        review["reviewer_notes"] = ""
        review_frames.append(review)

    if not review_frames:
        return pd.DataFrame()
    return pd.concat(review_frames, ignore_index=True).sort_values(
        ["k", "n_admissions", "cluster_label"],
        ascending=[True, False, True],
    )


def build_agglomerative_different_cluster_pairs(
    admissions: pd.DataFrame,
    n_clusters: int,
) -> pd.DataFrame:
    """Return matched pairs separated by an agglomerative clustering solution."""
    cluster_column = f"agglomerative_k{n_clusters}_cluster"
    rows = []
    for pair_id, pair_df in admissions.groupby("pair_id", sort=True):
        if len(pair_df) != 2:
            continue
        by_cohort = pair_df.set_index("cohort")
        if "MHH1_psychotic" not in by_cohort.index or "MHC0" not in by_cohort.index:
            continue

        mhh = by_cohort.loc["MHH1_psychotic"]
        mhc0 = by_cohort.loc["MHC0"]
        mhh_cluster = int(mhh[cluster_column])
        mhc0_cluster = int(mhc0[cluster_column])
        if mhh_cluster == mhc0_cluster:
            continue

        rows.append(
            {
                "pair_id": pair_id,
                "cosine_similarity": mhh.get("cosine_similarity"),
                "embedding_distance": mhh.get("embedding_distance"),
                "mhh_subject_id": mhh["subject_id"],
                "mhh_hadm_id": mhh["hadm_id"],
                "mhh_chief_complaint_normalized": mhh[
                    "chief_complaint_normalized"
                ],
                "mhh_cluster": mhh_cluster,
                "mhc0_subject_id": mhc0["subject_id"],
                "mhc0_hadm_id": mhc0["hadm_id"],
                "mhc0_chief_complaint_normalized": mhc0[
                    "chief_complaint_normalized"
                ],
                "mhc0_cluster": mhc0_cluster,
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    admissions: pd.DataFrame,
    pair_agreement: pd.DataFrame,
    cluster_summaries: dict[str, pd.DataFrame],
    overall_summary: pd.DataFrame,
    routes: set[str],
    lexical_parent_candidates: pd.DataFrame | None = None,
    lexical_admission_assignments: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write all clustering QC outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for route in {"bert", "tfidf"} - routes:
        for stale_path in [
            OUTPUT_DIR / f"{route}_cluster_summary.csv",
            OUTPUT_DIR / f"{route}_umap_clusters.png",
        ]:
            stale_path.unlink(missing_ok=True)

    outputs = {
        "matched_admissions_with_clusters": OUTPUT_DIR
        / "matched_admissions_with_clusters.parquet",
        "matched_admissions_with_clusters_csv": OUTPUT_DIR
        / "matched_admissions_with_clusters.csv",
        "pair_cluster_agreement": OUTPUT_DIR / "pair_cluster_agreement.csv",
        "overall_clustering_summary": OUTPUT_DIR / "overall_clustering_summary.csv",
    }
    for route in routes:
        if route in {"bert", "tfidf"}:
            outputs[f"{route}_cluster_summary"] = OUTPUT_DIR / f"{route}_cluster_summary.csv"
    if "bert" in routes:
        outputs["bert_different_cluster_normalized_pairs"] = (
            OUTPUT_DIR / "bert_different_cluster_normalized_pairs.csv"
        )
        outputs["bert_noise_admissions"] = OUTPUT_DIR / "bert_noise_admissions.csv"
        outputs["bert_noise_pairs"] = OUTPUT_DIR / "bert_noise_pairs.csv"
    if "agglomerative" in routes:
        outputs["agglomerative_pair_agreement_summary"] = (
            OUTPUT_DIR / "agglomerative_pair_agreement_summary.csv"
        )
        outputs["agglomerative_cluster_review_candidates"] = (
            OUTPUT_DIR / "agglomerative_cluster_review_candidates.csv"
        )
        for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
            outputs[f"agglomerative_cluster_summary_k{n_clusters}"] = (
                OUTPUT_DIR / f"agglomerative_cluster_summary_k{n_clusters}.csv"
            )
            outputs[f"agglomerative_different_cluster_pairs_k{n_clusters}"] = (
                OUTPUT_DIR / f"agglomerative_different_cluster_pairs_k{n_clusters}.csv"
            )
    if "lexical_hierarchy" in routes:
        outputs["lexical_parent_complaint_candidates"] = (
            OUTPUT_DIR / "lexical_parent_complaint_candidates.csv"
        )
        outputs["lexical_parent_complaint_admission_assignments"] = (
            OUTPUT_DIR / "lexical_parent_complaint_admission_assignments.csv"
        )

    admissions.to_parquet(outputs["matched_admissions_with_clusters"], index=False)
    admissions.to_csv(outputs["matched_admissions_with_clusters_csv"], index=False)
    pair_agreement.to_csv(outputs["pair_cluster_agreement"], index=False)
    for route, summary in cluster_summaries.items():
        if route in {"bert", "tfidf"}:
            summary.to_csv(outputs[f"{route}_cluster_summary"], index=False)
        elif route.startswith("agglomerative_k"):
            n_clusters = route.removeprefix("agglomerative_k")
            summary.to_csv(
                outputs[f"agglomerative_cluster_summary_k{n_clusters}"],
                index=False,
            )
    if "bert" in routes:
        build_bert_different_cluster_normalized_pairs(admissions).to_csv(
            outputs["bert_different_cluster_normalized_pairs"],
            index=False,
        )
        build_bert_noise_admissions(admissions).to_csv(
            outputs["bert_noise_admissions"],
            index=False,
        )
        build_bert_noise_pairs(admissions).to_csv(
            outputs["bert_noise_pairs"],
            index=False,
        )
    if "agglomerative" in routes:
        build_agglomerative_pair_agreement_summary(admissions, pair_agreement).to_csv(
            outputs["agglomerative_pair_agreement_summary"],
            index=False,
        )
        build_agglomerative_cluster_review_candidates(cluster_summaries).to_csv(
            outputs["agglomerative_cluster_review_candidates"],
            index=False,
        )
        for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
            build_agglomerative_different_cluster_pairs(admissions, n_clusters).to_csv(
                outputs[f"agglomerative_different_cluster_pairs_k{n_clusters}"],
                index=False,
            )
    if "lexical_hierarchy" in routes:
        if lexical_parent_candidates is None or lexical_admission_assignments is None:
            raise ValueError("Lexical hierarchy outputs were requested but not built.")
        lexical_parent_candidates.to_csv(
            outputs["lexical_parent_complaint_candidates"],
            index=False,
        )
        lexical_admission_assignments.to_csv(
            outputs["lexical_parent_complaint_admission_assignments"],
            index=False,
        )
    overall_summary.to_csv(outputs["overall_clustering_summary"], index=False)

    for path in maybe_write_umap_plots(admissions, routes):
        outputs[path.stem] = path

    return outputs


def main() -> None:
    """Run post-matching clustering QC and save local inspection outputs."""
    args = parse_args()
    route_name = normalize_route_value(args.route)
    routes = selected_routes(args.route)
    umap_module = None
    hdbscan_module = None
    if routes & {"bert", "tfidf"}:
        umap_module, hdbscan_module = require_clustering_dependencies()

    matched_pairs = load_matched_pairs()
    admissions = create_admission_rows(matched_pairs)
    embedding_metadata = load_embedding_metadata()
    admissions, embeddings = attach_embeddings(admissions, embedding_metadata)

    print(f"Loaded {len(admissions)} admissions from {len(matched_pairs)} pairs")
    print(f"Embedding matrix for matched admissions: {embeddings.shape}")
    print(f"Selected clustering route: {route_name}")

    bert_results = None
    tfidf_results = None
    agglomerative_results = None
    lexical_parent_candidates = None
    lexical_admission_assignments = None
    cluster_summaries = {}

    if "bert" in routes:
        assert umap_module is not None and hdbscan_module is not None
        bert_results = run_bert_clustering(embeddings, umap_module, hdbscan_module)

    if "tfidf" in routes:
        assert umap_module is not None and hdbscan_module is not None
        tfidf_results = run_tfidf_clustering(
            admissions["chief_complaint_normalized"],
            umap_module,
            hdbscan_module,
        )

    if "agglomerative" in routes:
        agglomerative_results = run_agglomerative_clustering(embeddings)

    if "lexical_hierarchy" in routes:
        lexical_parent_candidates, lexical_admission_assignments = (
            build_lexical_parent_complaint_candidates(admissions)
        )
        print(
            "Lexical hierarchy route: "
            f"{len(lexical_parent_candidates):,} parent candidates, "
            f"{len(lexical_admission_assignments):,} admission-parent assignments"
        )

    admissions = add_clustering_outputs_to_admissions(
        admissions,
        bert_results,
        tfidf_results,
        agglomerative_results,
    )
    pair_agreement = build_pair_agreement(admissions, routes)

    if "bert" in routes:
        cluster_summaries["bert"] = build_cluster_summary(
            admissions, "bert_cluster", "bert"
        )
    if "tfidf" in routes:
        cluster_summaries["tfidf"] = build_cluster_summary(
            admissions, "tfidf_cluster", "tfidf"
        )
    if "agglomerative" in routes:
        for n_clusters in AGGLOMERATIVE_N_CLUSTERS_LIST:
            route_key = f"agglomerative_k{n_clusters}"
            cluster_summaries[route_key] = build_cluster_summary(
                admissions,
                f"{route_key}_cluster",
                route_key,
            ).drop(columns=["is_noise"])

    overall_summary = build_overall_summary(admissions, pair_agreement, routes)

    outputs = write_outputs(
        admissions,
        pair_agreement,
        cluster_summaries,
        overall_summary,
        routes,
        lexical_parent_candidates,
        lexical_admission_assignments,
    )

    print("\n=== Overall Clustering QC Summary ===")
    print(overall_summary.to_string(index=False))
    if "agglomerative" in routes:
        print("\n=== Agglomerative Pair Agreement Summary ===")
        print(
            build_agglomerative_pair_agreement_summary(
                admissions,
                pair_agreement,
            ).to_string(index=False)
        )
    if "lexical_hierarchy" in routes and lexical_parent_candidates is not None:
        print("\n=== Top Lexical Parent Complaint Candidates ===")
        print(
            lexical_parent_candidates.head(20).loc[
                :,
                [
                    "parent_phrase",
                    "parent_exact_count",
                    "total_group_admissions",
                    "n_unique_child_complaints",
                    "n_MHH1_psychotic",
                    "n_MHC0",
                ],
            ].to_string(index=False)
        )
    print("\nSaved clustering QC outputs:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()

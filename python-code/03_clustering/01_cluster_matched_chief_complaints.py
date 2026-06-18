"""Cluster matched chief-complaint admissions for post-matching quality control.

This script is a validation and interpretability step only. It does not create,
modify, or filter the matched cohort. The matched cohort is already defined by
`02_cohort_matching/03_match_chief_complaint_cohorts.py`.

The script converts matched pairs into one row per admission, attaches the
already-computed Bio_ClinicalBERT embedding for each admission, and runs two
independent clustering routes:

1. BERT route:
   - cluster matched-admission BERT embeddings with UMAP + HDBSCAN
   - create a 2D UMAP projection for plotting

2. TF-IDF route:
   - vectorize normalized chief complaints with word n-gram TF-IDF
   - reduce with TruncatedSVD
   - cluster with UMAP + HDBSCAN
   - create a 2D UMAP projection for plotting

Outputs are local QC artifacts under:
    03_clustering/chief_complaint_cluster_qc_output/
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MATCHED_PAIRS_PATH = (
    PROJECT_DIR / "02_cohort_matching" / "matched_cohort_output" / "matched_pairs.parquet"
)
EMBEDDING_DIR = (
    PROJECT_DIR
    / "01_admission_notes"
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
        choices=["1", "2", "3", "bert", "tfidf", "both"],
        help=(
            "Clustering route to run. Use 1/bert for BERT, 2/tfidf for TF-IDF, "
            "or 3/both for both routes. Default comes from CLUSTERING_ROUTE."
        ),
    )
    return parser.parse_args()


def normalize_route_value(route_value: str) -> str:
    """Map numeric route choices to readable route names."""
    route = route_value.strip().lower()
    numeric_map = {"1": "bert", "2": "tfidf", "3": "both"}
    return numeric_map.get(route, route)


def selected_routes(route_value: str) -> set[str]:
    """Return the configured clustering routes."""
    route = normalize_route_value(route_value)
    if route == "both":
        return {"bert", "tfidf"}
    if route in {"bert", "tfidf"}:
        return {route}
    raise ValueError(
        "CLUSTERING_ROUTE must be one of: 'bert', 'tfidf', or 'both'. "
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
    return pd.DataFrame([summary])


def add_clustering_outputs_to_admissions(
    admissions: pd.DataFrame,
    bert_results: dict[str, np.ndarray] | None,
    tfidf_results: dict[str, Any] | None,
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
    return admissions


def maybe_write_umap_plots(admissions: pd.DataFrame, routes: set[str]) -> list[Path]:
    """Write optional UMAP scatter plots if matplotlib imports cleanly."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping UMAP plots because matplotlib is unavailable: {exc}")
        return []

    plot_paths = []
    route_plot_columns = {
        "bert": ("bert_umap_x", "bert_umap_y", "bert_cluster"),
        "tfidf": ("tfidf_umap_x", "tfidf_umap_y", "tfidf_cluster"),
    }
    for route in sorted(routes):
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


def write_outputs(
    admissions: pd.DataFrame,
    pair_agreement: pd.DataFrame,
    cluster_summaries: dict[str, pd.DataFrame],
    overall_summary: pd.DataFrame,
    routes: set[str],
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
        outputs[f"{route}_cluster_summary"] = OUTPUT_DIR / f"{route}_cluster_summary.csv"
    if "bert" in routes:
        outputs["bert_different_cluster_normalized_pairs"] = (
            OUTPUT_DIR / "bert_different_cluster_normalized_pairs.csv"
        )
        outputs["bert_noise_admissions"] = OUTPUT_DIR / "bert_noise_admissions.csv"
        outputs["bert_noise_pairs"] = OUTPUT_DIR / "bert_noise_pairs.csv"

    admissions.to_parquet(outputs["matched_admissions_with_clusters"], index=False)
    admissions.to_csv(outputs["matched_admissions_with_clusters_csv"], index=False)
    pair_agreement.to_csv(outputs["pair_cluster_agreement"], index=False)
    for route, summary in cluster_summaries.items():
        summary.to_csv(outputs[f"{route}_cluster_summary"], index=False)
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
    overall_summary.to_csv(outputs["overall_clustering_summary"], index=False)

    for path in maybe_write_umap_plots(admissions, routes):
        outputs[path.stem] = path

    return outputs


def main() -> None:
    """Run post-matching clustering QC and save local inspection outputs."""
    args = parse_args()
    route_name = normalize_route_value(args.route)
    routes = selected_routes(args.route)
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
    cluster_summaries = {}

    if "bert" in routes:
        bert_results = run_bert_clustering(embeddings, umap_module, hdbscan_module)

    if "tfidf" in routes:
        tfidf_results = run_tfidf_clustering(
            admissions["chief_complaint_normalized"],
            umap_module,
            hdbscan_module,
        )

    admissions = add_clustering_outputs_to_admissions(
        admissions, bert_results, tfidf_results
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

    overall_summary = build_overall_summary(admissions, pair_agreement, routes)

    outputs = write_outputs(
        admissions,
        pair_agreement,
        cluster_summaries,
        overall_summary,
        routes,
    )

    print("\n=== Overall Clustering QC Summary ===")
    print(overall_summary.to_string(index=False))
    print("\nSaved clustering QC outputs:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()

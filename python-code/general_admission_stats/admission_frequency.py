from __future__ import annotations

from pathlib import Path
from math import erfc, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "admission_frequency_output"
INPUT_PATH = OUTPUT_DIR / "number_of_admission_counts.csv"

GROUP_LABELS = {
    "Case (MHC=1 same-admission only)": "MHC1-sa",
    "Control (MHC=0)": "MHC0",
}

COLORS = {
    "MHC1-sa": "#ad78f4",
    "MHC0": "#55b468",
}

GROUP_ORDER = ["MHC1-sa", "MHC0"]


def normal_two_sided_p(z_score: float) -> float:
    return erfc(abs(z_score) / sqrt(2))


def format_p_value(p_value: float) -> str:
    if p_value == 0:
        return "<1e-300"
    if p_value < 0.001:
        return f"{p_value:.3e}"
    return f"{p_value:.6f}"


def plot_admission_histogram(
    df: pd.DataFrame,
    *,
    title: str,
    bins: range | int,
    output_name: str,
    max_admissions: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for group in GROUP_ORDER:
        values = df.loc[df["group_label"] == group, "n_admissions"]
        ax.hist(
            values,
            bins=bins,
            alpha=0.55,
            label=f"{group} (n={len(values):,})",
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.8,
        )

    if max_admissions is not None:
        ax.set_xlim(0.5, max_admissions + 0.5)

    ax.set_title(title)
    ax.set_xlabel("Admissions per subject")
    ax.set_ylabel("Number of subjects")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close(fig)


def gaussian_kde(values: pd.Series, grid: np.ndarray) -> np.ndarray:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.zeros_like(grid)

    sd = values.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0

    bandwidth = max(0.35, 1.06 * sd * len(values) ** (-1 / 5))
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * scaled**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def integer_bins(max_value: int) -> np.ndarray:
    return np.arange(0.5, max_value + 1.5, 1)


def plot_relative_histogram_with_density(
    df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    max_admissions: int | None = None,
) -> None:
    max_value = int(df["n_admissions"].max())
    bins = integer_bins(max_value)
    grid = np.linspace(1, max_value, max(250, max_value * 10))

    fig, ax = plt.subplots(figsize=(8, 5))

    for group in GROUP_ORDER:
        values = df.loc[df["group_label"] == group, "n_admissions"]
        weights = np.full(len(values), 100 / len(values))
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            alpha=0.35,
            label=f"{group} histogram (n={len(values):,})",
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.plot(
            grid,
            gaussian_kde(values, grid) * 100,
            color=COLORS[group],
            linewidth=2.2,
            label=f"{group} smoothed density",
        )

    if max_admissions is not None:
        ax.set_xlim(0.5, max_admissions + 0.5)

    ax.set_title(title)
    ax.set_xlabel("Admissions per subject")
    ax.set_ylabel("Subjects (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close(fig)


def plot_ecdf(
    df: pd.DataFrame,
    *,
    title: str,
    output_name: str,
    max_admissions: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for group in GROUP_ORDER:
        values = np.sort(df.loc[df["group_label"] == group, "n_admissions"].to_numpy())
        cumulative_pct = np.arange(1, len(values) + 1) / len(values) * 100
        ax.step(
            values,
            cumulative_pct,
            where="post",
            color=COLORS[group],
            linewidth=2,
            label=f"{group} (n={len(values):,})",
        )

    if max_admissions is not None:
        ax.set_xlim(0.5, max_admissions + 0.5)

    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_xlabel("Admissions per subject")
    ax.set_ylabel("Cumulative subjects (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close(fig)


def group_summary(values: pd.Series, group: str) -> dict[str, float | str]:
    return {
        "comparison": "group_summary",
        "group": group,
        "n_subjects": len(values),
        "mean": values.mean(),
        "sd": values.std(ddof=1),
        "q1": values.quantile(0.25),
        "median": values.median(),
        "q3": values.quantile(0.75),
        "iqr": values.quantile(0.75) - values.quantile(0.25),
        "min": values.min(),
        "max": values.max(),
        "statistic": np.nan,
        "p_value": np.nan,
        "p_value_text": "",
        "interpretation": "",
    }


def mann_whitney_u_test(x: pd.Series, y: pd.Series) -> dict[str, float | str]:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    n_x = len(x_values)
    n_y = len(y_values)

    combined = pd.Series(np.concatenate([x_values, y_values]))
    ranks = combined.rank(method="average").to_numpy()
    rank_sum_x = ranks[:n_x].sum()
    u_x = rank_sum_x - n_x * (n_x + 1) / 2
    mean_u = n_x * n_y / 2

    tie_counts = combined.value_counts().to_numpy()
    n_total = n_x + n_y
    tie_term = np.sum(tie_counts**3 - tie_counts)
    variance_u = n_x * n_y / 12 * ((n_total + 1) - tie_term / (n_total * (n_total - 1)))
    z_score = (u_x - mean_u) / np.sqrt(variance_u)
    p_value = normal_two_sided_p(z_score)

    common_language_effect = u_x / (n_x * n_y)
    cliffs_delta = 2 * common_language_effect - 1

    return {
        "comparison": "Mann-Whitney U",
        "group": "MHC1-sa vs MHC0",
        "n_subjects": np.nan,
        "mean": np.nan,
        "sd": np.nan,
        "q1": np.nan,
        "median": np.nan,
        "q3": np.nan,
        "iqr": np.nan,
        "min": np.nan,
        "max": np.nan,
        "statistic": u_x,
        "p_value": p_value,
        "p_value_text": format_p_value(p_value),
        "interpretation": (
            f"Common-language effect={common_language_effect:.4f}; "
            f"Cliff's delta={cliffs_delta:.4f}"
        ),
    }


def ks_test(x: pd.Series, y: pd.Series) -> dict[str, float | str]:
    x_values = np.sort(x.to_numpy(dtype=float))
    y_values = np.sort(y.to_numpy(dtype=float))
    grid = np.sort(np.unique(np.concatenate([x_values, y_values])))

    cdf_x = np.searchsorted(x_values, grid, side="right") / len(x_values)
    cdf_y = np.searchsorted(y_values, grid, side="right") / len(y_values)
    d_stat = np.max(np.abs(cdf_x - cdf_y))

    effective_n = len(x_values) * len(y_values) / (len(x_values) + len(y_values))
    lambda_value = (np.sqrt(effective_n) + 0.12 + 0.11 / np.sqrt(effective_n)) * d_stat
    p_value = 2 * sum(
        (-1) ** (k - 1) * np.exp(-2 * k * k * lambda_value * lambda_value)
        for k in range(1, 101)
    )
    p_value = float(min(max(p_value, 0), 1))

    return {
        "comparison": "Kolmogorov-Smirnov",
        "group": "MHC1-sa vs MHC0",
        "n_subjects": np.nan,
        "mean": np.nan,
        "sd": np.nan,
        "q1": np.nan,
        "median": np.nan,
        "q3": np.nan,
        "iqr": np.nan,
        "min": np.nan,
        "max": np.nan,
        "statistic": d_stat,
        "p_value": p_value,
        "p_value_text": format_p_value(p_value),
        "interpretation": "Maximum absolute difference between cumulative distributions",
    }


def mean_difference_test(x: pd.Series, y: pd.Series) -> dict[str, float | str]:
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    z_score = diff / se
    p_value = normal_two_sided_p(z_score)

    return {
        "comparison": "Mean difference",
        "group": "MHC1-sa minus MHC0",
        "n_subjects": np.nan,
        "mean": diff,
        "sd": np.nan,
        "q1": np.nan,
        "median": np.nan,
        "q3": np.nan,
        "iqr": np.nan,
        "min": np.nan,
        "max": np.nan,
        "statistic": z_score,
        "p_value": p_value,
        "p_value_text": format_p_value(p_value),
        "interpretation": f"Mean difference={diff:.4f} admissions per subject",
    }


def write_distribution_tests(df: pd.DataFrame) -> None:
    x = df.loc[df["group_label"] == "MHC1-sa", "n_admissions"]
    y = df.loc[df["group_label"] == "MHC0", "n_admissions"]

    rows = [
        group_summary(x, "MHC1-sa"),
        group_summary(y, "MHC0"),
        mann_whitney_u_test(x, y),
        ks_test(x, y),
        mean_difference_test(x, y),
    ]
    results = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "admission_distribution_tests.csv", index=False)

    print("\n=== Admission count distribution tests ===")
    print(results.to_string(index=False))


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["group_label"] = df["group_name"].replace(GROUP_LABELS)

    plot_admission_histogram(
        df,
        title="Admissions per subject",
        bins=range(1, int(df["n_admissions"].max()) + 2),
        output_name="admissions_per_subject_full.png",
    )

    plot_admission_histogram(
        df,
        title="Admissions per subject, zoomed to <= 15",
        bins=range(1, 17),
        output_name="admissions_per_subject_zoom_15.png",
        max_admissions=15,
    )

    plot_relative_histogram_with_density(
        df,
        title="Admissions per subject, relative frequency",
        output_name="admissions_per_subject_relative_density.png",
    )

    plot_relative_histogram_with_density(
        df,
        title="Admissions per subject, relative frequency, zoomed to <= 15",
        output_name="admissions_per_subject_relative_density_zoom_15.png",
        max_admissions=15,
    )

    plot_ecdf(
        df,
        title="Admissions per subject, cumulative distribution",
        output_name="admissions_per_subject_ecdf.png",
    )

    plot_ecdf(
        df,
        title="Admissions per subject, cumulative distribution, zoomed to <= 15",
        output_name="admissions_per_subject_ecdf_zoom_15.png",
        max_admissions=15,
    )

    write_distribution_tests(df)


if __name__ == "__main__":
    main()

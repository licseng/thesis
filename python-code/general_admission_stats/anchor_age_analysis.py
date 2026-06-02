from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "anchor_age_analysis_output"
INPUT_PATH = OUTPUT_DIR / "anchor_age_patient_level.csv"
OUTPUT_PATH = OUTPUT_DIR / "anchor_age_distribution_tests.csv"

GROUP_LABELS = {
    "Case (MHC1-sa)": "MHC1-sa",
    "Control (MHC0)": "MHC0",
}


def normal_two_sided_p(z_score: float) -> float:
    return erfc(abs(z_score) / sqrt(2))


def format_p_value(p_value: float) -> str:
    if p_value == 0:
        return "<1e-300"
    if p_value < 0.001:
        return f"{p_value:.3e}"
    return f"{p_value:.6f}"


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
        "interpretation": f"Mean difference={diff:.4f} years",
    }


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["group_label"] = df["group_name"].replace(GROUP_LABELS)
    df = df.dropna(subset=["group_label", "anchor_age"])

    x = df.loc[df["group_label"] == "MHC1-sa", "anchor_age"]
    y = df.loc[df["group_label"] == "MHC0", "anchor_age"]

    results = pd.DataFrame(
        [
            group_summary(x, "MHC1-sa"),
            group_summary(y, "MHC0"),
            mann_whitney_u_test(x, y),
            ks_test(x, y),
            mean_difference_test(x, y),
        ]
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Anchor age distribution tests ===")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

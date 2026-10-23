"""Compare saved prompt-A and prompt-B diagnostic-overshadowing runs offline.

Reports agreement, Cohen's kappa, label-specific overlap, and disagreements
on the frozen 100-admission sample. No API client or credentials are used.
"""
from pathlib import Path
import argparse
import hashlib
import json

import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = HERE / "prefilter_output/diagnostic_overshadowing_admission_input.parquet"
LABELS = ["positive", "negative", "unclear"]
COMPARISON_N = 100


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison_paths(seed):
    stem = HERE / "prefilter_output" / f"prompt_comparison_sample_{COMPARISON_N}_seed_{seed}"
    return stem.with_suffix(".csv"), stem.with_suffix(".json")


def prompt_output(version, n, seed, shared):
    scope = "comparison" if shared else "pilot"
    return HERE / f"output_prompt_{version}_{scope}_{n}_seed_{seed}"


def compare_saved_prompts(seed):
    """Compare local labels only; never loads keys or calls the API."""
    _, sample_manifest = comparison_paths(seed)
    sample = json.loads(sample_manifest.read_text())
    expected = sample["selected_input_ids"]
    if sample["input_sha256"] != sha256(INPUT):
        raise ValueError("Shared sample does not match current prefilter input")
    tables, manifests = {}, {}
    for version in ["A", "B"]:
        folder = prompt_output(version, COMPARISON_N, seed, True)
        path = folder / "diagnostic_overshadowing_admission_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Prompt {version} has no saved results; classification has not been run")
        manifests[version] = json.loads((folder / "run_manifest.json").read_text())
        run = manifests[version]
        if (run["selected_input_ids"] != expected or run["input_sha256"] != sample["input_sha256"]
                or run["prompt_version"] != version):
            raise ValueError("Prompt outputs were not generated on the same frozen sample")
        tables[version] = pd.read_csv(path)
        table = tables[version]
        label_column = "diagnostic_overshadowing_label" if version == "A" else "psychiatric_overshadowing_label"
        # Preserve rejected labels separately; agreement is not evidence validation.
        table["accepted_label"] = table[label_column]
        table["comparison_label"] = table["model_reported_label"]
        if (table.classification_input_id.duplicated().any()
                or not set(table.classification_input_id).issubset(expected)
                or not table.comparison_label.isin(LABELS).all()
                or not table.validation_status.isin(["valid", "invalid"]).all()
                or not table.loc[table.validation_status.eq("valid"), "accepted_label"].isin(LABELS).all()
                or table.loc[table.validation_status.eq("invalid"), "accepted_label"].notna().any()):
            raise ValueError("Invalid or mismatched saved classification results")
    for field in ["model", "api_url", "max_output_tokens", "temperature", "thinking_enabled", "chunking", "classification_unit"]:
        if manifests["A"][field] != manifests["B"][field]:
            raise ValueError(f"Prompt runs differ on non-prompt setting: {field}")
    combined = tables["A"].merge(tables["B"], on="classification_input_id",
        how="outer", suffixes=("_A", "_B"), validate="one_to_one", indicator=True)
    both = combined[combined._merge.eq("both")].copy()
    for key in ["cohort", "subject_id", "hadm_id", "note_id", "pair_id"]:
        if not both[f"{key}_A"].eq(both[f"{key}_B"]).all():
            raise ValueError("Admission metadata differ across prompt outputs")
    left, right = both.comparison_label_A, both.comparison_label_B
    valid_both = both.validation_status_A.eq("valid") & both.validation_status_B.eq("valid")
    matrix = pd.crosstab(left, right).reindex(index=LABELS, columns=LABELS, fill_value=0)
    matrix.index.name = "prompt_A_label"; matrix.columns.name = "prompt_B_label"
    n = len(both)
    agreement = float((left == right).mean()) if n else None
    chance = float((matrix.sum(axis=1)/n * matrix.sum(axis=0)/n).sum()) if n else None
    kappa = (agreement-chance)/(1-chance) if n and chance < 1 else None
    per_label = []
    for label in LABELS:
        a, b = left.eq(label), right.eq(label)
        intersection, union = int((a & b).sum()), int((a | b).sum())
        denominator = int(a.sum()+b.sum())
        per_label.append({"label": label, "n_A": int(a.sum()), "n_B": int(b.sum()),
            "n_both": intersection, "jaccard": intersection/union if union else None,
            "label_specific_agreement": 2*intersection/denominator if denominator else None})
    summary = {
        "n_expected_admissions": len(expected), "n_results_A": len(tables["A"]),
        "n_results_B": len(tables["B"]), "n_paired_results": n,
        "n_missing_A": len(expected)-len(tables["A"]),
        "n_missing_B": len(expected)-len(tables["B"]),
        "complete_100_admission_comparison": n == COMPARISON_N,
        "n_agree": int((left == right).sum()), "n_disagree": int((left != right).sum()),
        "agreement_percent": 100*agreement if n else None, "cohens_kappa": kappa,
        "n_invalid_A": int(tables["A"].validation_status.eq("invalid").sum()),
        "n_invalid_B": int(tables["B"].validation_status.eq("invalid").sum()),
        "n_both_validation_passed": int(valid_both.sum()),
        "agreement_percent_both_validation_passed": float((left[valid_both] == right[valid_both]).mean()*100) if valid_both.any() else None,
        "interpretation": "Agreement of model-reported labels, including unvalidated responses; not accuracy or human validation. Accepted labels and validation status are preserved separately.",
    }
    out = HERE / f"output_prompt_comparison_{COMPARISON_N}_seed_{seed}"
    out.mkdir(exist_ok=True)
    both.drop(columns="_merge").to_csv(out / "admission_prompt_comparison.csv", index=False)
    both.loc[left != right].drop(columns="_merge").to_csv(out / "disagreement_review.csv", index=False)
    matrix.to_csv(out / "label_confusion_matrix.csv")
    pd.DataFrame(per_label).to_csv(out / "label_specific_overlap.csv", index=False)
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "comparison_manifest.json").write_text(json.dumps({
        "sample": sample, "run_A": manifests["A"], "run_B": manifests["B"]}, indent=2))
    print(json.dumps(summary, indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    compare_saved_prompts(args.seed)


if __name__ == "__main__":
    main()

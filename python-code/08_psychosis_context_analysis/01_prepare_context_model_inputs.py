#!/usr/bin/env python3
"""Prepare exploratory admission-level context comparisons; no note text is read.

Reuses the current matched cohort and numerical outcome/covariate exports.
MHC1 subtypes can vary across admissions for the same patient.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent
OUTPUT = HERE / "analysis_output"
KEY = ["pair_id", "cohort", "subject_id", "hadm_id"]
COV = ["age_at_admission_per_10y", "elixhauser_score_per_5pt",
       "log1p_prior_all_admissions"]
LANG = PYTHON / "03_discharge_note_text_analysis/01_language_complexity"


def main():
    common_path = PYTHON / "02_cohort_matching/_matched_cohort_characterization_common.py"
    spec = importlib.util.spec_from_file_location("context_common", common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    pairs_path = common.MATCHED_PAIRS_PATH
    pairs = pd.read_csv(pairs_path)
    frames = []
    for cohort, prefix in [("MHC0", "mhc0"), ("MHC1_psychotic", "mhc1")]:
        frames.append(pd.DataFrame({
            "pair_id": pairs.pair_id, "cohort": cohort,
            "subject_id": pairs[f"{prefix}_subject_id"],
            "hadm_id": pairs[f"{prefix}_hadm_id"]}))
    base = pd.concat(frames, ignore_index=True)
    assert not base.hadm_id.duplicated().any()
    assert set(base[base.cohort == "MHC0"].subject_id).isdisjoint(
        base[base.cohort == "MHC1_psychotic"].subject_id)
    context = common.load_required_table("psychosis_context")
    flags = ["has_prior_psychosis", "has_current_secondary_psychosis"]
    context = context[KEY + flags]
    assert not context.hadm_id.duplicated().any()
    cases = base[base.cohort == "MHC1_psychotic"]
    assert set(context.hadm_id) == set(cases.hadm_id), "Stale/mismatched context export"
    base = base.merge(context, on=KEY, how="left", validate="one_to_one")
    for flag in flags:
        assert base.loc[base.cohort == "MHC1_psychotic", flag].isin([0, 1]).all()
    case_mask = base.cohort == "MHC1_psychotic"
    assert (base.loc[case_mask, flags].sum(axis=1) > 0).all()
    base["context_group"] = "MHC0"
    base.loc[case_mask & (base.has_current_secondary_psychosis == 0),
             "context_group"] = "history_only"
    base.loc[case_mask & (base.has_current_secondary_psychosis == 1),
             "context_group"] = "current_with_or_without_history"
    cov_path = LANG / "analysis_output_language_complexity_inference/language_complexity_model_covariates.csv"
    cov = pd.read_csv(cov_path, usecols=KEY + COV)
    assert set(cov.hadm_id) == set(base.hadm_id)
    base = base.merge(cov, on=KEY, validate="one_to_one")
    assert np.isfinite(base[COV].to_numpy()).all()
    section_path = LANG / "analysis_output_language_complexity/language_complexity_prose_section_metrics.csv"
    sections = pd.read_csv(section_path, usecols=[
        "cohort", "subject_id", "hadm_id", "section_name", "flesch_reading_ease"])
    instructions = sections[sections.section_name == "discharge_instructions"].copy()
    assert not instructions.hadm_id.duplicated().any()
    readability = base.merge(instructions.drop(columns="section_name"),
        on=KEY[1:], how="left", validate="one_to_one")
    readability["outcome"] = "discharge_instructions_flesch_reading_ease"
    readability["value"] = readability.flesch_reading_ease
    stay_path = PYTHON / "04_clinical_route_analyses/analysis_output_whole_matched_stay_mortality/model_inputs.csv"
    stay = pd.read_csv(stay_path)
    stay = stay[stay.outcome == "hospital_los_days"].copy()
    if stay.empty:
        # Accept the established primary script's LOS outcome label.
        available = pd.read_csv(stay_path, usecols=["outcome"]).outcome.unique()
        raise ValueError(f"Check LOS outcome label; available labels: {available}")
    assert not stay.hadm_id.duplicated().any()
    los = base.merge(stay[KEY + COV + ["value"]], on=KEY, how="left",
                     validate="one_to_one", suffixes=("", "_stay"))
    for col in COV:
        valid = los.value.notna()
        assert np.allclose(los.loc[valid, col], los.loc[valid, col + "_stay"])
    los["outcome"] = "hospital_los_days"
    OUTPUT.mkdir(exist_ok=True)
    base.groupby("context_group").agg(n_admissions=("hadm_id", "size"),
        n_subjects=("subject_id", "nunique")).to_csv(OUTPUT / "group_counts.csv")
    membership = base[base.cohort == "MHC1_psychotic"].groupby("subject_id").context_group.nunique()
    pd.DataFrame([{"n_MHC1_subjects_in_both_subgroups": int((membership == 2).sum())}]).to_csv(
        OUTPUT / "subject_overlap.csv", index=False)
    retained, coverage, descriptions = [], [], []
    for data in [readability, los]:
        eligible = np.isfinite(data.value)
        if data.outcome.iloc[0] == "hospital_los_days":
            eligible &= data.value >= 0
        complete = data.loc[eligible].groupby("pair_id").size()
        keep = complete.index[complete == 2]
        data = data[data.pair_id.isin(keep)].copy()
        assert data.groupby("pair_id").size().eq(2).all()
        retained.append(data[KEY + ["context_group", "outcome", "value"] + COV])
        coverage.append({"outcome": data.outcome.iloc[0],
            "n_admissions": len(data), "n_complete_pairs": len(keep),
            "excluded_pairs": len(pairs) - len(keep)})
        for group, d in data.groupby("context_group"):
            descriptions.append({"outcome": d.outcome.iloc[0], "context_group": group,
                "n_admissions": len(d), "n_subjects": d.subject_id.nunique(),
                "mean": d.value.mean(), "sd": d.value.std(),
                "median": d.value.median(), "q1": d.value.quantile(.25),
                "q3": d.value.quantile(.75)})
    pd.concat(retained).to_csv(OUTPUT / "model_inputs.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUTPUT / "coverage.csv", index=False)
    pd.DataFrame(descriptions).to_csv(OUTPUT / "descriptives.csv", index=False)
    paths = [pairs_path, cov_path, section_path, stay_path]
    provenance = {str(path.relative_to(PYTHON)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths}
    provenance["context_table_sha256"] = hashlib.sha256(
        context.sort_values(KEY).to_csv(index=False).encode()).hexdigest()
    (OUTPUT / "input_manifest.json").write_text(json.dumps(provenance, indent=2))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()

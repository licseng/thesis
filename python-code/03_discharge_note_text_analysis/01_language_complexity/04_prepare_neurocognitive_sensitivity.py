"""Prepare admission-specific neurocognitive context using existing ICD categories.

Presence means any prior coded diagnosis or current secondary diagnosis, with
the same mutually exclusive category rules as cohort characterization. Future
diagnoses do not count. MHC0 has no psychiatric codes by cohort definition.
No clinical free text is read or exported by this script.
"""
from pathlib import Path
import sys
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "02_cohort_matching"))
import _matched_cohort_characterization_common as common

OUTPUT_DIR = SCRIPT_DIR / "analysis_output_neurocognitive_sensitivity"


def main():
    covariates = pd.read_csv(SCRIPT_DIR / "analysis_output_language_complexity_inference" / "language_complexity_model_covariates.csv")
    history = common.load_required_table("subject_diagnosis_history")
    history["category"] = history.apply(common.classify_psychiatric_icd, axis=1)
    history = history.loc[history.category.eq("neurocognitive")].copy()
    history["admittime"] = pd.to_datetime(history.admittime, errors="raise")
    history["seq_num"] = pd.to_numeric(history.seq_num, errors="raise")
    context = common.load_required_table("psychosis_context")
    targets = covariates.loc[covariates.cohort.eq("MHC1_psychotic"), ["subject_id", "hadm_id"]].merge(
        context[["subject_id", "hadm_id", "admittime"]],
        on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    targets["admittime"] = pd.to_datetime(targets.admittime, errors="raise")
    if targets.admittime.isna().any():
        raise ValueError("Missing admission dates for matched MHC1 admissions")
    joined = targets.rename(columns={"hadm_id": "target_hadm_id", "admittime": "target_admittime"}).merge(
        history[["subject_id", "hadm_id", "admittime", "seq_num"]], on="subject_id", how="left")
    joined["neurocognitive"] = (
        joined.admittime.lt(joined.target_admittime)
        | (joined.hadm_id.eq(joined.target_hadm_id) & joined.seq_num.gt(1)))
    flags = joined.groupby(["subject_id", "target_hadm_id"], as_index=False).neurocognitive.max().rename(columns={"target_hadm_id": "hadm_id"})
    covariates = covariates.merge(flags, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    if covariates.loc[covariates.cohort.eq("MHC1_psychotic"), "neurocognitive"].isna().any():
        raise ValueError("Missing MHC1 neurocognitive flags")
    covariates["neurocognitive"] = covariates.neurocognitive.fillna(False).astype(int)
    if covariates.loc[covariates.cohort.eq("MHC0"), "neurocognitive"].any():
        raise ValueError("Patient overlap between MHC0 and MHC1")
    OUTPUT_DIR.mkdir(exist_ok=True)
    covariates.to_csv(OUTPUT_DIR / "neurocognitive_model_covariates.csv", index=False)
    summary = covariates.groupby("cohort").agg(n_admissions=("hadm_id", "size"), n_neurocognitive=("neurocognitive", "sum"))
    summary.to_csv(OUTPUT_DIR / "neurocognitive_coverage.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()

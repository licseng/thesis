#!/usr/bin/env python3
"""Prepare exploratory admission-level context comparisons; no note text is read.

Reuses the current matched cohort and numerical outcome/covariate exports.
MHC1 subtypes can vary across admissions for the same patient.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent
OUTPUT = HERE / "analysis_output"
KEY = ["pair_id", "cohort", "subject_id", "hadm_id"]
COV = ["age_at_admission_per_10y", "elixhauser_score_per_5pt",
       "log1p_prior_all_admissions"]
LANG = PYTHON / "03_discharge_note_text_analysis/01_language_complexity"


def summarize_context_balance(base, context, common):
    """Describe admission-level balance between the two psychosis contexts."""
    groups = ["history_only", "current_with_or_without_history"]
    cases = base.loc[base.context_group.isin(groups)].copy()
    categories = common.matched_MHC1_admission_comorbidity_categories(
        context, common.load_required_table("subject_diagnosis_history"))
    counts = categories.groupby(["subject_id", "hadm_id"]).size().rename("psychiatric_categories")
    cases = cases.merge(counts, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    cases["psychiatric_categories"] = cases.psychiatric_categories.fillna(0).astype(int)
    rows = []
    for measure in ["age_at_admission", "elixhauser_score", "psychiatric_categories"]:
        values = [cases.loc[cases.context_group.eq(group), measure] for group in groups]
        history, current = values
        pooled_sd = np.sqrt((history.var() + current.var()) / 2)
        smd = (current.mean() - history.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        for group, x in zip(groups, values):
            rows.append(dict(measure=measure, context_group=group, n_admissions=int(x.notna().sum()),
                n_patients=cases.loc[x.dropna().index, "subject_id"].nunique(),
                mean=x.mean(), sd=x.std(), median=x.median(), q1=x.quantile(.25), q3=x.quantile(.75),
                smd_current_minus_history=smd))
    out = pd.DataFrame(rows)
    OUTPUT.mkdir(exist_ok=True)
    out.to_csv(OUTPUT / "context_balance.csv", index=False)
    counts = pd.crosstab(cases.psychiatric_categories, cases.context_group)
    counts.to_csv(OUTPUT / "context_comorbidity_count_distribution.csv")
    print(out.to_string(index=False))
    print(counts.to_string())


def summarize_overshadowing(base):
    """Compare detection yields and accepted-label positivity within MHC1 contexts."""
    import statsmodels.api as sm

    source_path = PYTHON / ("06_classification/01_classification_diag_overshadowing/"
        "output_prompt_B_semantic_verification_gemma/semantic_judgement_summary/"
        "combined_labels_with_verification_status.csv")
    labels = pd.read_csv(source_path, keep_default_na=False,
        usecols=KEY + ["semantic_accepted_label", "semantic_verdict", "supporting_excerpt_status"])
    if labels.hadm_id.duplicated().any() or not labels.semantic_accepted_label.isin(["positive", "negative", "unclear", ""]).all():
        raise ValueError("Duplicate admissions or unexpected overshadowing labels")
    cases = base.loc[base.cohort.eq("MHC1_psychotic"), KEY + ["context_group",
        "has_prior_psychosis", "has_current_secondary_psychosis"]].copy()
    if not set(labels.hadm_id).issubset(set(cases.hadm_id)):
        raise ValueError("Classification admissions are outside the current matched context cohort")
    joined = cases.merge(labels, on=KEY, how="left", validate="one_to_one", indicator=True)
    if joined._merge.eq("both").sum() != len(labels):
        raise ValueError("Classification metadata do not match context membership")
    joined["selected_for_classification"] = joined._merge.eq("both")
    joined = joined.drop(columns="_merge")
    rows = []
    for group, data in joined.groupby("context_group"):
        selected = data.selected_for_classification
        accepted = selected & data.semantic_accepted_label.isin(["positive", "negative", "unclear"])
        positive = selected & data.semantic_accepted_label.eq("positive")
        n_selected, n_accepted, n_positive = int(selected.sum()), int(accepted.sum()), int(positive.sum())
        rows.append({"context_group": group, "n_matched_admissions": len(data),
            "n_matched_subjects": data.subject_id.nunique(), "n_classified_admissions": n_selected,
            "n_not_selected_by_keyword_prefilter": int((~selected).sum()),
            "classification_coverage_pct": 100 * n_selected / len(data),
            "n_admissions_with_accepted_label": n_accepted,
            "n_unresolved_admissions": int((selected & ~accepted).sum()),
            "n_positive_admissions": n_positive,
            "n_negative_admissions": int((selected & data.semantic_accepted_label.eq("negative")).sum()),
            "n_unclear_admissions": int((selected & data.semantic_accepted_label.eq("unclear")).sum()),
            "positive_pct_among_all_matched_admissions": 100 * n_positive / len(data),
            "positive_pct_among_accepted_labels": 100 * n_positive / n_accepted if n_accepted else np.nan,
            "positive_pct_among_selected_admissions": 100 * n_positive / n_selected if n_selected else np.nan,
            "n_selected_subjects": data.loc[selected, "subject_id"].nunique(),
            "n_subjects_with_positive_admission": data.loc[positive, "subject_id"].nunique()})
    summary = pd.DataFrame(rows)
    assert summary.n_positive_admissions.sum() == labels.semantic_accepted_label.eq("positive").sum()
    assert summary.n_classified_admissions.sum() == len(labels)
    assert (summary.n_admissions_with_accepted_label + summary.n_unresolved_admissions).eq(summary.n_classified_admissions).all()
    out = OUTPUT / "diagnostic_overshadowing"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "context_overshadowing_summary.csv", index=False)
    joined.to_csv(out / "context_overshadowing_admissions.csv", index=False)
    comparisons = []
    for sample, data in [
        ("all_matched_detection_yield", joined),
        ("accepted_positive_or_negative", joined.loc[
            joined.semantic_accepted_label.isin(["positive", "negative"])]),
    ]:
        positive = data.semantic_accepted_label.eq("positive").astype(int)
        current = data.context_group.eq("current_with_or_without_history").astype(int)
        if pd.crosstab(current, positive).shape != (2, 2) or (pd.crosstab(current, positive) == 0).any().any():
            raise ValueError(f"Both psychosis groups need positive and nonpositive observations: {sample}")
        fit = sm.GLM(positive, sm.add_constant(current.rename("current")),
                     family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": data.subject_id})
        if not fit.converged or not np.isfinite(fit.bse["current"]) or fit.bse["current"] <= 0:
            raise ValueError(f"Invalid patient-clustered logistic fit: {sample}")
        low, high = np.exp(fit.conf_int().loc["current"])
        history_count = int((current == 0).sum())
        current_count = int(current.sum())
        history_positive = int(positive[current == 0].sum())
        current_positive = int(positive[current == 1].sum())
        comparisons.append(dict(sample=sample, n_patients=data.subject_id.nunique(),
            history_n=history_count, history_positive=history_positive,
            current_n=current_count, current_positive=current_positive,
            difference_pp=100*(current_positive/current_count-history_positive/history_count),
            odds_ratio=np.exp(fit.params["current"]), ci_low=low, ci_high=high,
            p_value=fit.pvalues["current"]))
    results = pd.DataFrame(comparisons)
    results.to_csv(out / "context_overshadowing_group_comparison.csv", index=False)
    manifest = {"classification_source": str(source_path.relative_to(PYTHON)),
        "classification_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "context_membership_sha256": hashlib.sha256(cases.sort_values(KEY).to_csv(index=False).encode()).hexdigest(),
        "label_column": "semantic_accepted_label", "semantic_verifier": "google/gemma-4-31B-it-qat-w4a16-ct",
        "denominator": "Whole-cohort percentage uses all matched MHC1-psychosis admissions in each context group; classified-subset percentages are retained separately",
        "not_selected": "Admissions not selected by psychiatric keyword prefilter have no classification and are not counted as negative",
        "scope": "Admission-level logistic regression, current versus history-only, with patient-clustered standard errors and no covariate adjustment.",
        "inference": "Whole-cohort detection yield compares receiving an accepted positive classification versus all other workflow outcomes. Sensitivity analysis compares positives versus negatives among accepted binary labels. Both p-values are raw; the sensitivity analysis repeats the same contrast on a restricted sample."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    print(results.to_string(index=False))
    print(f"Saved context-specific overshadowing ratios to {out}")


def prepare_clinical_activity(base, common):
    """Whole matched cohort; reuse established inpatient event/time definitions.

    Only identifiers and timestamps are queried. NULL lab admission links are
    recovered only against an unambiguous window among ALL MIMIC admissions.
    No microbiology date-only records are assigned artificial event times.
    """
    out = OUTPUT / "clinical_activity"
    out.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(common.DB_PATH), read_only=True) as con:
        con.register("context_base", base[KEY + ["context_group"] + COV])
        con.execute("""CREATE TEMP TABLE selected AS
            SELECT b.*,a.admittime,a.dischtime
            FROM context_base b JOIN admissions a USING(subject_id,hadm_id)""")
        selected = con.execute("SELECT * FROM selected").fetchdf()
        assert len(selected) == len(base)
        valid = selected.dischtime.gt(selected.admittime)
        keep = selected.assign(valid=valid).groupby("pair_id").valid.all()
        timed = selected[selected.pair_id.isin(keep[keep].index)].copy()
        assert timed.groupby("pair_id").size().eq(2).all()
        con.register("timed", timed)
        con.execute("""CREATE TEMP TABLE candidates AS
            SELECT t.hadm_id,t.subject_id,l.labevent_id,l.specimen_id,l.charttime
            FROM timed t JOIN labevents l ON t.subject_id=l.subject_id
            AND l.hadm_id IS NULL AND l.charttime>=t.admittime
            AND l.charttime<t.dischtime""")
        con.execute("""CREATE TEMP TABLE unique_windows AS
            SELECT c.labevent_id,count(DISTINCT a.hadm_id) AS n_windows
            FROM (SELECT DISTINCT labevent_id,subject_id,charttime FROM candidates) c
            JOIN admissions a ON c.subject_id=a.subject_id
            AND c.charttime>=a.admittime AND c.charttime<a.dischtime
            GROUP BY c.labevent_id""")
        con.execute("""CREATE TEMP TABLE corrected_labs AS
            SELECT t.hadm_id,t.subject_id,l.labevent_id,l.specimen_id,l.charttime,
                'existing' AS linkage
            FROM timed t JOIN labevents l USING(subject_id,hadm_id)
            UNION ALL
            SELECT c.*,'recovered' AS linkage FROM candidates c
            JOIN unique_windows u USING(labevent_id) WHERE u.n_windows=1""")
        assert con.execute("SELECT count(*)=count(DISTINCT labevent_id) FROM corrected_labs").fetchone()[0]
        con.execute("""CREATE TEMP TABLE events AS
            SELECT hadm_id,charttime,'laboratory_measurements' AS measure
            FROM corrected_labs
            UNION ALL
            SELECT hadm_id,min(charttime),'laboratory_specimens'
            FROM corrected_labs WHERE specimen_id IS NOT NULL
            GROUP BY hadm_id,specimen_id
            UNION ALL
            SELECT m.hadm_id,m.charttime,'microbiology_records'
            FROM microbiologyevents m JOIN timed t USING(subject_id,hadm_id)
            UNION ALL
            SELECT p.hadm_id,p.ordertime,'provider_order_activity'
            FROM poe p JOIN timed t USING(subject_id,hadm_id)""")
        audit = con.execute("""SELECT measure,
            CASE WHEN charttime IS NULL THEN 'missing_timestamp'
            WHEN charttime<admittime THEN 'before_admission'
            WHEN charttime>=dischtime THEN 'at_or_after_discharge'
            ELSE 'inpatient' END AS scope,count(*) AS n_records
            FROM events JOIN timed USING(hadm_id) GROUP BY ALL""").fetchdf()
        audit.to_csv(out / "timestamp_audit.csv", index=False)
        recovery = con.execute("""SELECT
            (SELECT count(DISTINCT labevent_id) FROM candidates) AS candidate_lab_rows,
            (SELECT count(*) FROM unique_windows WHERE n_windows>1) AS ambiguous_lab_rows,
            count(*) AS recovered_lab_rows,count(DISTINCT hadm_id) AS recovered_admissions
            FROM corrected_labs WHERE linkage='recovered'""").fetchdf()
        recovery.to_csv(out / "lab_recovery_audit.csv", index=False)
        bins = [("0–24h",0,24),("24–48h",24,48),("48–72h",48,72),
                ("days 4–7",72,168),("after day 7",168,np.inf)]
        frames = []
        hours = (timed.dischtime-timed.admittime).dt.total_seconds()/3600
        for label, low, high in bins:
            frame = timed[KEY + ["context_group"] + COV].copy()
            frame["interval"] = label
            frame["exposure_days"] = (np.minimum(hours,high)-low).clip(lower=0)/24
            frames.append(frame[frame.exposure_days>0])
        windows = pd.concat(frames,ignore_index=True)
        con.execute("""CREATE TEMP TABLE event_counts AS
            SELECT e.hadm_id,e.measure,
            CASE WHEN epoch(e.charttime-t.admittime)/3600<24 THEN '0–24h'
            WHEN epoch(e.charttime-t.admittime)/3600<48 THEN '24–48h'
            WHEN epoch(e.charttime-t.admittime)/3600<72 THEN '48–72h'
            WHEN epoch(e.charttime-t.admittime)/3600<168 THEN 'days 4–7'
            ELSE 'after day 7' END AS interval,count(*) AS n_events
            FROM events e JOIN timed t USING(hadm_id)
            WHERE e.charttime>=t.admittime AND e.charttime<t.dischtime
            GROUP BY ALL""")
        counts = con.execute("SELECT * FROM event_counts").fetchdf()
    measures = ["laboratory_measurements","laboratory_specimens",
                "microbiology_records","provider_order_activity"]
    inputs = windows.merge(pd.DataFrame({"measure":measures}),how="cross").merge(
        counts,on=["hadm_id","measure","interval"],how="left",validate="one_to_one")
    inputs["n_events"] = inputs.n_events.fillna(0).astype(int)
    assert inputs.n_events.sum() == counts.n_events.sum()
    exposure = windows.groupby("hadm_id").exposure_days.sum()
    indexed = timed.set_index("hadm_id")
    duration = (indexed.dischtime-indexed.admittime).dt.total_seconds()/86400
    assert np.allclose(exposure.sort_index(),duration.sort_index())
    # Verify the overlapping admissions reproduce the established CC analysis.
    old_path = PYTHON / "04_clinical_route_analyses/analysis_output_lab_linkage_and_timing/clinical_event_admission_intervals.csv"
    old = pd.read_csv(old_path)
    old = old[old.exposure_days>0]
    check = old.merge(inputs,on=["hadm_id","measure","interval"],suffixes=("_old","_new"),
                      validate="one_to_one")
    assert len(check)==len(old), "Prior CC sample not fully reproduced"
    assert (check.n_events_old==check.n_events_new).all()
    assert np.allclose(check.exposure_days_old,check.exposure_days_new)
    inputs.to_csv(out / "model_inputs.csv",index=False)
    whole = inputs.groupby(KEY + ["context_group","measure"],as_index=False).agg(
        n_events=("n_events","sum"),exposure_days=("exposure_days","sum"))
    summary = whole.groupby(["context_group","measure"]).agg(
        n_admissions=("hadm_id","size"),n_patients=("subject_id","nunique"),
        total_events=("n_events","sum"),mean_count=("n_events","mean"),
        median_count=("n_events","median"),observed_days=("exposure_days","sum"))
    summary["pooled_events_per_day"] = summary.total_events/summary.observed_days
    summary.to_csv(out / "descriptives.csv")
    pd.DataFrame([dict(n_admissions=len(timed),n_pairs=timed.pair_id.nunique(),
        excluded_pairs=len(keep)-keep.sum(),prior_CC_intervals_verified=len(check))]).to_csv(
        out / "coverage.csv",index=False)
    (out / "input_sha256.txt").write_text(hashlib.sha256(
        (out / "model_inputs.csv").read_bytes()).hexdigest())
    print(f"Clinical activity: {len(timed)} admissions, {timed.pair_id.nunique()} pairs; "
          f"{len(check)} prior CC interval counts reproduced.",flush=True)


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
    if "--balance" in sys.argv:
        comorbidity_context = context[KEY + ["admittime"]].copy()
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
    if "--overshadowing" in sys.argv:
        summarize_overshadowing(base)
        return
    cov_path = LANG / "analysis_output_language_complexity_inference/language_complexity_model_covariates.csv"
    metrics = ["age_at_admission", "elixhauser_score"] if "--balance" in sys.argv else []
    cov = pd.read_csv(cov_path, usecols=KEY + COV + metrics)
    assert set(cov.hadm_id) == set(base.hadm_id)
    base = base.merge(cov, on=KEY, validate="one_to_one")
    assert np.isfinite(base[COV].to_numpy()).all()
    if "--balance" in sys.argv:
        summarize_context_balance(base, comorbidity_context, common)
        return
    if "--clinical-events" in sys.argv:
        prepare_clinical_activity(base, common)
        return
    section_path = LANG / "analysis_output_language_complexity/language_complexity_prose_section_metrics.csv"
    readability_metrics = ["flesch_reading_ease", "automated_readability_index"]
    sections = pd.read_csv(section_path, usecols=[
        *KEY[1:], "section_name", *readability_metrics])
    instructions = sections[sections.section_name == "discharge_instructions"].copy()
    assert not instructions.hadm_id.duplicated().any()
    readability = base.merge(instructions.drop(columns="section_name"),
        on=KEY[1:], how="left", validate="one_to_one")
    note_path = LANG / "analysis_output_language_complexity/language_complexity_note_level_metrics.csv"
    notes = pd.read_csv(note_path, usecols=KEY[1:] + readability_metrics)
    assert not notes.hadm_id.duplicated().any()
    assert set(notes.hadm_id).issubset(set(base.hadm_id))
    whole_note = base.merge(notes, on=KEY[1:], how="left", validate="one_to_one")
    outcomes = [
        data.assign(outcome=f"{scope}_{metric}", value=data[metric])
        for scope, data in [("discharge_instructions", readability), ("whole_note", whole_note)]
        for metric in readability_metrics
    ]
    stay_path = PYTHON / "04_clinical_route_analyses/analysis_output_whole_matched_stay_mortality/model_inputs.csv"
    stay = pd.read_csv(stay_path)
    mortality = ["in_hospital_mortality", "post_discharge_death_within_1y"]
    for outcome in ["hospital_los_days", "ed_los_hours", *mortality]:
        endpoint = stay[stay.outcome == outcome].copy()
        if endpoint.empty:
            raise ValueError(f"Missing hospital outcome: {outcome}")
        assert not endpoint.hadm_id.duplicated().any()
        assert set(endpoint.hadm_id).issubset(set(base.hadm_id))
        joined = base.merge(endpoint[KEY + COV + ["value"]], on=KEY, how="left",
                            validate="one_to_one", suffixes=("", "_stay"))
        if joined.value.notna().sum() != len(endpoint):
            raise ValueError(f"Hospital endpoint rows do not match the context cohort: {outcome}")
        for col in COV:
            valid = joined.value.notna()
            assert np.allclose(joined.loc[valid, col], joined.loc[valid, col + "_stay"])
        joined["outcome"] = outcome
        outcomes.append(joined)
    OUTPUT.mkdir(exist_ok=True)
    base.groupby("context_group").agg(n_admissions=("hadm_id", "size"),
        n_subjects=("subject_id", "nunique")).to_csv(OUTPUT / "group_counts.csv")
    membership = base[base.cohort == "MHC1_psychotic"].groupby("subject_id").context_group.nunique()
    pd.DataFrame([{"n_MHC1_subjects_in_both_subgroups": int((membership == 2).sum())}]).to_csv(
        OUTPUT / "subject_overlap.csv", index=False)
    retained, coverage, descriptions = [], [], []
    for data in outcomes:
        outcome = data.outcome.iloc[0]
        eligible = np.isfinite(data.value)
        if outcome in ["hospital_los_days", "ed_los_hours"]:
            eligible &= data.value >= 0
        if outcome in mortality and not data.loc[eligible, "value"].isin([0, 1]).all():
            raise ValueError(f"Expected binary mortality outcomes: {outcome}")
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
                "q3": d.value.quantile(.75),
                "n_positive_admissions": int(d.value.sum()) if outcome in mortality else None,
                "n_positive_subjects": d.loc[d.value.eq(1), "subject_id"].nunique()
                if outcome in mortality else None})
    pd.concat(retained).to_csv(OUTPUT / "model_inputs.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUTPUT / "coverage.csv", index=False)
    pd.DataFrame(descriptions).to_csv(OUTPUT / "descriptives.csv", index=False)
    paths = [pairs_path, cov_path, section_path, note_path, stay_path]
    provenance = {str(path.relative_to(PYTHON)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths}
    provenance["context_table_sha256"] = hashlib.sha256(
        context.sort_values(KEY).to_csv(index=False).encode()).hexdigest()
    (OUTPUT / "input_manifest.json").write_text(json.dumps(provenance, indent=2))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()

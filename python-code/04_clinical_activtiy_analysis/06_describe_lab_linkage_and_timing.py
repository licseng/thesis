"""Describe clinical-event frequencies, lab linkage and within-CC variation.

Read-only source access; no clinical text/results are read or exported. A NULL
hadm_id lab is recovered only when patient and charttime match exactly one valid
hospitalization in the complete admissions table. Existing links are untouched.
All retained pairs and extreme counts stay in the corrected count dataset.
Timing summaries alone require valid positive LOS for BOTH members of a pair.
Outputs are local derivatives, not replacements for SQL exports or model inputs.
"""
from pathlib import Path
import importlib.util

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("workup", HERE / "03_analyze_chief_complaint_subgroup_outcomes.py")
workup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workup)
OUT = HERE / "analysis_output_lab_linkage_and_timing"
KEYS = ["cohort", "subject_id", "hadm_id"]
BINS = [("0–24h", 0, 24), ("24–48h", 24, 48), ("48–72h", 48, 72),
        ("days 4–7", 72, 168), ("after day 7", 168, np.inf)]


def describe_variation(data):
    assignments = pd.read_csv(HERE / "analysis_output_top_pure_chief_complaints" / "top_seven_cc_admission_assignments.csv")
    rows = []
    for group in sorted(data.pure_cc_group.unique()):
        flag = "has_" + group.replace(" ", "_")
        has_cc = assignments[flag].astype(str).str.lower().isin(["true", "1"])
        subsets = {
            "all_matched_admissions_with_this_CC": assignments[has_cc],
            "pure_CC_before_pair_restriction": assignments[has_cc & assignments.selection_status.eq("pure")],
            "overlapping_selected_CCs": assignments[has_cc & assignments.selection_status.eq("multiple_selected_groups")],
            "retained_pure_complete_pairs": data[data.pure_cc_group.eq(group)],
        }
        for population, frame in subsets.items():
            for cohort, f in frame.groupby("cohort"):
                for measure in ["age_at_admission", "elixhauser_score"]:
                    y = f[measure].dropna()
                    rows.append(dict(cc_group=group, cohort=cohort, population=population,
                        measure=measure, n_admissions=len(f), n_patients=f.subject_id.nunique(),
                        n_nonmissing=len(y), mean=y.mean(), sd=y.std(), median=y.median(),
                        q1=y.quantile(.25), q3=y.quantile(.75), minimum=y.min(), maximum=y.max()))
    pd.DataFrame(rows).to_csv(OUT / "within_CC_score_variation.csv", index=False)


def plot_timing(summary, value_column="pooled_lab_rows_per_day",
                event_label="Laboratory measurement rows", stem="lab_timing_rates"):
    """Plot descriptive pooled rates, without implying inferential intervals."""
    import os
    cache = OUT / ".matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    order = [label for label, _, _ in BINS]
    labels = ["Day 1", "Day 2", "Day 3", "Days 4–7", "After day 7"]
    colors = {"MHC0": "#238b45", "MHC1_psychotic": "#8055ad"}
    for ax, group in zip(axes.flat, sorted(summary.cc_group.unique())):
        for cohort, label in [("MHC0", "MHC0"), ("MHC1_psychotic", "MHC1–psychosis")]:
            y = summary[(summary.cc_group == group) & (summary.cohort == cohort)].set_index("interval").loc[order]
            ax.plot(range(5), y[value_column], marker="o", color=colors[cohort], label=label)
        ax.set_title(group.capitalize(), fontsize=11)
        ax.set_xticks(range(5), labels, rotation=20, fontsize=9)
        ax.set_ylim(0, summary[value_column].max()*1.15)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    handles, legends = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, legends, loc="upper left", frameon=False)
    axes.flat[-1].text(.02, .65, "Pooled rate = recorded events\n÷ observed inpatient days.\n\nPartial days use actual exposure.\nLater intervals contain only patients\nstill hospitalized.\n\nDescriptive: no significance tests.", transform=axes.flat[-1].transAxes, fontsize=10, va="top")
    fig.suptitle(event_label + " over the hospital stay", fontsize=15)
    fig.supylabel(event_label + " per observed hospital day")
    fig.tight_layout(rect=[.02, 0, 1, .95])
    fig.savefig(OUT / (stem + ".png"), dpi=180)
    fig.savefig(OUT / (stem + ".svg"))
    plt.close(fig)


def summarize_clinical_events(retained, timed, corrected_labs, other_events):
    """Comparable inpatient windows for three families and lab specimens.

Microbiology/POE use existing admission links only; no new attribution silently
introduced. Missing-time records are audited, not assigned midnight timestamps.
"""
    configurations = {
        "laboratory_measurements": (corrected_labs, "labevent_id", False),
        "laboratory_specimens": (corrected_labs, "specimen_id", True),
        "microbiology_records": (other_events["microbiology_records"], "microevent_id", False),
        "provider_order_activity": (other_events["provider_order_activity"], "poe_id", False),
    }
    summaries, totals, scope, admission_intervals = [], [], [], []
    duration = (timed.dischtime-timed.admittime).dt.total_seconds()/3600
    for measure, (source, identifier, distinct) in configurations.items():
        # source contains only IDs and a relevant timestamp; never clinical text.
        event = source[KEYS + [identifier, "charttime"]].copy()
        if distinct:
            # Assign each specimen once using its earliest recorded timestamp,
            # rather than counting it again if its measurements span intervals.
            event = event.groupby(KEYS + [identifier], as_index=False).charttime.min()
        if not distinct and event[identifier].duplicated().any():
            raise ValueError(f"Duplicate source IDs: {measure}")
        event = event.merge(retained[KEYS + ["pure_cc_group", "admittime", "dischtime"]], on=KEYS, validate="many_to_one")
        event["scope"] = np.select(
            [event.charttime.isna(), ~event.dischtime.gt(event.admittime),
             event.charttime.lt(event.admittime), event.charttime.ge(event.dischtime)],
            ["missing_timestamp", "invalid_admission_window", "before_admission", "at_or_after_discharge"], default="inpatient")
        s = event.groupby(["pure_cc_group", "cohort", "scope"]).size().rename("n_record_rows").reset_index()
        s["measure"] = measure
        scope.append(s)
        event = event.merge(timed[KEYS], on=KEYS, validate="many_to_one")
        event = event[event.scope.eq("inpatient")]
        event["hours"] = (event.charttime-event.admittime).dt.total_seconds()/3600
        def count_records(frame):
            grouped = frame.groupby(KEYS)[identifier]
            return (grouped.nunique() if distinct else grouped.size()).rename("n_events").reset_index()
        whole = timed[KEYS + ["pair_id", "pure_cc_group", "hospital_los_days"]].merge(count_records(event), on=KEYS, how="left", validate="one_to_one")
        whole["n_events"] = whole.n_events.fillna(0).astype(int)
        whole["rate"] = whole.n_events/whole.hospital_los_days
        for (group, cohort), f in whole.groupby(["pure_cc_group", "cohort"]):
            totals.append(dict(cc_group=group, cohort=cohort, measure=measure,
                n_admissions=len(f), n_pairs=f.pair_id.nunique(), total_events=int(f.n_events.sum()),
                median_count=f.n_events.median(), q1_count=f.n_events.quantile(.25), q3_count=f.n_events.quantile(.75),
                mean_count=f.n_events.mean(), sd_count=f.n_events.std(), maximum_count=f.n_events.max(),
                n_zero=int(f.n_events.eq(0).sum()), observed_hospital_days=f.hospital_los_days.sum(),
                pooled_events_per_day=f.n_events.sum()/f.hospital_los_days.sum(),
                median_admission_rate=f.rate.median(), q1_admission_rate=f.rate.quantile(.25), q3_admission_rate=f.rate.quantile(.75)))
        for label, low, high in BINS:
            f = timed[KEYS + ["pair_id", "pure_cc_group"]].copy()
            f["exposure_days"] = (np.minimum(duration,high)-low).clip(lower=0)/24
            f = f.merge(count_records(event[event.hours.ge(low) & event.hours.lt(high)]), on=KEYS, how="left", validate="one_to_one")
            f["n_events"] = f.n_events.fillna(0).astype(int)
            f["rate"] = f.n_events/f.exposure_days.where(f.exposure_days.gt(0))
            f["interval"] = label; f["measure"] = measure
            admission_intervals.append(f)
            for (group, cohort), g in f.groupby(["pure_cc_group", "cohort"]):
                risk = g[g.exposure_days.gt(0)]
                exposure = g.exposure_days.sum()
                summaries.append(dict(cc_group=group, cohort=cohort, measure=measure, interval=label,
                    n_admissions_at_risk=len(risk), total_events=int(g.n_events.sum()), observed_hospital_days=exposure,
                    pooled_events_per_day=g.n_events.sum()/exposure if exposure else np.nan,
                    median_admission_rate=risk.rate.median(), q1_admission_rate=risk.rate.quantile(.25), q3_admission_rate=risk.rate.quantile(.75)))
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / "clinical_event_timing_summary.csv", index=False)
    pd.DataFrame(totals).to_csv(OUT / "clinical_event_whole_stay_summary.csv", index=False)
    pd.concat(scope, ignore_index=True).to_csv(OUT / "clinical_event_timestamp_audit.csv", index=False)
    pd.concat(admission_intervals, ignore_index=True).to_csv(OUT / "clinical_event_admission_intervals.csv", index=False)
    for measure, title in [("laboratory_specimens", "Laboratory specimens"),
                           ("microbiology_records", "Microbiology record rows"),
                           ("provider_order_activity", "Provider-order activity")]:
        plot_timing(result[result.measure.eq(measure)], "pooled_events_per_day", title, measure + "_timing_rates")


def main():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(workup.PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_dataset.csv")
    # Remain idempotent after 03 promotes this correction into active inputs.
    if "n_labevents_rows_original" in data.columns:
        data["n_labevents_rows"] = data.n_labevents_rows_original
    data = data.drop(columns=[c for c in ["n_labevents_rows_original", "n_recovered_lab_rows",
        "n_inpatient_lab_rows", "n_inpatient_lab_specimens", "inpatient_lab_rows_per_hospital_day", "n_linked_source_rows"] if c in data.columns])
    desc = workup.load_descriptors()[KEYS + ["admittime", "dischtime"]]
    retained = data.merge(desc, on=KEYS, validate="one_to_one")
    assert not retained.duplicated(KEYS).any()
    assert retained.groupby("pair_id").size().eq(2).all()
    with duckdb.connect(str(workup.common.DB_PATH), read_only=True) as con:
        tables = set(con.execute("SHOW TABLES").fetchdf().iloc[:, 0])
        if not {"admissions", "labevents"}.issubset(tables):
            raise ValueError("Complete admissions and labevents source tables required")
        con.register("retained", retained)
        # Only IDs and timestamps are selected: never lab values/comments.
        con.execute("""CREATE TEMP TABLE linked AS
            SELECT r.cohort,r.subject_id,r.hadm_id,l.labevent_id,l.specimen_id,l.charttime
            FROM retained r JOIN labevents l
            ON r.subject_id=l.subject_id AND r.hadm_id=l.hadm_id""")
        con.execute("""CREATE TEMP TABLE candidates AS
            SELECT r.cohort,r.subject_id,r.hadm_id,l.labevent_id,l.specimen_id,l.charttime
            FROM retained r JOIN labevents l ON r.subject_id=l.subject_id
            AND l.hadm_id IS NULL AND l.charttime >= r.admittime
            AND l.charttime < r.dischtime AND r.dischtime > r.admittime""")
        # Check against ALL hospitalizations, not just selected matched admissions.
        con.execute("""CREATE TEMP TABLE unique_windows AS
            SELECT c.labevent_id,count(DISTINCT a.hadm_id) AS n_windows
            FROM (SELECT DISTINCT labevent_id,subject_id,charttime FROM candidates) c
            JOIN admissions a ON c.subject_id=a.subject_id
            AND c.charttime >= a.admittime AND c.charttime < a.dischtime
            AND a.dischtime > a.admittime GROUP BY c.labevent_id""")
        linked = con.execute("SELECT * FROM linked").fetchdf()
        recovered = con.execute("""SELECT c.* FROM candidates c
            JOIN unique_windows u USING(labevent_id) WHERE u.n_windows=1""").fetchdf()
        candidate_qc = con.execute("""SELECT count(DISTINCT c.labevent_id) AS n_candidate_rows,
            count(DISTINCT CASE WHEN u.n_windows>1 THEN c.labevent_id END) AS n_ambiguous_rows
            FROM candidates c JOIN unique_windows u USING(labevent_id)""").fetchdf()
        other_events = {}
        for label, table, identifier, time_column in [
            ("microbiology_records", "microbiologyevents", "microevent_id", "charttime"),
            ("provider_order_activity", "poe", "poe_id", "ordertime")]:
            other_events[label] = con.execute(f"""SELECT r.cohort,r.subject_id,r.hadm_id,
                e.{identifier},e.{time_column} AS charttime FROM retained r JOIN {table} e
                ON r.subject_id=e.subject_id AND r.hadm_id=e.hadm_id""").fetchdf()
            original = "n_microbiologyevents_rows" if table == "microbiologyevents" else "n_poe_rows"
            check = retained[KEYS + [original]].merge(
                other_events[label].groupby(KEYS).size().rename("source_count").reset_index(), on=KEYS, how="left", validate="one_to_one")
            if not check[original].eq(check.source_count.fillna(0)).all():
                raise ValueError(f"Cached/source mismatch for {table}")
    if linked.labevent_id.duplicated().any() or recovered.labevent_id.duplicated().any():
        raise ValueError("Duplicate lab attribution detected")
    if set(linked.labevent_id) & set(recovered.labevent_id):
        raise ValueError("Recovered lab duplicated an existing link")
    linked_counts = linked.groupby(KEYS).size().rename("n_linked_source_rows").reset_index()
    recovery_counts = recovered.groupby(KEYS).size().rename("n_recovered_lab_rows").reset_index()
    corrected = retained.merge(linked_counts, on=KEYS, how="left", validate="one_to_one").merge(
        recovery_counts, on=KEYS, how="left", validate="one_to_one")
    for column in ["n_linked_source_rows", "n_recovered_lab_rows"]:
        corrected[column] = corrected[column].fillna(0).astype(int)
    if not corrected.n_linked_source_rows.eq(corrected.n_labevents_rows).all():
        raise ValueError("Source labs do not reproduce cached counts; stop before correction")
    corrected["n_labevents_rows_original"] = corrected.n_labevents_rows
    corrected["n_labevents_rows"] += corrected.n_recovered_lab_rows
    corrected["n_labevents_rows_per_hospital_day"] = corrected.n_labevents_rows / corrected.hospital_los_days.where(corrected.hospital_los_days.gt(0))
    linkage = []
    for (group, cohort), f in corrected.groupby(["pure_cc_group", "cohort"]):
        linkage.append(dict(cc_group=group, cohort=cohort, n_admissions=len(f),
            n_original_zero=int(f.n_labevents_rows_original.eq(0).sum()),
            n_original_zero_recovered=int((f.n_labevents_rows_original.eq(0) & f.n_recovered_lab_rows.gt(0)).sum()),
            n_admissions_with_recovery=int(f.n_recovered_lab_rows.gt(0).sum()),
            n_added_lab_rows=int(f.n_recovered_lab_rows.sum()),
            n_remaining_zero=int(f.n_labevents_rows.eq(0).sum())))
    pd.DataFrame(linkage).to_csv(OUT / "lab_linkage_recovery_summary.csv", index=False)
    events = pd.concat([linked.assign(linkage="existing"), recovered.assign(linkage="recovered")], ignore_index=True)
    events = events.merge(retained[KEYS + ["pure_cc_group", "admittime", "dischtime"]], on=KEYS, validate="many_to_one")
    events["hours_since_admission"] = (events.charttime - events.admittime).dt.total_seconds()/3600
    events["timing_category"] = np.select(
        [events.charttime.isna(), ~(events.dischtime.gt(events.admittime)), events.charttime.lt(events.admittime), events.charttime.ge(events.dischtime)],
        ["missing_timestamp", "invalid_admission_window", "before_admission", "at_or_after_discharge"], default="inpatient")
    events.groupby(["pure_cc_group", "cohort", "linkage", "timing_category"]).size().rename("n_lab_rows").reset_index().to_csv(OUT / "lab_timestamp_scope_audit.csv", index=False)
    inpatient = events[events.timing_category.eq("inpatient")].groupby(KEYS).agg(
        n_inpatient_lab_rows=("labevent_id", "size"), n_inpatient_lab_specimens=("specimen_id", "nunique")).reset_index()
    corrected = corrected.merge(inpatient, on=KEYS, how="left", validate="one_to_one")
    for column in ["n_inpatient_lab_rows", "n_inpatient_lab_specimens"]:
        corrected[column] = corrected[column].fillna(0)
        corrected.loc[~corrected.dischtime.gt(corrected.admittime), column] = np.nan
    corrected["inpatient_lab_rows_per_hospital_day"] = corrected.n_inpatient_lab_rows / corrected.hospital_los_days.where(corrected.hospital_los_days.gt(0))
    # Separate total linked-admission counts (including pre-admission events)
    # from inpatient-window counts: their exposure denominators differ.
    # No raw timestamps or individual events in saved derivatives.
    corrected.drop(columns=["admittime", "dischtime"]).to_csv(OUT / "lab_linkage_corrected_workup_dataset.csv", index=False)
    valid = retained.admittime.notna() & retained.dischtime.notna() & retained.dischtime.gt(retained.admittime)
    complete_ids = retained.assign(valid=valid).groupby("pair_id").valid.all()
    ids = complete_ids[complete_ids].index
    timed = retained[retained.pair_id.isin(ids)].copy()
    summarize_clinical_events(retained, timed,
        pd.concat([linked, recovered], ignore_index=True), other_events)
    events = events.merge(timed[KEYS], on=KEYS, validate="many_to_one")
    events = events[events.timing_category.eq("inpatient")]
    intervals = []
    duration = (timed.dischtime - timed.admittime).dt.total_seconds()/3600
    for label, low, high in BINS:
        f = timed[KEYS + ["pair_id", "pure_cc_group"]].copy()
        f["interval"] = label
        f["exposure_days"] = (np.minimum(duration, high)-low).clip(lower=0)/24
        ev = events[events.hours_since_admission.ge(low) & events.hours_since_admission.lt(high)]
        counts = ev.groupby(KEYS).agg(n_lab_rows=("labevent_id", "size"), n_specimens=("specimen_id", "nunique")).reset_index()
        f = f.merge(counts, on=KEYS, how="left", validate="one_to_one")
        f[["n_lab_rows", "n_specimens"]] = f[["n_lab_rows", "n_specimens"]].fillna(0).astype(int)
        f["lab_rows_per_day"] = f.n_lab_rows/f.exposure_days.where(f.exposure_days.gt(0))
        f["specimens_per_day"] = f.n_specimens/f.exposure_days.where(f.exposure_days.gt(0))
        intervals.append(f)
    intervals = pd.concat(intervals, ignore_index=True)
    exposure_check = intervals.groupby(KEYS).exposure_days.sum().reset_index().merge(
        timed[KEYS + ["hospital_los_days"]], on=KEYS, validate="one_to_one")
    if not np.allclose(exposure_check.exposure_days, exposure_check.hospital_los_days):
        raise ValueError("Interval exposures do not sum to hospitalization duration")
    if intervals.n_lab_rows.sum() != len(events):
        raise ValueError("Inpatient event intervals do not partition the lab rows")
    intervals.to_csv(OUT / "lab_timing_admission_intervals.csv", index=False)
    total_labs = intervals.groupby(["pure_cc_group", "cohort"]).n_lab_rows.sum()
    rows = []
    for (group, cohort, interval), f in intervals.groupby(["pure_cc_group", "cohort", "interval"], sort=False):
        at_risk = f[f.exposure_days.gt(0)]
        denom = f.exposure_days.sum()
        rows.append(dict(cc_group=group, cohort=cohort, interval=interval,
            n_admissions_at_risk=len(at_risk), n_subjects_at_risk=at_risk.subject_id.nunique(),
            n_lab_rows=int(f.n_lab_rows.sum()), n_specimen_intervals=int(f.n_specimens.sum()),
            observed_hospital_days=denom, pooled_lab_rows_per_day=f.n_lab_rows.sum()/denom if denom else np.nan,
            pooled_specimens_per_day=f.n_specimens.sum()/denom if denom else np.nan,
            median_admission_lab_rate=at_risk.lab_rows_per_day.median(),
            q1_admission_lab_rate=at_risk.lab_rows_per_day.quantile(.25), q3_admission_lab_rate=at_risk.lab_rows_per_day.quantile(.75),
            pct_all_inpatient_lab_rows=100*f.n_lab_rows.sum()/total_labs.loc[group,cohort]))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "lab_timing_summary.csv", index=False)
    plot_timing(summary)
    qc = candidate_qc.iloc[0].to_dict()
    qc.update(n_retained_admissions=len(data), n_retained_pairs=data.pair_id.nunique(),
        n_timing_admissions=len(timed), n_timing_pairs=timed.pair_id.nunique(),
        n_invalid_time_pairs=int((~complete_ids).sum()), n_recovered_lab_rows=len(recovered))
    pd.DataFrame([qc]).to_csv(OUT / "lab_timing_qc.csv", index=False)
    describe_variation(data)
    (OUT / "README.txt").write_text(
        "Descriptive only; primary outputs/database not overwritten; extreme cases retained.\n"
        "Recovery: NULL hadm_id, same subject, charttime in [admittime,dischtime), exactly one matching hospitalization globally.\n"
        "Recovery applied consistently to all retained admissions; nonmissing original links not reassigned.\n"
        "Existing linked records outside inpatient windows remain in corrected total counts but are excluded from inpatient timing.\n"
        "Timing excludes entire pairs with invalid/nonpositive duration. Partial intervals use actual exposure.\n"
        "Late intervals include only patients still hospitalized: changing risk sets prevent causal interpretation of temporal/cohort differences.\n"
        "Rows are laboratory measurements, not distinct tests. Specimen counts are distinct per admission/interval, not globally additive.\n"
        "Within-CC age/Elixhauser summaries are admission-weighted and cannot establish clinical homogeneity.\n"
        "Clinical-event summaries cover three families plus distinct lab specimens. POE detail excluded.\n"
        "Microbiology and POE use original admission links; NULL admission-ID recovery has only been implemented for labs.\n"
        "Microbiology timing uses charttime (collection proxy), POE uses ordertime; records lacking exact time are excluded from timing, not imputed.\n"
        "POE retains all categories/transactions: order activity, not distinct diagnostic decisions.\n"
        "Clinical specimen timing assigns each distinct specimen once, at its earliest recorded timestamp; specimens are samples, not necessarily blood draws.\n"
    )
    print(pd.DataFrame(linkage).to_string(index=False))
    print("Timing/linkage QC:", qc)


if __name__ == "__main__":
    main()

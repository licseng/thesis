"""Explore seven common grouped complaints in the current matched cohort.

Rank mapped symptom groups by admission prevalence in each cohort and pooled.
Synonyms are combined using normalized CC phrases and QuickUMLS concepts.
Select the pooled top seven, then retain admissions matching exactly one of
these seven groups. This operational 'purity' does not exclude additional
symptoms outside the selected seven. Counts refer to the matched cohort.
Outputs contain identifiers/flags or aggregates, never raw chief complaints.
No work-up comparisons or significance tests are performed here.
"""
from pathlib import Path
import importlib.util
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_top_pure_chief_complaints"


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "cc_subgroups", SCRIPT_DIR / "01_describe_chief_complaint_subgroups.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_groups(helper):
    groups = {name: {key: list(values) for key, values in config.items()}
              for name, config in helper.CHIEF_COMPLAINT_SUBGROUPS.items()}
    # Broaden ranking beyond the five older, predefined groups. These are
    # common concepts observed in aggregate QuickUMLS frequency summaries.
    aliases = {
        "fall": ["fall", "falls", "falling", "mechanical fall"],
        "fever": ["fever", "fevers", "febrile"],
        "weakness": ["weakness", "generalized weakness"],
        "cough": ["cough", "coughing"],
        "hypotension": ["hypotension", "low blood pressure"],
        "diarrhea": ["diarrhea", "diarrhoea"],
        "syncope": ["syncope", "syncopal", "fainting"],
        "seizure": ["seizure", "seizures"],
        "headache": ["headache", "headaches"],
        "hypoxia": ["hypoxia", "hypoxemia"],
        "dizziness": ["dizziness", "dizzy", "lightheadedness"],
        "back pain": ["back pain", "low back pain"],
        "fatigue": ["fatigue", "tiredness"],
        "hyperglycemia": ["hyperglycemia", "high blood sugar"],
        "anemia": ["anemia", "anaemia"],
        "rectal bleeding": ["bright red blood per rectum", "brbpr", "rectal bleeding", "hematochezia"],
        "failure to thrive": ["failure to thrive", "ftt"],
        "leg swelling": ["leg swelling", "lower extremity swelling"],
    }
    for name, terms in aliases.items():
        groups[name] = {"text_phrases": terms, "quickumls_terms": terms}
    return groups


def main():
    helper = load_helper()
    data = helper.build_admission_level_complaints(helper.load_matched_pairs())
    if data.duplicated(["cohort", "subject_id", "hadm_id"]).any():
        raise ValueError("Duplicate matched admission keys")
    groups = candidate_groups(helper)
    flags = pd.DataFrame({name: data.apply(lambda row: helper.subgroup_hit(row, config), axis=1)
                          for name, config in groups.items()}, index=data.index)
    rankings = []
    for cohort in ["MHC0", "MHC1_psychotic", "overall"]:
        mask = pd.Series(True, index=data.index) if cohort == "overall" else data.cohort.eq(cohort)
        table = flags.loc[mask].sum().rename_axis("complaint_group").reset_index(name="n_admissions")
        table = table.sort_values(["n_admissions", "complaint_group"], ascending=[False, True])
        table["rank"] = range(1, len(table) + 1)
        table["cohort"] = cohort
        table["pct_cohort_admissions"] = 100 * table.n_admissions / mask.sum()
        rankings.append(table)
    ranking = pd.concat(rankings, ignore_index=True)
    selected = ranking.loc[ranking.cohort.eq("overall") & ranking["rank"].le(7), "complaint_group"].tolist()
    n_matches = flags[selected].sum(axis=1)
    assignments = data.drop(columns=["chief_complaint_normalized", "quickumls_terms", "derived_quickumls_overlap_terms"]).copy()
    assignments["n_selected_groups_matched"] = n_matches
    assignments["selection_status"] = n_matches.map(lambda n: "outside_selected_groups" if n == 0 else "pure" if n == 1 else "multiple_selected_groups")
    assignments["pure_cc_group"] = flags[selected].idxmax(axis=1).where(n_matches.eq(1))
    for name in selected:
        assignments["has_" + name.replace(" ", "_")] = flags[name]
    pure = assignments[assignments.selection_status.eq("pure")]
    counts = []
    pairs = []
    demographics = []
    for name in selected:
        selected_data = pure[pure.pure_cc_group.eq(name)]
        pair_sizes = selected_data.groupby("pair_id").cohort.nunique()
        pairs.append({"complaint_group": name, "n_complete_same_group_pairs": int(pair_sizes.eq(2).sum()),
                      "n_one_sided_pairs": int(pair_sizes.eq(1).sum())})
        for cohort in ["MHC0", "MHC1_psychotic"]:
            group = selected_data[selected_data.cohort.eq(cohort)]
            before = int(flags.loc[data.cohort.eq(cohort), name].sum())
            counts.append({"complaint_group": name, "cohort": cohort, "n_before_purity_filter": before,
                           "n_pure_admissions": len(group), "n_subjects": group.subject_id.nunique(),
                           "n_removed_for_overlap": before-len(group),
                           "pct_cohort_admissions": 100*len(group)/int(data.cohort.eq(cohort).sum())})
            for variable in ["age_at_admission", "elixhauser_score"]:
                demographics.append({"complaint_group": name, "cohort": cohort, "variable": variable,
                                     **helper.summarize_numeric(group[variable])})
    OUTPUT_DIR.mkdir(exist_ok=True)
    ranking.to_csv(OUTPUT_DIR / "candidate_group_ranking.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "top_seven_cc_admission_assignments.csv", index=False)
    pd.DataFrame(counts).to_csv(OUTPUT_DIR / "pure_cc_counts.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUTPUT_DIR / "pure_cc_pair_coverage.csv", index=False)
    pd.DataFrame(demographics).to_csv(OUTPUT_DIR / "pure_cc_numeric_demographics.csv", index=False)
    selection = assignments.groupby(["cohort", "selection_status"]).size().reset_index(name="n_admissions")
    selection.to_csv(OUTPUT_DIR / "pure_cc_selection_summary.csv", index=False)
    definitions = [{"complaint_group": name, "text_phrases": " | ".join(config["text_phrases"]),
                    "quickumls_terms": " | ".join(config["quickumls_terms"]), "selected": name in selected}
                   for name, config in groups.items()]
    pd.DataFrame(definitions).to_csv(OUTPUT_DIR / "candidate_group_definitions.csv", index=False)
    print("Top seven grouped complaints before purity filtering:")
    print(ranking[ranking["rank"].le(7)].to_string(index=False))
    print("\nPure subgroup counts:")
    print(pd.DataFrame(counts).to_string(index=False))
    print("\nSelection:")
    print(selection.to_string(index=False))
    print("\nPair coverage:")
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()

"""Prepare six-section, admission-level inputs for diagnostic overshadowing.

Only current matched MHC1-psychosis admissions are considered. A psychiatry
keyword in ANY selected section retains the admission; ALL available selected
sections then supply context, with headings, in one classification input.
No preliminary LLM, chunking, truncation, or API requests are performed.
The vocabulary is the existing broad exploratory psychiatric vocabulary,
including medications, substance use, and cognitive/behavioral terms.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import re

import pandas as pd

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent.parent
OUTPUT = HERE / "prefilter_output"
SOURCE = PYTHON / "01_discharge_note_preprocessing/01_discharge_note_parsing/full_discharge_note_sections/MHC1_psychotic_matched_full_discharge_note_sections.parquet"
PAIRS = PYTHON / "02_cohort_matching/matched_cohort_output/matched_pairs.csv"
VOCABULARY = PYTHON / "03_discharge_note_text_analysis/02_keyword_matching/02_explore_psych_keywords_by_section.py"
SECTIONS = [
    "present_illness", "brief_hospital_course", "problems",
    "pertinent_results", "discharge_diagnosis", "discharge_instructions",
]
HEADINGS = {
    "present_illness": "Present illness",
    "brief_hospital_course": "Brief hospital course",
    "problems": "Problems",
    "pertinent_results": "Pertinent results",
    "discharge_diagnosis": "Discharge diagnosis",
    "discharge_instructions": "Discharge instructions",
}
METADATA = ["pair_id", "cohort", "subject_id", "hadm_id", "note_id", "charttime"]


def load_vocabulary():
    spec = importlib.util.spec_from_file_location("diagnostic_psych_vocabulary", VOCABULARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KEYWORD_PATTERNS


def assemble_context(row):
    """Preserve each selected section verbatim apart from surrounding whitespace."""
    return "\n\n".join(
        f"### {HEADINGS[section]}\n{row[section]}"
        for section in SECTIONS if row[section]
    )


def main():
    vocabulary = load_vocabulary()
    patterns = {
        group: [re.compile(pattern, re.IGNORECASE) for pattern in expressions]
        for group, expressions in vocabulary.items()
    }
    source = pd.read_parquet(SOURCE, columns=["subject_id", "hadm_id", "note_id",
                                             "charttime"] + SECTIONS)
    pairs = pd.read_csv(PAIRS, usecols=["pair_id", "mhc1_subject_id", "mhc1_hadm_id"])
    pairs = pairs.rename(columns={"mhc1_subject_id": "subject_id", "mhc1_hadm_id": "hadm_id"})
    if source.hadm_id.duplicated().any() or pairs.hadm_id.duplicated().any():
        raise ValueError("Expected one parsed note and one match per MHC1 admission")
    if set(zip(source.subject_id, source.hadm_id)) != set(zip(pairs.subject_id, pairs.hadm_id)):
        raise ValueError("Parsed notes do not exactly match current matched MHC1 admissions")
    source = pairs.merge(source, on=["subject_id", "hadm_id"], validate="one_to_one")
    source.insert(1, "cohort", "MHC1_psychotic")
    for section in SECTIONS:
        source[section] = source[section].fillna("").astype(str).str.strip()
    audit = source[METADATA].copy()
    section_rows = []
    group_hits = {group: pd.Series(False, index=source.index) for group in patterns}
    for section in SECTIONS:
        print(f"Checking {HEADINGS[section]}...", flush=True)
        nonempty = source[section].ne("")
        section_positive = pd.Series(False, index=source.index)
        for group, compiled in patterns.items():
            hits = source[section].map(lambda text: any(p.search(text) for p in compiled))
            group_hits[group] |= hits
            section_positive |= hits
        audit[f"has_{section}"] = nonempty
        audit[f"keyword_positive_{section}"] = section_positive
        audit[f"n_words_{section}"] = source[section].str.split().str.len()
        section_rows.append({
            "section_name": section, "n_source_admissions": len(source),
            "n_nonempty_sections": int(nonempty.sum()),
            "n_keyword_positive_sections": int(section_positive.sum()),
        })
    positive_columns = [f"keyword_positive_{s}" for s in SECTIONS]
    word_columns = [f"n_words_{s}" for s in SECTIONS]
    audit["passes_psych_keyword_filter"] = audit[positive_columns].any(axis=1)
    audit["n_keyword_positive_sections"] = audit[positive_columns].sum(axis=1)
    audit["n_available_sections"] = audit[[f"has_{s}" for s in SECTIONS]].sum(axis=1)
    audit["n_selected_section_words"] = audit[word_columns].sum(axis=1)
    for group, hits in group_hits.items():
        audit[f"has_keyword_group_{group}"] = hits
    selected = audit.passes_psych_keyword_filter
    retained = source.loc[selected, METADATA + SECTIONS].copy()
    retained["classification_input_id"] = "diagnostic_overshadowing:" + retained.hadm_id.astype(str)
    retained["included_sections"] = retained.apply(
        lambda row: " | ".join(s for s in SECTIONS if row[s]), axis=1)
    retained["keyword_positive_sections"] = audit.loc[selected].apply(
        lambda row: " | ".join(s for s in SECTIONS if row[f"keyword_positive_{s}"]), axis=1)
    retained["classification_text"] = retained.apply(assemble_context, axis=1)
    retained["n_input_words"] = retained.classification_text.str.split().str.len()
    retained["n_input_characters"] = retained.classification_text.str.len()
    if retained.classification_text.eq("").any() or retained.classification_input_id.duplicated().any():
        raise ValueError("Empty or duplicate classification input")
    audit = audit.merge(retained[["hadm_id", "classification_input_id", "included_sections",
        "keyword_positive_sections", "n_input_words", "n_input_characters"]],
        on="hadm_id", how="left", validate="one_to_one")
    for section_row in section_rows:
        s = section_row["section_name"]
        section_row["n_sections_in_retained_inputs"] = int((selected & source[s].ne("")).sum())
    lengths = retained.n_input_words
    summary = {
        "n_source_admissions": len(source),
        "n_source_subjects": int(source.subject_id.nunique()),
        "n_retained_admissions": len(retained),
        "n_retained_subjects": int(retained.subject_id.nunique()),
        "n_excluded_no_keyword": int((~selected).sum()),
        "pct_admissions_retained": round(100 * selected.mean(), 2),
        "n_keyword_positive_sections": int(audit.n_keyword_positive_sections.sum()),
        "n_context_sections_retained": int(audit.loc[selected, "n_available_sections"].sum()),
        "n_nonkeyword_context_sections_retained": int(
            audit.loc[selected, "n_available_sections"].sum() - audit.n_keyword_positive_sections.sum()),
        "planned_one_request_per_admission": len(retained),
        "total_input_words_with_headings": int(lengths.sum()),
        "median_input_words": float(lengths.median()),
        "p95_input_words": float(lengths.quantile(.95)),
        "maximum_input_words": int(lengths.max()),
        "source_six_section_words": int(audit.n_selected_section_words.sum()),
        "retained_six_section_words": int(audit.loc[selected, "n_selected_section_words"].sum()),
        "n_api_requests_made": 0,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # The only text-bearing derivative is local, ignored parquet. CSVs contain metadata only.
    retained.to_parquet(OUTPUT / "diagnostic_overshadowing_admission_input.parquet", index=False)
    audit.to_csv(OUTPUT / "diagnostic_overshadowing_prefilter_admission_metadata.csv", index=False)
    pd.DataFrame(section_rows).to_csv(OUTPUT / "diagnostic_overshadowing_prefilter_section_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUT / "diagnostic_overshadowing_prefilter_summary.csv", index=False)
    group_summary = [
        {"keyword_group": group, "n_positive_admissions": int(hits.sum())}
        for group, hits in group_hits.items()
    ]
    pd.DataFrame(group_summary).to_csv(OUTPUT / "diagnostic_overshadowing_keyword_group_summary.csv", index=False)
    manifest = {
        "cohort": "MHC1_psychotic", "selected_sections": SECTIONS,
        "filter_rule": "Any psychiatric keyword in any of the six sections",
        "context_rule": "All available selected sections for each retained admission",
        "keyword_patterns": vocabulary, "first_LLM_gate": False,
        "chunking": False, "truncation": False,
        "word_counts_are_not_token_counts": True, "summary": summary,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "pairs_sha256": hashlib.sha256(PAIRS.read_bytes()).hexdigest(),
        "vocabulary_script_sha256": hashlib.sha256(VOCABULARY.read_bytes()).hexdigest(),
        "prefilter_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_parquet_sha256": hashlib.sha256(
            (OUTPUT / "diagnostic_overshadowing_admission_input.parquet").read_bytes()).hexdigest(),
    }
    (OUTPUT / "diagnostic_overshadowing_prefilter_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(pd.DataFrame(section_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

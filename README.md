# Thesis Code Repository

This repository contains the SQL and Python workflow for a diagnostic
overshadowing thesis using MIMIC-IV data.

The current main comparison is:

- **Exposed cohort:** `MHC1_psychotic`
- **Control cohort:** `MHC0` (`only_MHC0` in some matching-file names)

The downstream cohort matching uses:

- chief complaint semantic similarity,
- age,
- sex,
- insurance group,
- Elixhauser comorbidity score.

Local data files, generated outputs, embeddings, and database files are ignored
by Git. The repository tracks code and SQL scripts, not MIMIC data.

## SQL Workflow

SQL scripts are stored in:

```text
sql-scripts/Finalized-cohort/
```

These scripts define ICD-based psychiatric cohorts, construct MIMIC-IV cohort
tables, inspect cohort definitions, and create DuckDB export tables consumed by
the Python pipeline. Stages 01–04 construct and export the eligible cohorts;
05 characterizes the pre-matching pool; 06 imports the matched admission IDs
and exports the additional information for matched-cohort analyses. Other SQL
folders are retained as historical references.

The import script uses a DBeaver variable for local MIMIC paths:

```sql
@set mimic_dirs=
```

Set that locally in DBeaver or in a local-only helper file. Do not commit
absolute local paths, credentials, or patient-level exports.

## Python Workflow

Python scripts are stored in:

```text
python-code/
```

The numbered folders follow the main analysis order.

### 01 Discharge Note Preprocessing

```text
python-code/01_discharge_note_preprocessing/
```

Discharge-note parsing scripts:

```text
01_discharge_note_parsing/01_parse_chief_complaints_from_discharge_notes.py
01_discharge_note_parsing/02_parse_full_discharge_notes.py
```

The first parser extracts only the chief complaint from MIMIC-IV discharge
notes exported from DuckDB. Its local outputs are used by the downstream chief
complaint preprocessing step.

The second parser creates a broader full-discharge-note section map for later
psychiatric-history detection. It parses only admissions that are already in the
matched case/control cohort, preserves the original note text locally, and also
writes named section columns plus a JSON map of detected headings.

### Chief Complaint Preprocessing

```text
python-code/01_discharge_note_preprocessing/02_chief_complaint/01_preprocessing/
```

Main scripts:

```text
01_export_chief_complaint_parquets.py
02_analyze_raw_chief_complaint.py
03_preprocess_chief_complaints.py
04_analyze_preprocessed_chief_complaints.py
05_finalize_chief_complaints.py
06_analyze_final_chief_complaints.py
07_analyze_quickumls_terms.py
```

This step exports chief complaint tables, normalizes chief complaint text,
applies MedSpaCy psychiatric/substance/self-harm flags, extracts local QuickUMLS
terms, runs sanity checks, and creates finalized chief complaint files for
embedding.

QuickUMLS runs locally. The index path should be supplied through the
`QUICKUMLS_INDEX_DIR` environment variable when needed:

```bash
QUICKUMLS_INDEX_DIR="/path/to/quickumls_index" python python-code/01_discharge_note_preprocessing/02_chief_complaint/01_preprocessing/03_preprocess_chief_complaints.py
```

### Chief Complaint Embedding

```text
python-code/01_discharge_note_preprocessing/02_chief_complaint/02_embedding/
```

Main script:

```text
01_embed_chief_complaints.py
```

This embeds finalized chief complaints locally with
`emilyalsentzer/Bio_ClinicalBERT`. The current default embedding text is:

```python
TEXT_COLUMN = "chief_complaint_normalized"
```

The script saves one metadata parquet and one `.npy` embedding matrix per cohort.

### 02 Cohort Matching

```text
python-code/02_cohort_matching/
```

Main scripts:

```text
01_elixhauser/01_calculate_elixhauser_scores.py
02_matching_variables/01_create_matching_variable_tables.py
03_match_chief_complaint_cohorts.py
04_analyze_matched_cohort.py
```

The Elixhauser script computes admission-level comorbidity scores from ICD
diagnosis rows using `comorbidipy`.

The matching-variable script joins embedding metadata, age, sex, insurance, and Elixhauser
score into one admission-level table per cohort.

The matching script performs greedy 1:1 matching without replacement. It uses
chief complaint BERT embedding similarity as the main signal, with exact sex
and insurance-group matching, the same/adjacent age band, QuickUMLS term overlap
with a fallback, and strict/relaxed Elixhauser calipers. Within the 0.01
similarity-tie window, candidates are ordered by absolute age difference,
absolute Elixhauser difference, and then higher similarity.

The matched-cohort analysis script summarizes match quality, including cosine
similarity, age-bin distance, age-year distance, match type, and Elixhauser
balance.

### 03 Discharge Note Text Analysis

`python-code/03_discharge_note_text_analysis/01_language_complexity/` contains
descriptive complexity scores, paired tests, patient/pair linear mixed models,
and language, race and neurocognitive sensitivity analyses.
`02_keyword_matching/` contains psychiatric and candidate stigmatizing-language
keyword searches. Keyword hits are not validated stigma classifications.

### 04 Clinical Route Analyses

`python-code/04_clinical_route_analyses/` contains chief-complaint subgroup
selection, laboratory-linkage/timing checks, clinical-event analyses and
hospital/ED stay and mortality analyses. The current five-group complete-pair
workflow is implemented in scripts 01, 06 and the command-line modes of 03.
Do not run every numbered file indiscriminately: script 04 is an earlier
nausea-including subgroup exploration, and script 07 still targets the removed
legacy folder-09 mortality dataset; it is not a current pipeline step.

Current clinical-event inference uses log-link PPML with patient/pair clustered
standard errors. Current hospital outcomes use PPML for duration and logistic
regression for mortality with the same clustering. Earlier NB/ZI and clinical
mixed-model outputs remain exploratory alternatives, not the primary results.

Whole matched-cohort LOS, ED stay and both mortality endpoints can be fitted
without changing the CC-specific outputs:

```bash
python python-code/04_clinical_route_analyses/03_analyze_chief_complaint_subgroup_outcomes.py --fit-whole-cohort-stay-mortality
```

This saves M0/M2 results and diagnostics to
`analysis_output_whole_matched_stay_mortality/`, with complete pairs per outcome
and a separate four-endpoint BH FDR family within each model stage. It does not
fit whole-cohort clinical-event/work-up models.

For sensitivity adjustments on the identical saved endpoint-specific samples:

```bash
python python-code/04_clinical_route_analyses/03_analyze_chief_complaint_subgroup_outcomes.py --fit-whole-cohort-stay-mortality-sensitivity
```

M3 adds recorded language group to M2; M4 adds the recorded race/ethnicity group
to M3, using the existing language-complexity covariate groupings. Outputs are
saved under `analysis_output_whole_matched_stay_mortality/language_race_sensitivity/`;
primary M0/M2 and CC-specific outputs are preserved. The current per-stage FDR
convention is retained for comparison, not claimed as thesis-wide correction.

### Next: Annotation and LLM Classification

Folders 05–07 contain annotation preparation, psychiatric-context/diagnostic-
overshadowing classifiers and sentiment classification. These are existing
implementations, not yet a finalized one-stage classification workflow. See
[PIPELINE_AUDIT.md](PIPELINE_AUDIT.md) for the checks and remaining decisions
before a full new-cohort LLM run. The active folder 09 was removed; its legacy
regression scripts remain under
`analysis-archive/MHH1_original_2026-09-25/python-code/09_regression_analysis/`.
Current modelling belongs in the respective analysis folders.

## Generated Outputs

Generated outputs are intentionally ignored by Git. Examples include:

```text
parsed_chief_complaints/
full_discharge_note_sections/
chief_complaint_preprocessed/
chief_complaint_final/
chief_complaint_embeddings/
elixhauser_scores_output/
matching_variable_tables_output/
matched_cohort_output/
analysis_output_*/
```

Regenerate these locally by running the pipeline scripts in order.

# Thesis Code Repository

This repository contains the SQL and Python workflow for a diagnostic
overshadowing thesis using MIMIC-IV data.

The current main comparison is:

- **Exposed cohort:** `MHH_psychotic`
- **Control cohort:** `only_MHC0`

The downstream cohort matching uses:

- chief complaint semantic similarity,
- age,
- sex,
- Elixhauser comorbidity score.

Local data files, generated outputs, embeddings, and database files are ignored
by Git. The repository tracks code and SQL scripts, not MIMIC data.

## SQL Workflow

SQL scripts are stored in:

```text
sql-scripts/
```

These scripts define ICD-based psychiatric cohorts, construct MIMIC-IV cohort
tables, inspect cohort definitions, and create DuckDB export tables consumed by
the Python pipeline.

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

### 01 Admission Notes

```text
python-code/01_admission_notes/
```

Main script:

```text
01_parse_discharge_notes_for_admission_part.py
```

This parses admission-relevant sections from MIMIC-IV discharge notes exported
from DuckDB. It extracts fields such as chief complaint and present illness into
local parquet outputs.

### Chief Complaint Preprocessing

```text
python-code/01_admission_notes/02_chief_complaint/01_preprocessing/
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
QUICKUMLS_INDEX_DIR="/path/to/quickumls_index" python python-code/01_admission_notes/02_chief_complaint/01_preprocessing/03_preprocess_chief_complaints.py
```

### Chief Complaint Embedding

```text
python-code/01_admission_notes/02_chief_complaint/02_embedding/
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

The matching-variable script joins embedding metadata, age, sex, and Elixhauser
score into one admission-level table per cohort.

The matching script performs greedy 1:1 matching without replacement. It uses
chief complaint BERT embedding similarity as the main signal, with sex, nearby
age bins, optional QuickUMLS term overlap, and Elixhauser calipers as candidate
restrictions or tie-breakers.

The matched-cohort analysis script summarizes match quality, including cosine
similarity, age-bin distance, age-year distance, match type, and Elixhauser
balance.

## Generated Outputs

Generated outputs are intentionally ignored by Git. Examples include:

```text
parsed_admission_notes/
chief_complaint_preprocessed/
chief_complaint_final/
chief_complaint_embeddings/
elixhauser_scores_output/
matching_variable_tables_output/
matched_cohort_output/
analysis_output_*/
```

Regenerate these locally by running the pipeline scripts in order.

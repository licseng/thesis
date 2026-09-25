-- Finalized cohort construction: stage 4
--
-- Prerequisites:
--   Run 01-Load_and_create_base_tables.sql.
--   Run 02-Create_MHC0_and_MHC1.sql.
--   Run 03-Create_MHC1_psychosis.sql.
--
-- This script creates the two admission-level tables consumed by the Python
-- discharge-note preprocessing and cohort-matching pipeline:
--   * export_only_MHC0
--   * export_MHC1_psychotic
--
-- It also creates the long-format ICD tables used to calculate admission-level
-- Elixhauser scores:
--   * admission_icd_lists_only_MHC0
--   * admission_icd_lists_MHC1_psychotic
--
-- Eligibility retained from the original matching pipeline:
--   * recorded binary sex in MIMIC-IV (F or M); and
--   * age at admission from 18 through 120 years.


-- ---------------------------------------------------------------------------
-- 1. MHC0 matching-source table
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE export_only_MHC0 AS
WITH mhc0_with_matching_fields AS (
    SELECT
        m.subject_id,
        m.hadm_id,
        m.admittime,

        'MHC0' AS cohort,
        0 AS has_prior_psychiatric_history,
        0 AS has_current_secondary_psychiatric_icd,

        pat.gender AS sex,
        pat.anchor_age,
        pat.anchor_year,
        pat.anchor_age
            + (EXTRACT(YEAR FROM m.admittime) - pat.anchor_year)
            AS age_at_admission,

        n.note_id,
        n.charttime,
        n.storetime,
        n.text

    FROM finalized_MHC0 m
    JOIN patients pat
        ON m.subject_id = pat.subject_id
    JOIN discharge n
        ON m.subject_id = n.subject_id
       AND m.hadm_id = n.hadm_id
)

SELECT *
FROM mhc0_with_matching_fields
WHERE sex IN ('F', 'M')
  AND age_at_admission BETWEEN 18 AND 120;


-- ---------------------------------------------------------------------------
-- 2. MHC1-psychosis matching-source table
-- ---------------------------------------------------------------------------

-- This includes every finalized MHC1-psychosis admission, regardless of
-- whether psychosis context came from prior history, a current secondary code,
-- or both. It is therefore not the old history-only MHH1 cohort.
CREATE OR REPLACE TABLE export_MHC1_psychotic AS
WITH mhc1_psychosis_with_matching_fields AS (
    SELECT
        m.subject_id,
        m.hadm_id,
        m.admittime,

        'MHC1_psychotic' AS cohort,
        m.has_prior_psychiatric_history,
        m.has_current_secondary_psychiatric_icd,
        m.mhc1_version,
        m.has_prior_psychosis,
        m.has_current_secondary_psychosis,
        m.psychosis_context_version,

        pat.gender AS sex,
        pat.anchor_age,
        pat.anchor_year,
        pat.anchor_age
            + (EXTRACT(YEAR FROM m.admittime) - pat.anchor_year)
            AS age_at_admission,

        n.note_id,
        n.charttime,
        n.storetime,
        n.text

    FROM finalized_MHC1_psychosis m
    JOIN patients pat
        ON m.subject_id = pat.subject_id
    JOIN discharge n
        ON m.subject_id = n.subject_id
       AND m.hadm_id = n.hadm_id
)

SELECT *
FROM mhc1_psychosis_with_matching_fields
WHERE sex IN ('F', 'M')
  AND age_at_admission BETWEEN 18 AND 120;


-- ---------------------------------------------------------------------------
-- 3. Final pre-matching cohort counts
-- ---------------------------------------------------------------------------

SELECT
    cohort,
    COUNT(*) AS n_rows,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM (
    SELECT cohort, subject_id, hadm_id
    FROM export_only_MHC0

    UNION ALL

    SELECT cohort, subject_id, hadm_id
    FROM export_MHC1_psychotic
) cohorts
GROUP BY cohort
ORDER BY cohort;


-- Expected result: zero rows. Matching inputs must contain exactly one
-- discharge-note row per admission.
SELECT
    cohort,
    subject_id,
    hadm_id,
    COUNT(*) AS n_rows
FROM (
    SELECT cohort, subject_id, hadm_id
    FROM export_only_MHC0

    UNION ALL

    SELECT cohort, subject_id, hadm_id
    FROM export_MHC1_psychotic
) cohorts
GROUP BY cohort, subject_id, hadm_id
HAVING COUNT(*) > 1;


-- Expected result: zero overlapping subjects and admissions.
SELECT
    COUNT(DISTINCT c.subject_id) AS n_overlapping_subjects,
    COUNT(DISTINCT CASE
        WHEN c.hadm_id = e.hadm_id THEN c.hadm_id
    END) AS n_overlapping_admissions
FROM export_only_MHC0 c
JOIN export_MHC1_psychotic e
    ON c.subject_id = e.subject_id;


-- ---------------------------------------------------------------------------
-- 4. Long-format admission ICD tables for Elixhauser scoring
-- ---------------------------------------------------------------------------

-- These tables contain every diagnosis assigned to the selected physical
-- admission, not only psychiatric diagnoses. The Python matching pipeline uses
-- them to calculate admission-level Elixhauser comorbidity scores.
CREATE OR REPLACE TABLE admission_icd_lists_only_MHC0 AS
SELECT DISTINCT
    d.subject_id,
    d.hadm_id,
    d.icd_version,
    d.icd_code
FROM diagnoses_icd d
JOIN export_only_MHC0 c
    ON d.subject_id = c.subject_id
   AND d.hadm_id = c.hadm_id;


CREATE OR REPLACE TABLE admission_icd_lists_MHC1_psychotic AS
SELECT DISTINCT
    d.subject_id,
    d.hadm_id,
    d.icd_version,
    d.icd_code
FROM diagnoses_icd d
JOIN export_MHC1_psychotic c
    ON d.subject_id = c.subject_id
   AND d.hadm_id = c.hadm_id;


-- Every exported admission should be represented in its ICD table because a
-- primary diagnosis was required when the physical-admission base was built.
-- Expected result: zero admissions missing ICD rows in both cohorts.
SELECT
    'MHC0' AS cohort,
    COUNT(*) AS n_exported_admissions_without_icd_rows
FROM export_only_MHC0 c
WHERE NOT EXISTS (
    SELECT 1
    FROM admission_icd_lists_only_MHC0 d
    WHERE d.subject_id = c.subject_id
      AND d.hadm_id = c.hadm_id
)

UNION ALL

SELECT
    'MHC1_psychotic' AS cohort,
    COUNT(*) AS n_exported_admissions_without_icd_rows
FROM export_MHC1_psychotic c
WHERE NOT EXISTS (
    SELECT 1
    FROM admission_icd_lists_MHC1_psychotic d
    WHERE d.subject_id = c.subject_id
      AND d.hadm_id = c.hadm_id
)

ORDER BY cohort;

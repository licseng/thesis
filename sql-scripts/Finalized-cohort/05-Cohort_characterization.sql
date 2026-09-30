-- Finalized cohort analysis: stage 5 - pre-matching characterization
--
-- Prerequisites:
--   Run 01-Load_and_create_base_tables.sql.
--   Run 02-Create_MHC0_and_MHC1.sql.
--   Run 03-Create_MHC1_psychosis.sql.
--
-- This script contains descriptive analyses of the complete eligible
-- pre-matching MHC1-psychosis pool, including its psychosis-code composition.
-- Matched-cohort psychiatric comorbidity and utilization are analyzed by the
-- Python admission- and subject-level characterization scripts after stage 6.
-- Characterization uses the stage-3 cohort excluding subjects with more than
-- 50 total MIMIC-IV hospital admissions.
-- Add further cohort-characterization analyses below.


-- ---------------------------------------------------------------------------
-- 1. MHC1-psychosis cohort description
-- ---------------------------------------------------------------------------

SELECT
    'MHC1_all' AS cohort,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM finalized_MHC1

UNION ALL

SELECT
    'MHC1_psychosis' AS cohort,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM finalized_MHC1_psychosis_excluding_over_50_MIMIC_admissions;


-- Psychosis-source versions are mutually exclusive at admission level. A
-- subject may occur in multiple versions across different admissions.
SELECT
    psychosis_context_version,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM finalized_MHC1_psychosis_excluding_over_50_MIMIC_admissions
GROUP BY psychosis_context_version
ORDER BY psychosis_context_version;


-- Which extended-definition components place admissions in MHC1-psychosis?
-- Multiple groups may be present for one admission, so rows are not additive.
CREATE OR REPLACE TABLE finalized_MHC1_psychosis_diagnostic_groups AS
WITH group_context AS (
    SELECT DISTINCT
        m.subject_id,
        m.hadm_id,
        m.hadm_id AS source_hadm_id,
        m.admittime AS source_admittime,
        d.icd_version,
        d.icd_code,
        p.long_title,
        p.definition_tier,
        p.diagnostic_group,
        'current_secondary' AS context_source
    FROM finalized_MHC1_psychosis_excluding_over_50_MIMIC_admissions m
    JOIN diagnoses_icd d
        ON m.subject_id = d.subject_id
       AND m.hadm_id = d.hadm_id
    JOIN finalized_psychosis_icd_codes_extended p
        ON d.icd_version = p.icd_version
       AND d.icd_code = p.icd_code
    WHERE d.seq_num > 1

    UNION

    SELECT DISTINCT
        m.subject_id,
        m.hadm_id,
        a_previous.hadm_id AS source_hadm_id,
        a_previous.admittime AS source_admittime,
        d_previous.icd_version,
        d_previous.icd_code,
        p.long_title,
        p.definition_tier,
        p.diagnostic_group,
        'prior_history' AS context_source
    FROM finalized_MHC1_psychosis_excluding_over_50_MIMIC_admissions m
    JOIN admissions a_previous
        ON m.subject_id = a_previous.subject_id
       AND a_previous.hadm_id <> m.hadm_id
       AND a_previous.admittime < m.admittime
    JOIN diagnoses_icd d_previous
        ON a_previous.subject_id = d_previous.subject_id
       AND a_previous.hadm_id = d_previous.hadm_id
    JOIN finalized_psychosis_icd_codes_extended p
        ON d_previous.icd_version = p.icd_version
       AND d_previous.icd_code = p.icd_code
)
SELECT *
FROM group_context;

SELECT
    definition_tier,
    diagnostic_group,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions
FROM finalized_MHC1_psychosis_diagnostic_groups
GROUP BY definition_tier, diagnostic_group
ORDER BY definition_tier, n_admissions DESC, diagnostic_group;


-- Exact ICD codes underlying each diagnostic-group count. The target-admission
-- count can exceed the source-admission count because one historical diagnosis
-- can establish psychosis context for several later physical admissions.
SELECT
    definition_tier,
    diagnostic_group,
    icd_version,
    icd_code,
    long_title,
    context_source,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT source_hadm_id) AS n_source_psychosis_admissions,
    COUNT(DISTINCT hadm_id) AS n_MHC1_psychosis_admissions
FROM finalized_MHC1_psychosis_diagnostic_groups
GROUP BY
    definition_tier,
    diagnostic_group,
    icd_version,
    icd_code,
    long_title,
    context_source
ORDER BY
    n_MHC1_psychosis_admissions DESC,
    diagnostic_group,
    icd_version,
    icd_code,
    context_source;


-- Summary of historical carry-forward by diagnostic group.
SELECT
    diagnostic_group,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT source_hadm_id) AS n_source_psychosis_admissions,
    COUNT(DISTINCT hadm_id) AS n_MHC1_psychosis_admissions,
    ROUND(
        1.0 * COUNT(DISTINCT hadm_id)
        / NULLIF(COUNT(DISTINCT source_hadm_id), 0),
        2
    ) AS target_admissions_per_source_admission
FROM finalized_MHC1_psychosis_diagnostic_groups
GROUP BY diagnostic_group
ORDER BY n_MHC1_psychosis_admissions DESC, diagnostic_group;


-- ---------------------------------------------------------------------------
-- 2. Audit of potentially nonphysical AMS/confusion admissions
-- ---------------------------------------------------------------------------

-- This audit is intentionally restricted to cleaned, pre-matching
-- MHC1-psychosis admissions that:
--   * presented with a QuickUMLS concept of altered mental status and/or
--     confusion; and
--   * received a secondary psychosis ICD code in the current admission.
--
-- The imported file contains only subject/admission IDs and boolean/context
-- flags. It contains no chief-complaint or discharge-note text.
--
-- Run the following Python script before this section:
--   python-code/01_discharge_note_preprocessing/02_chief_complaint/
--   01_preprocessing/08_export_ams_confusion_audit_ids.py

@set cohort_audit_dirs=/Users/licseng/Downloads/thesis/thesis_code/python-code/01_discharge_note_preprocessing/02_chief_complaint/01_preprocessing/analysis_output_chief_complaint_final/

CREATE OR REPLACE TABLE finalized_ams_confusion_current_psychosis_audit_ids AS
SELECT *
FROM read_csv_auto(
    '${cohort_audit_dirs}MHC1_psychotic_current_secondary_psychosis_ams_confusion_ids_for_dbeaver.csv'
);


-- Construct an admission-level diagnostic audit. The physical-disease flag is
-- deliberately conservative:
--   * ICD-10 disease/injury chapters A-Q and S-T count as substantive physical
--     diagnoses;
--   * ICD-9 numeric chapters 001-779 count, after psychiatric codes are
--     removed through psychiatric_icd_codes;
--   * ICD-10 R codes and ICD-9 780-799 are symptom/sign codes and do not, by
--     themselves, establish a physical disease;
--   * Z/V/E and other supplementary/external-cause codes do not establish one.
--
-- This is a screening classification, not proof that an admission was or was
-- not genuinely physical.
CREATE OR REPLACE TABLE finalized_ams_confusion_current_psychosis_diagnostic_audit AS
WITH diagnosis_rows AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.has_altered_mental_status_cc,
        a.has_confusion_cc,
        a.has_prior_psychosis,
        a.psychosis_context_version,
        d.seq_num,
        d.icd_version,
        d.icd_code,
        dict.long_title,
        CASE WHEN p.icd_code IS NOT NULL THEN 1 ELSE 0 END
            AS is_psychiatric_code,
        CASE
            WHEN d.icd_version = 10 AND d.icd_code LIKE 'R%' THEN 1
            WHEN d.icd_version = 9
             AND TRY_CAST(SUBSTR(d.icd_code, 1, 3) AS INTEGER)
                 BETWEEN 780 AND 799 THEN 1
            ELSE 0
        END AS is_symptom_or_sign_code,
        CASE
            WHEN d.icd_version = 10
             AND SUBSTR(d.icd_code, 1, 1) BETWEEN 'A' AND 'Q'
             AND p.icd_code IS NULL THEN 1
            WHEN d.icd_version = 10
             AND SUBSTR(d.icd_code, 1, 1) IN ('S', 'T')
             AND p.icd_code IS NULL THEN 1
            WHEN d.icd_version = 9
             AND TRY_CAST(SUBSTR(d.icd_code, 1, 3) AS INTEGER)
                 BETWEEN 1 AND 779
             AND p.icd_code IS NULL THEN 1
            ELSE 0
        END AS is_physical_disease_or_injury_code
    FROM finalized_ams_confusion_current_psychosis_audit_ids a
    JOIN diagnoses_icd d
        ON a.subject_id = d.subject_id
       AND a.hadm_id = d.hadm_id
    LEFT JOIN d_icd_diagnoses dict
        ON d.icd_version = dict.icd_version
       AND d.icd_code = dict.icd_code
    LEFT JOIN psychiatric_icd_codes p
        ON d.icd_version = p.icd_version
       AND d.icd_code = p.icd_code
),
admission_flags AS (
    SELECT
        subject_id,
        hadm_id,
        has_altered_mental_status_cc,
        has_confusion_cc,
        has_prior_psychosis,
        psychosis_context_version,
        MAX(CASE WHEN seq_num = 1 THEN icd_version END)
            AS primary_icd_version,
        MAX(CASE WHEN seq_num = 1 THEN icd_code END)
            AS primary_icd_code,
        MAX(CASE WHEN seq_num = 1 THEN long_title END)
            AS primary_icd_title,
        MAX(CASE WHEN seq_num = 1 THEN is_symptom_or_sign_code ELSE 0 END)
            AS primary_is_symptom_or_sign,
        MAX(
            CASE
                WHEN seq_num = 1 THEN is_physical_disease_or_injury_code
                ELSE 0
            END
        ) AS primary_is_physical_disease_or_injury,
        MAX(
            CASE
                WHEN seq_num = 1
                 AND (
                        lower(COALESCE(long_title, '')) LIKE '%altered mental status%'
                     OR lower(COALESCE(long_title, '')) LIKE '%confusion%'
                     OR lower(COALESCE(long_title, '')) LIKE '%disorientation%'
                 )
                THEN 1 ELSE 0
            END
        ) AS primary_is_ams_confusion_or_disorientation,
        MAX(is_physical_disease_or_injury_code)
            AS has_any_physical_disease_or_injury_code
    FROM diagnosis_rows
    GROUP BY
        subject_id,
        hadm_id,
        has_altered_mental_status_cc,
        has_confusion_cc,
        has_prior_psychosis,
        psychosis_context_version
)
SELECT
    *,
    CASE
        WHEN has_any_physical_disease_or_injury_code = 0
            THEN 1 ELSE 0
    END AS possible_psychiatric_admission_without_physical_disease_code,
    CASE
        WHEN primary_is_symptom_or_sign = 1
         AND has_any_physical_disease_or_injury_code = 1
            THEN 1 ELSE 0
    END AS symptom_primary_with_secondary_physical_disease_code
FROM admission_flags;


-- Main audit result, separated into current-only versus prior-and-current
-- psychosis context. No row-level data are returned.
SELECT
    psychosis_context_version,
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    SUM(primary_is_symptom_or_sign) AS n_primary_symptom_or_sign,
    SUM(primary_is_ams_confusion_or_disorientation)
        AS n_primary_ams_confusion_or_disorientation,
    SUM(primary_is_physical_disease_or_injury)
        AS n_primary_physical_disease_or_injury,
    SUM(symptom_primary_with_secondary_physical_disease_code)
        AS n_symptom_primary_with_secondary_physical_code,
    SUM(has_any_physical_disease_or_injury_code)
        AS n_with_physical_disease_or_injury_code,
    SUM(possible_psychiatric_admission_without_physical_disease_code)
        AS n_without_physical_disease_or_injury_code,
    ROUND(
        100.0 * SUM(possible_psychiatric_admission_without_physical_disease_code)
        / NULLIF(COUNT(DISTINCT hadm_id), 0),
        2
    ) AS pct_without_physical_disease_or_injury_code
FROM finalized_ams_confusion_current_psychosis_diagnostic_audit
GROUP BY psychosis_context_version
ORDER BY psychosis_context_version;


-- Aggregate primary-diagnosis distribution. This identifies which symptom or
-- nonpsychiatric code allowed potentially ambiguous admissions through the
-- physical-admission prefilter.
SELECT
    primary_icd_version,
    primary_icd_code,
    primary_icd_title,
    primary_is_symptom_or_sign,
    primary_is_physical_disease_or_injury,
    has_any_physical_disease_or_injury_code,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    COUNT(DISTINCT subject_id) AS n_subjects
FROM finalized_ams_confusion_current_psychosis_diagnostic_audit
GROUP BY
    primary_icd_version,
    primary_icd_code,
    primary_icd_title,
    primary_is_symptom_or_sign,
    primary_is_physical_disease_or_injury,
    has_any_physical_disease_or_injury_code
ORDER BY n_admissions DESC, primary_icd_version, primary_icd_code;
